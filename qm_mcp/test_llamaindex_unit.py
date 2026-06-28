"""Unit tests for LlamaIndexEngine — no real LlamaIndex or embeddings needed.

All LlamaIndex imports are deferred inside method bodies. These tests exercise:
- Freshness logic (_is_stale, _save_meta, _index_meta, invalidate)
- search() with a mock index injected directly onto _index
- Module-level singleton helpers (get_engine, invalidate_engine)
- _maybe_invalidate_llamaindex in ingest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qm_mcp.llamaindex.engine import (
    LlamaIndexEngine,
    get_engine,
    invalidate_engine,
)

# ── helpers ────────────────────────────────────────────────────────────


def _make_store(n_items: int) -> MagicMock:
    store = MagicMock()
    store.__len__ = MagicMock(return_value=n_items)
    store.list_records = MagicMock(
        return_value=[
            {
                "id": f"item{i:02d}",
                "full_context": f"Context for item {i}",
                "summary": f"Summary {i}",
            }
            for i in range(n_items)
        ]
    )
    return store


def _make_node(item_id: str, score: float) -> MagicMock:
    node = MagicMock()
    node.metadata = {"id": item_id}
    node.score = score
    return node


# ── freshness / marker tests ───────────────────────────────────────────


class TestFreshnessMarker:
    """Freshness marker: _is_stale, _save_meta, _index_meta roundtrips."""

    def test_is_stale_when_no_docstore(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_store(3)
        assert engine._is_stale(store)

    def test_is_stale_when_count_differs(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        (tmp_path / "docstore.json").write_text("{}")
        engine._save_meta(count=2)
        store = _make_store(5)
        assert engine._is_stale(store)

    def test_is_stale_when_model_differs(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        (tmp_path / "docstore.json").write_text("{}")
        # Write meta with a different model name
        (tmp_path / "qm_index_meta.json").write_text(
            json.dumps({"count": 3, "model": "BAAI/bge-small-en-v1.5"})
        )
        store = _make_store(3)
        assert engine._is_stale(store)

    def test_not_stale_when_fresh(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        (tmp_path / "docstore.json").write_text("{}")
        engine._save_meta(count=4)
        store = _make_store(4)
        assert not engine._is_stale(store)

    def test_save_and_load_meta_roundtrip(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine._save_meta(7)
        meta = engine._index_meta()
        assert meta["count"] == 7
        assert meta["model"] == "BAAI/bge-base-en-v1.5"

    def test_index_meta_returns_empty_on_missing_file(
        self, tmp_path: Path
    ) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        assert engine._index_meta() == {}

    def test_index_meta_returns_empty_on_corrupt_file(
        self, tmp_path: Path
    ) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine._marker_path().write_text("not json", encoding="utf-8")
        assert engine._index_meta() == {}


# ── invalidate ─────────────────────────────────────────────────────────


class TestInvalidate:
    """invalidate() clears marker and in-memory index."""

    def test_invalidate_removes_marker(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine._save_meta(3)
        assert engine._marker_path().is_file()
        engine.invalidate()
        assert not engine._marker_path().is_file()

    def test_invalidate_clears_in_memory_index(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine._index = MagicMock()  # pretend index is loaded
        engine.invalidate()
        assert engine._index is None

    def test_invalidate_is_idempotent(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine.invalidate()  # marker already absent
        engine.invalidate()  # second call is safe


# ── search with injected mock index ────────────────────────────────────


class TestSearch:
    """search() with a mock index injected directly onto _index."""

    def test_search_returns_id_score_pairs(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_node("abc1", 0.92),
            _make_node("xyz9", 0.45),
        ]
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        engine._index = mock_index

        store = _make_store(2)
        results = engine.search("VWAP strategy", k=5, store=store)

        assert results == [("abc1", 0.92), ("xyz9", 0.45)]
        mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
        mock_retriever.retrieve.assert_called_once_with("VWAP strategy")

    def test_search_skips_nodes_without_id(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        node_no_id = MagicMock()
        node_no_id.metadata = {}  # no "id" key
        node_no_id.score = 0.88
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node_no_id]
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        engine._index = mock_index

        results = engine.search("Kelly criterion", k=3, store=_make_store(1))
        assert results == []

    def test_search_handles_none_score(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        node = MagicMock()
        node.metadata = {"id": "item00"}
        node.score = None
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [node]
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        engine._index = mock_index

        results = engine.search("test", k=3, store=_make_store(1))
        assert results == [("item00", 0.0)]

    def test_search_empty_when_no_index_and_empty_store(
        self, tmp_path: Path
    ) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        # Empty store → build() will produce an index with 0 docs.
        # Patch build to avoid real LlamaIndex calls.
        store = _make_store(0)
        with patch.object(engine, "build") as mock_build:
            mock_build.side_effect = lambda s: setattr(engine, "_index", None)
            results = engine.search("test", k=3, store=store)
        assert results == []


# ── ensure_index dispatch ──────────────────────────────────────────────


class TestEnsureIndex:
    """_ensure_index dispatch: in-memory → disk load → rebuild."""

    def test_uses_in_memory_index_when_already_loaded(
        self, tmp_path: Path
    ) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        sentinel = MagicMock()
        engine._index = sentinel

        with (
            patch.object(engine, "build") as mock_build,
            patch.object(engine, "_load_from_disk") as mock_load,
        ):
            engine._ensure_index(_make_store(3))
            mock_build.assert_not_called()
            mock_load.assert_not_called()

        assert engine._index is sentinel

    def test_loads_from_disk_when_fresh(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        (tmp_path / "docstore.json").write_text("{}")
        engine._save_meta(3)

        loaded_idx = MagicMock()
        with (
            patch.object(
                engine, "_load_from_disk", return_value=loaded_idx
            ) as mock_load,
            patch.object(engine, "build") as mock_build,
        ):
            engine._ensure_index(_make_store(3))
            mock_load.assert_called_once()
            mock_build.assert_not_called()

        assert engine._index is loaded_idx

    def test_rebuilds_when_disk_load_fails(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        (tmp_path / "docstore.json").write_text("{}")
        engine._save_meta(3)
        store = _make_store(3)

        with (
            patch.object(
                engine,
                "_load_from_disk",
                side_effect=RuntimeError("corrupt"),
            ),
            patch.object(engine, "build") as mock_build,
        ):
            engine._ensure_index(store)
            mock_build.assert_called_once_with(store)

    def test_builds_when_stale(self, tmp_path: Path) -> None:
        engine = LlamaIndexEngine(index_dir=tmp_path)
        # No docstore.json → always stale
        store = _make_store(2)

        with patch.object(engine, "build") as mock_build:
            engine._ensure_index(store)
            mock_build.assert_called_once_with(store)


# ── module-level singleton ─────────────────────────────────────────────


class TestSingleton:
    """Module-level singleton: get_engine and invalidate_engine."""

    def test_get_engine_returns_same_instance(self) -> None:
        import qm_mcp.llamaindex.engine as mod

        mod._engine = None  # reset
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2
        mod._engine = None  # cleanup

    def test_invalidate_engine_calls_engine_invalidate(
        self, tmp_path: Path
    ) -> None:
        import qm_mcp.llamaindex.engine as mod

        engine = LlamaIndexEngine(index_dir=tmp_path)
        engine._save_meta(5)
        mod._engine = engine

        invalidate_engine()

        assert not engine._marker_path().is_file()
        assert engine._index is None
        mod._engine = None  # cleanup

    def test_invalidate_engine_is_noop_when_no_engine(self) -> None:
        import qm_mcp.llamaindex.engine as mod

        mod._engine = None
        invalidate_engine()  # should not raise


# ── ingest invalidation hook ───────────────────────────────────────────


class TestIngestInvalidation:
    """Verify _maybe_invalidate_llamaindex in ingest.py fires correctly."""

    def test_invalidates_when_flag_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # invalidate_engine is imported INSIDE _maybe_invalidate_llamaindex,
        # so the correct patch target is the function in its source module.
        monkeypatch.setenv("QM_USE_LLAMA_INDEX", "true")
        with patch("qm_mcp.llamaindex.engine.invalidate_engine") as mock_inv:
            from qm_mcp.ingest import _maybe_invalidate_llamaindex

            _maybe_invalidate_llamaindex()
            mock_inv.assert_called_once()

    def test_noop_when_flag_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("QM_USE_LLAMA_INDEX", raising=False)
        with patch("qm_mcp.llamaindex.engine.invalidate_engine") as mock_inv:
            from qm_mcp.ingest import _maybe_invalidate_llamaindex

            _maybe_invalidate_llamaindex()
            mock_inv.assert_not_called()

    def test_silent_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Verify the except-swallow works: RuntimeError from the engine must
        # not propagate out of _maybe_invalidate_llamaindex.
        monkeypatch.setenv("QM_USE_LLAMA_INDEX", "1")
        with patch(
            "qm_mcp.llamaindex.engine.invalidate_engine",
            side_effect=RuntimeError("simulated failure"),
        ):
            from qm_mcp.ingest import _maybe_invalidate_llamaindex

            _maybe_invalidate_llamaindex()  # must not raise
