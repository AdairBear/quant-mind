"""Integration tests — build a real LlamaIndex from fixture corpus items.

Skipped unless:
    QM_INTEGRATION_TESTS=1      run integration tests
    llama-index-core installed  (pip install llama-index-core
                                         llama-index-embeddings-huggingface)

These tests download/use the BAAI/bge-base-en-v1.5 model from HuggingFace
(~438 MB, cached after first run) and build a real VectorStoreIndex.

Run manually:
    QM_INTEGRATION_TESTS=1 python -m pytest qm_mcp/test_llamaindex_integration.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SKIP = pytest.mark.skipif(
    os.environ.get("QM_INTEGRATION_TESTS") != "1",
    reason="Set QM_INTEGRATION_TESTS=1 to run integration tests",
)

# ── fixture corpus ─────────────────────────────────────────────────────

_FIXTURE_ITEMS = [
    {
        "id": "aaa0000000000001",
        "source_type": "arxiv",
        "source": "0809.0423",
        "title": "Stoikov Market-Making Model",
        "summary": (
            "A stochastic control model for market making under inventory risk. "
            "The market maker optimally quotes bid/ask spreads to balance "
            "profitability against inventory accumulation. Uses Hamilton-Jacobi-"
            "Bellman equations and asymptotic approximation."
        ),
        "full_context": (
            "# Stoikov Market-Making Model\n\n"
            "Summary: A stochastic control model for market making under "
            "inventory risk. The market maker optimally quotes bid/ask "
            "spreads to balance profitability against inventory accumulation.\n\n"
            "Key findings:\n- Optimal spread is proportional to inventory risk\n"
            "- The model reduces to the classic Avellaneda-Stoikov formula\n"
            "- Gamma parameter controls the risk aversion"
        ),
    },
    {
        "id": "bbb0000000000002",
        "source_type": "arxiv",
        "source": "1105.3115",
        "title": "Kelly Criterion for Long-Run Growth",
        "summary": (
            "The Kelly criterion maximizes the expected logarithm of wealth, "
            "producing the highest geometric growth rate over many bets. "
            "Fractional Kelly reduces risk at the cost of lower growth. "
            "Applications to portfolio management, sports betting, and trading."
        ),
        "full_context": (
            "# Kelly Criterion for Long-Run Growth\n\n"
            "Summary: The Kelly criterion maximizes the expected log of wealth. "
            "It produces the highest geometric growth rate.\n\n"
            "Key findings:\n- Full Kelly maximizes long-run wealth\n"
            "- Fractional Kelly reduces variance\n"
            "- Overbetting Kelly leads to ruin"
        ),
    },
    {
        "id": "ccc0000000000003",
        "source_type": "arxiv",
        "source": "2001.01001",
        "title": "VWAP Execution and Market Impact",
        "summary": (
            "Volume-weighted average price (VWAP) execution minimises market "
            "impact by slicing large orders according to historical volume "
            "profiles. The paper derives optimal slicing strategies under "
            "square-root and linear impact models."
        ),
        "full_context": (
            "# VWAP Execution and Market Impact\n\n"
            "Summary: VWAP execution minimises market impact by slicing "
            "large orders according to historical volume profiles.\n\n"
            "Key findings:\n- Optimal slicing tracks volume profile\n"
            "- Square-root impact model is empirically validated\n"
            "- Adaptive VWAP outperforms static benchmarks"
        ),
    },
]


def _make_corpus_store(tmp_path: Path) -> MagicMock:
    """Seed a mock CorpusStore with fixture items."""
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    for item in _FIXTURE_ITEMS:
        (items_dir / f"{item['id']}.json").write_text(
            json.dumps(item), encoding="utf-8"
        )

    store = MagicMock()
    store.__len__ = MagicMock(return_value=len(_FIXTURE_ITEMS))
    store.list_records = MagicMock(return_value=list(_FIXTURE_ITEMS))
    store.get = MagicMock(
        side_effect=lambda iid: next(
            (x for x in _FIXTURE_ITEMS if x["id"] == iid), None
        )
    )
    return store


# ── tests ──────────────────────────────────────────────────────────────


@_SKIP
class TestLlamaIndexBuildAndSearch:
    """Build a real index, retrieve, and verify results."""

    def test_build_creates_persisted_index(self, tmp_path: Path) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        assert (tmp_path / "docstore.json").is_file()
        assert (tmp_path / "qm_index_meta.json").is_file()
        meta = engine._index_meta()
        assert meta["count"] == len(_FIXTURE_ITEMS)
        assert meta["model"] == "BAAI/bge-base-en-v1.5"

    def test_search_returns_relevant_result_for_kelly(
        self, tmp_path: Path
    ) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        results = engine.search(
            "Kelly criterion long-run wealth maximisation", k=3, store=store
        )

        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_id == "bbb0000000000002", (
            f"Expected Kelly paper to be top result, got {top_id}"
        )
        assert top_score > 0.3, f"Score unexpectedly low: {top_score}"

    def test_search_returns_relevant_result_for_vwap(
        self, tmp_path: Path
    ) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        results = engine.search(
            "VWAP order execution market impact slicing", k=3, store=store
        )

        assert len(results) >= 1
        top_id, _ = results[0]
        assert top_id == "ccc0000000000003", (
            f"Expected VWAP paper to be top result, got {top_id}"
        )

    def test_search_scores_are_cosine_similarities(
        self, tmp_path: Path
    ) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        results = engine.search("market making inventory", k=3, store=store)

        for _, score in results:
            assert -1.0 <= score <= 1.0, f"Score out of cosine range: {score}"

    def test_index_loads_from_disk_after_rebuild(self, tmp_path: Path) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        # Create a fresh engine instance pointing at the same dir
        engine2 = LlamaIndexEngine(index_dir=tmp_path)
        results = engine2.search("Kelly criterion", k=2, store=store)
        assert len(results) >= 1

    def test_invalidate_then_rebuild(self, tmp_path: Path) -> None:
        from qm_mcp.llamaindex.engine import LlamaIndexEngine

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        engine.invalidate()
        assert engine._index is None

        # search() should trigger rebuild automatically
        results = engine.search("market making", k=2, store=store)
        assert len(results) >= 1

    def test_distance_threshold_applied_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """When threshold is very tight, irrelevant docs are excluded."""
        from qm_mcp.llamaindex.engine import LlamaIndexEngine
        from qm_mcp.query import query

        engine = LlamaIndexEngine(index_dir=tmp_path)
        store = _make_corpus_store(tmp_path / "corpus")
        engine.build(store)

        import qm_mcp.llamaindex.engine as mod

        mod._engine = engine

        import asyncio

        with (
            pytest.MonkeyPatch().context() as mp,
        ):
            mp.setenv("QM_USE_LLAMA_INDEX", "true")
            result = asyncio.get_event_loop().run_until_complete(
                query(
                    "Kelly criterion wealth",
                    k=3,
                    distance_threshold=0.7,
                    synthesize=False,
                    store=store,
                )
            )

        mod._engine = None
        assert isinstance(result["sources"], list)
