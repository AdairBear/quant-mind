"""Back-compatibility tests for the existing qm_query interface.

Verifies that without QM_USE_LLAMA_INDEX the behaviour is unchanged:
- distance_threshold default (0.7) works as before
- Custom distance_threshold is respected
- Empty corpus returns "ingest some research first" message
- LlamaIndex is never imported on the default path (no import cost)

Also verifies the flag-dispatch path: when QM_USE_LLAMA_INDEX=true the
LlamaIndex retrieval function is called instead of embed_text+store.search.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from qm_mcp.query import query

_DUMMY_VEC = np.zeros(768, dtype=np.float32)

_DUMMY_RECORD = {
    "title": "Stoikov Market Making",
    "source_type": "arxiv",
    "source": "0809.0423",
    "authors": ["Stoikov"],
    "summary": "A market-making model.",
    "full_context": "# Stoikov Market Making\nSummary: A market-making model.",
}


def _make_store(hits: list[tuple[str, float]]) -> MagicMock:
    store = MagicMock()
    store.__len__ = MagicMock(return_value=max(1, len(hits)))
    store.search = MagicMock(return_value=hits)
    store.get = MagicMock(return_value=_DUMMY_RECORD)
    return store


@pytest.fixture(autouse=True)
def _patch_io(monkeypatch: pytest.MonkeyPatch):
    """Suppress all network/disk I/O and ensure flag is off by default."""
    monkeypatch.delenv("QM_USE_LLAMA_INDEX", raising=False)
    with (
        patch("qm_mcp.query.embed_text", return_value=_DUMMY_VEC),
        patch("qm_mcp.query.synthesize_answer", return_value="Synthesized."),
        patch("qm_mcp.query.load_secrets"),
    ):
        yield


# ── existing API contract ──────────────────────────────────────────────


class TestExistingApiUnchanged:
    """qm_query signature and semantics are unchanged when flag is unset."""

    async def test_default_distance_threshold_applied(self) -> None:
        # score=0.85 → distance=0.15 → under default 0.7 → returned
        store = _make_store([("abc1", 0.85)])
        result = await query("VWAP", store=store)
        assert len(result["sources"]) == 1
        assert result["sources"][0]["id"] == "abc1"

    async def test_distant_match_filtered_at_default_threshold(self) -> None:
        # score=0.25 → distance=0.75 > 0.7 → filtered
        store = _make_store([("xyz9", 0.25)])
        result = await query("Kelly criterion", store=store)
        assert result["sources"] == []
        assert result["answer"] is None

    async def test_custom_threshold_respected(self) -> None:
        # threshold=0.5 → min_score=0.5; score=0.60 passes, score=0.40 doesn't
        store = _make_store([("good", 0.60), ("bad", 0.40)])
        result = await query("Lyapunov", store=store, distance_threshold=0.5)
        ids = [s["id"] for s in result["sources"]]
        assert "good" in ids
        assert "bad" not in ids

    async def test_empty_corpus_returns_ingest_message(self) -> None:
        store = MagicMock()
        store.__len__ = MagicMock(return_value=0)
        result = await query("anything", store=store)
        assert "ingest" in result["answer"].lower()
        assert result["sources"] == []

    async def test_result_structure(self) -> None:
        store = _make_store([("id1", 0.90)])
        result = await query("test", store=store)
        assert set(result.keys()) == {"question", "answer", "sources"}
        assert result["question"] == "test"
        src = result["sources"][0]
        assert "id" in src and "title" in src and "score" in src

    async def test_synthesize_false_skips_answer(self) -> None:
        store = _make_store([("id1", 0.90)])
        result = await query("test", store=store, synthesize=False)
        assert result["answer"] is None
        assert len(result["sources"]) == 1

    async def test_no_candidates_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _make_store([("id1", 0.05)])
        with caplog.at_level(logging.INFO, logger="qm_mcp.query"):
            await query("obscure topic", store=store)
        assert "no candidates above threshold" in caplog.text

    async def test_openai_embed_called_when_flag_off(self) -> None:
        """embed_text should be called when QM_USE_LLAMA_INDEX is not set."""
        store = _make_store([("id1", 0.90)])
        with patch(
            "qm_mcp.query.embed_text", return_value=_DUMMY_VEC
        ) as mock_embed:
            await query("test", store=store)
        mock_embed.assert_called_once()

    async def test_llamaindex_not_called_when_flag_off(self) -> None:
        """_retrieve_llamaindex must never be called when flag is unset."""
        store = _make_store([("id1", 0.90)])
        with patch(
            "qm_mcp.query._retrieve_llamaindex",
            new_callable=AsyncMock,
        ) as mock_li:
            await query("test", store=store)
        mock_li.assert_not_called()


# ── flag dispatch ──────────────────────────────────────────────────────


class TestFlagDispatch:
    """When QM_USE_LLAMA_INDEX=true the LlamaIndex path is used."""

    async def test_llamaindex_called_when_flag_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QM_USE_LLAMA_INDEX", "true")
        store = _make_store([])

        with patch(
            "qm_mcp.query._retrieve_llamaindex",
            new_callable=AsyncMock,
            return_value=[("li_item", 0.88)],
        ) as mock_li:
            store.get = MagicMock(return_value=_DUMMY_RECORD)
            await query("test", store=store)

        mock_li.assert_called_once()

    async def test_openai_embed_not_called_when_flag_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QM_USE_LLAMA_INDEX", "1")
        store = _make_store([])

        with (
            patch(
                "qm_mcp.query._retrieve_llamaindex",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "qm_mcp.query.embed_text", return_value=_DUMMY_VEC
            ) as mock_embed,
        ):
            await query("test", store=store)

        mock_embed.assert_not_called()

    async def test_distance_threshold_applied_to_llamaindex_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QM_USE_LLAMA_INDEX", "yes")
        store = _make_store([])
        store.get = MagicMock(return_value=_DUMMY_RECORD)

        # LlamaIndex returns two results; one below threshold
        with patch(
            "qm_mcp.query._retrieve_llamaindex",
            new_callable=AsyncMock,
            return_value=[("pass", 0.85), ("fail", 0.20)],
        ):
            result = await query("test", store=store, distance_threshold=0.7)

        ids = [s["id"] for s in result["sources"]]
        assert "pass" in ids
        assert "fail" not in ids
