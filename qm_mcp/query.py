"""Semantic query over the corpus: embed -> cosine top-k -> grounded answer."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from qm_mcp.config import load_secrets
from qm_mcp.embed import embed_text, synthesize_answer
from qm_mcp.store import CorpusStore

_log = logging.getLogger(__name__)


def _use_llamaindex() -> bool:
    """Return True when the LlamaIndex retrieval backend is opted-in."""
    return os.environ.get("QM_USE_LLAMA_INDEX", "").lower() in (
        "1",
        "true",
        "yes",
    )


async def query(
    question: str,
    *,
    k: int = 6,
    distance_threshold: float = 0.7,
    synthesize: bool = True,
    store: CorpusStore | None = None,
) -> dict[str, Any]:
    """Answer ``question`` from the corpus.

    Returns ``{question, answer, sources:[{id,title,score,source,authors}]}``.
    ``answer`` is None when ``synthesize=False`` (retrieval-only mode).

    Chunks with cosine distance > ``distance_threshold`` are filtered out;
    returns empty sources when no candidate clears the threshold.
    This is the VECTOR_DISTANCE_THRESHOLD pattern from Ch 14 (Gulli 2026) —
    when the corpus genuinely lacks the topic, the honest signal is an empty
    result rather than the k least-bad noise chunks.

    Set env var ``QM_USE_LLAMA_INDEX=true`` to swap the retrieval backend from
    OpenAI text-embedding-3-small + numpy cosine to BAAI/bge-base-en-v1.5
    (local, free) + LlamaIndex SimpleVectorStore. The ``qm_query`` API and
    the distance-threshold semantics are identical in both modes.

    Args:
        question: Natural-language question to answer from the corpus.
        k: Maximum number of candidates to retrieve before threshold filter.
        distance_threshold: Max cosine distance for a match to be included.
            Default 0.7 (conservative). Raise to 0.9 to widen recall.
        synthesize: If True, a grounded answer string is generated from the
            retrieved context. If False, only the source list is returned
            (``answer`` is None). Useful for retrieval-only callers.
        store: CorpusStore override (default constructs from env/default dir).
    """
    load_secrets()
    store = store or CorpusStore()

    if len(store) == 0:
        return {
            "question": question,
            "answer": "The corpus is empty — ingest some research first.",
            "sources": [],
        }

    # ── retrieval backend dispatch ────────────────────────────────────
    if _use_llamaindex():
        hits = await _retrieve_llamaindex(question, k=k, store=store)
    else:
        q_vec = await asyncio.to_thread(embed_text, question)
        hits = store.search(q_vec, k=k)

    # ── VECTOR_DISTANCE_THRESHOLD filter ──────────────────────────────
    # Both backends return cosine similarity (higher = more similar).
    # cosine_distance = 1 − similarity; keep only close-enough matches.
    min_score = 1.0 - distance_threshold
    hits = [(item_id, score) for item_id, score in hits if score >= min_score]

    if not hits:
        _log.info(
            "qm_query: no candidates above threshold %.2f for question %r",
            distance_threshold,
            question[:80],
        )
        return {"question": question, "answer": None, "sources": []}

    # ── build result payload ──────────────────────────────────────────
    sources: list[dict[str, Any]] = []
    contexts: list[dict[str, str]] = []
    for item_id, score in hits:
        rec = store.get(item_id)
        if not rec:
            continue
        sources.append(
            {
                "id": item_id,
                "title": rec.get("title"),
                "score": round(score, 4),
                "source_type": rec.get("source_type"),
                "source": rec.get("source"),
                "authors": rec.get("authors", []),
            }
        )
        contexts.append(
            {
                "title": rec.get("title", "untitled"),
                "source": rec.get("arxiv_id") or rec.get("source", "?"),
                "text": rec.get("full_context") or rec.get("summary", ""),
            }
        )

    answer = None
    if synthesize:
        answer = await asyncio.to_thread(synthesize_answer, question, contexts)

    return {"question": question, "answer": answer, "sources": sources}


async def _retrieve_llamaindex(
    question: str,
    k: int,
    store: CorpusStore,
) -> list[tuple[str, float]]:
    """Dispatch retrieval to the LlamaIndex engine (deferred import)."""
    from qm_mcp.llamaindex.engine import get_engine

    engine = get_engine()
    return await asyncio.to_thread(engine.search, question, k, store)
