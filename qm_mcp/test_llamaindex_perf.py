"""Performance benchmark — LlamaIndex retrieval vs numpy cosine baseline.

Compares the retrieval step in isolation (embedding is mocked so the benchmark
measures only the search algorithm, not the embedding model latency):

    Baseline:  CorpusStore.search() — numpy cosine over pre-loaded .npy files
    LlamaIndex: LlamaIndexEngine.search() — SimpleVectorStore ANN

Both are run N_QUERIES times over a synthetic corpus of N_DOCS items.
The test asserts LlamaIndex retrieval is within MAX_SLOWDOWN_FACTOR× of numpy.

Skipped unless QM_PERF_TESTS=1.

Run:
    QM_PERF_TESTS=1 python -m pytest qm_mcp/test_llamaindex_perf.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_SKIP = pytest.mark.skipif(
    os.environ.get("QM_PERF_TESTS") != "1",
    reason="Set QM_PERF_TESTS=1 to run performance benchmarks",
)

# Benchmark parameters
N_DOCS = 50
N_QUERIES = 20
TOP_K = 6
DIM = 768  # matches BAAI/bge-base-en-v1.5 output dimension
# LlamaIndex is allowed to be this many times slower than numpy cosine.
# SimpleVectorStore is also O(n) cosine (no ANN indexing at small corpus size)
# so the difference should be small. We give generous headroom for cold-start.
MAX_SLOWDOWN_FACTOR = 10.0


# ── synthetic corpus helpers ───────────────────────────────────────────


def _make_numpy_store(tmp_path: Path, n_docs: int) -> tuple[MagicMock, list]:
    """Seed a CorpusStore-like mock with random numpy vectors on disk."""
    from qm_mcp.store import CorpusStore

    store = CorpusStore(root=tmp_path)
    items = []
    rng = np.random.default_rng(42)
    for i in range(n_docs):
        item_id = f"bench{i:04d}"
        vec = rng.standard_normal(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        rec = {
            "id": item_id,
            "source_type": "text",
            "source": f"bench-{i}",
            "title": f"Benchmark Document {i}",
            "summary": f"This is benchmark document number {i}.",
            "full_context": f"# Document {i}\nSummary: Benchmark item {i}.",
            "authors": [],
        }
        store.add(rec, vec)
        items.append(rec)
    return store, items


def _make_llamaindex_engine(tmp_path: Path, store: MagicMock):
    """Build a LlamaIndex engine with mocked (random) embeddings."""
    from qm_mcp.llamaindex.engine import LlamaIndexEngine

    engine = LlamaIndexEngine(index_dir=tmp_path / "llama_idx")

    # Patch the embedding model so the build step uses deterministic random
    # vectors rather than downloading BAAI/bge-base-en-v1.5.
    rng = np.random.default_rng(42)
    embed_calls = [0]

    def _fake_embed(texts, **kwargs):
        n = len(texts) if hasattr(texts, "__len__") else 1
        embed_calls[0] += n
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()

    class _FakeEmbedModel:
        def get_text_embedding_batch(self, texts, **kwargs):
            return _fake_embed(texts, **kwargs)

        def get_query_embedding(self, query):
            return _fake_embed([query])[0]

        def get_text_embedding(self, text):
            return _fake_embed([text])[0]

    with patch.object(
        LlamaIndexEngine,
        "_configure_settings",
        side_effect=lambda: _patch_settings(_FakeEmbedModel()),
    ):
        engine.build(store)

    return engine


def _patch_settings(embed_model: object) -> None:
    """Apply a fake embed model to LlamaIndex Settings."""
    try:
        from llama_index.core import Settings  # type: ignore[import-untyped]

        Settings.embed_model = embed_model  # type: ignore[assignment]
        Settings.llm = None
    except ImportError:
        pass


# ── benchmark test ─────────────────────────────────────────────────────


@_SKIP
class TestRetrievalPerf:
    """Compare retrieval latency: numpy baseline vs LlamaIndex engine."""

    @pytest.fixture(scope="class")
    def bench_env(self, tmp_path_factory: pytest.TempPathFactory):
        """Build both backends once; reuse across test methods."""
        root = tmp_path_factory.mktemp("bench")
        store, _ = _make_numpy_store(root / "corpus", N_DOCS)
        engine = _make_llamaindex_engine(root, store)
        return {"store": store, "engine": engine}

    def _random_query_vecs(self) -> list[np.ndarray]:
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return list(vecs)

    def test_numpy_baseline_latency(self, bench_env: dict) -> None:
        store = bench_env["store"]
        query_vecs = self._random_query_vecs()

        t0 = time.perf_counter()
        for vec in query_vecs:
            store.search(vec, k=TOP_K)
        elapsed = time.perf_counter() - t0
        per_query_ms = elapsed / N_QUERIES * 1000

        print(
            f"\nnumpy cosine: {N_QUERIES} queries in {elapsed * 1000:.1f} ms "
            f"({per_query_ms:.2f} ms/query)"
        )
        # Sanity: numpy cosine over 50 docs should be < 50 ms per query
        assert per_query_ms < 50.0, (
            f"numpy baseline unexpectedly slow: {per_query_ms:.2f} ms/query"
        )

    def test_llamaindex_latency_within_budget(self, bench_env: dict) -> None:
        store = bench_env["store"]
        engine = bench_env["engine"]
        query_texts = [f"benchmark query number {i}" for i in range(N_QUERIES)]

        # Warm-up: one query to amortize any lazy init
        engine.search(query_texts[0], k=TOP_K, store=store)

        t0 = time.perf_counter()
        for text in query_texts:
            engine.search(text, k=TOP_K, store=store)
        elapsed_li = time.perf_counter() - t0
        per_query_ms_li = elapsed_li / N_QUERIES * 1000

        # Measure numpy for comparison
        query_vecs = self._random_query_vecs()
        t1 = time.perf_counter()
        for vec in query_vecs:
            store.search(vec, k=TOP_K)
        elapsed_np = time.perf_counter() - t1

        ratio = elapsed_li / max(elapsed_np, 1e-9)
        print(
            f"\nLlamaIndex:  {N_QUERIES} queries in {elapsed_li * 1000:.1f} ms "
            f"({per_query_ms_li:.2f} ms/query)\n"
            f"numpy:       {N_QUERIES} queries in {elapsed_np * 1000:.1f} ms\n"
            f"slowdown:    {ratio:.2f}×"
        )
        assert ratio <= MAX_SLOWDOWN_FACTOR, (
            f"LlamaIndex retrieval is {ratio:.2f}× slower than numpy cosine "
            f"(limit {MAX_SLOWDOWN_FACTOR}×). Consider profiling _ensure_index."
        )

    def test_llamaindex_result_count(self, bench_env: dict) -> None:
        """Engine returns at most k results."""
        store = bench_env["store"]
        engine = bench_env["engine"]
        results = engine.search("test query", k=TOP_K, store=store)
        assert len(results) <= TOP_K

    def test_llamaindex_scores_are_floats(self, bench_env: dict) -> None:
        store = bench_env["store"]
        engine = bench_env["engine"]
        results = engine.search("test query", k=TOP_K, store=store)
        for _, score in results:
            assert isinstance(score, float)
