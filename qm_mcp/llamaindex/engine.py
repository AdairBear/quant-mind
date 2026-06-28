"""LlamaIndex retrieval engine — BAAI/bge-base-en-v1.5 local embeddings.

Adapter pattern: presents the same ``(item_id, cosine_similarity)`` interface
as ``CorpusStore.search()`` so ``query.py`` can swap retrieval backends via
the ``QM_USE_LLAMA_INDEX`` env flag without changing the ``qm_query`` API.

No LLM is used. No hosted services. All deps are Apache-2.0 / MIT.

Dependency justifications (Karpathy Rule VIII — every dep is permanent code
you don't control; justify each transitive pull):

``llama-index-core`` (MIT / Apache-2.0, 49k+ GitHub stars):
    Provides SimpleVectorStore (in-memory + disk persistence via JSON) and
    the retrieval pipeline. Transitive pulls: numpy (already in deps),
    pydantic v2 (already in deps), SQLite via stdlib, typing-extensions.
    No network at runtime — only the one-time HuggingFace model download.

``llama-index-embeddings-huggingface`` (Apache-2.0):
    Thin wrapper around sentence-transformers. Accepted because sentence-
    transformers is the de-facto standard for local embedding models; no
    viable lighter alternative that runs BAAI/bge at this quality tier.

``sentence-transformers`` (Apache-2.0) + ``torch`` (BSD):
    torch is large (~2 GB wheel). One-time cost. Zero per-call API cost
    versus OpenAI text-embedding-3-small's per-token billing. CPU-only
    inference on the Mac dev machine is fast enough for <200 corpus items.

``BAAI/bge-base-en-v1.5`` (MIT, ~438 MB):
    768-dim encoder, strong MTEB performance on retrieval tasks. Preferred
    over bge-small-en-v1.5 (384-dim) per task spec — slightly larger model
    for better precision on dense quant-finance vocabulary. Downloaded once
    to ``~/.cache/huggingface/hub/``; cached for all subsequent runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from qm_mcp.store import CorpusStore

_log = logging.getLogger(__name__)

_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
_MARKER_FILE = "qm_index_meta.json"


class LlamaIndexEngine:
    """Local-embedding retrieval engine backed by LlamaIndex SimpleVectorStore.

    Lifecycle:
    - First call to ``search()`` triggers ``_ensure_index()``.
    - If the on-disk index matches the corpus item count, it is loaded.
    - If stale (or absent), ``build()`` is called to re-embed from scratch.
    - ``invalidate()`` clears the in-memory index and the freshness marker so
      the next query triggers a rebuild. Call it after any corpus write when
      ``QM_USE_LLAMA_INDEX`` is active.

    Thread-safety: ``search()`` is synchronous; callers should wrap it in
    ``asyncio.to_thread``.
    """

    def __init__(self, index_dir: Path | None = None) -> None:
        self._index_dir: Path = index_dir or (
            Path.home() / ".quantmind" / "llama_index"
        )
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index: Any = None  # VectorStoreIndex; populated lazily

    # ── freshness bookkeeping ─────────────────────────────────────────

    def _marker_path(self) -> Path:
        return self._index_dir / _MARKER_FILE

    def _index_meta(self) -> dict[str, Any]:
        p = self._marker_path()
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_meta(self, count: int) -> None:
        self._marker_path().write_text(
            json.dumps({"count": count, "model": _EMBED_MODEL}),
            encoding="utf-8",
        )

    def _is_stale(self, store: CorpusStore) -> bool:
        """Return True when the on-disk index doesn't match the corpus state."""
        if not (self._index_dir / "docstore.json").is_file():
            return True
        meta = self._index_meta()
        return (
            meta.get("count", -1) != len(store)
            or meta.get("model") != _EMBED_MODEL
        )

    # ── LlamaIndex global settings ────────────────────────────────────

    @staticmethod
    def _configure_settings() -> None:
        """Set LlamaIndex globals: BAAI embeddings, no LLM."""
        from llama_index.core import Settings  # type: ignore[import-untyped]
        from llama_index.embeddings.huggingface import (  # type: ignore[import-untyped]
            HuggingFaceEmbedding,
        )

        current = getattr(Settings, "embed_model", None)
        if not isinstance(current, HuggingFaceEmbedding):
            _log.info(
                "LlamaIndexEngine: loading embed model %s (first-time HF "
                "download ~438 MB if not cached)",
                _EMBED_MODEL,
            )
            Settings.embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL)
        Settings.llm = None  # retrieval-only; LLM not needed

    # ── index build / load ────────────────────────────────────────────

    def build(self, store: CorpusStore) -> None:
        """(Re)build the index from all CorpusStore items.

        Reads ``full_context`` (rich text with title + abstract + summary +
        key findings) from each record. Items without usable text are skipped.
        The resulting VectorStoreIndex is persisted to ``self._index_dir``.
        """
        from llama_index.core import (
            VectorStoreIndex,  # type: ignore[import-untyped]
        )
        from llama_index.core.schema import (
            Document,  # type: ignore[import-untyped]
        )

        self._configure_settings()

        records = store.list_records(light=False)
        docs: list[Any] = []
        for rec in records:
            text = rec.get("full_context") or rec.get("summary", "")
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={"id": rec["id"]},
                    doc_id=rec["id"],
                )
            )

        _log.info(
            "LlamaIndexEngine.build: indexing %d of %d corpus items",
            len(docs),
            len(records),
        )
        self._index = VectorStoreIndex.from_documents(docs, show_progress=False)
        self._index.storage_context.persist(persist_dir=str(self._index_dir))
        self._save_meta(len(records))
        _log.info(
            "LlamaIndexEngine.build: index persisted to %s", self._index_dir
        )

    def _load_from_disk(self) -> Any:
        from llama_index.core import (  # type: ignore[import-untyped]
            StorageContext,
            load_index_from_storage,
        )

        self._configure_settings()
        sc = StorageContext.from_defaults(persist_dir=str(self._index_dir))
        idx = load_index_from_storage(sc)
        _log.info(
            "LlamaIndexEngine: loaded persisted index from %s",
            self._index_dir,
        )
        return idx

    def _ensure_index(self, store: CorpusStore) -> None:
        """Guarantee ``self._index`` is populated and fresh."""
        if self._index is not None:
            return

        if not self._is_stale(store):
            try:
                self._index = self._load_from_disk()
                return
            except Exception as exc:
                _log.warning(
                    "LlamaIndexEngine: disk load failed (%s) — rebuilding",
                    exc,
                )

        self.build(store)

    def invalidate(self) -> None:
        """Mark the index stale after a corpus write.

        Removes the freshness marker and clears the in-memory index so the
        next ``search()`` call triggers a fresh build.
        """
        p = self._marker_path()
        if p.is_file():
            p.unlink()
        self._index = None
        _log.debug("LlamaIndexEngine: index invalidated")

    # ── retrieval ─────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        k: int,
        store: CorpusStore,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k items by semantic similarity.

        Returns ``[(item_id, cosine_similarity), ...]`` sorted descending —
        the same shape as ``CorpusStore.search()`` so ``query.py`` can swap
        backends transparently.

        ``score`` is the dot-product similarity of L2-normalised vectors
        (≈ cosine similarity ∈ [−1, 1]). Distance-threshold filtering is
        applied by the caller (``query.py``) using ``1 − score``.

        Args:
            query_text: Natural-language question to embed and retrieve for.
            k: Maximum number of candidates to return before threshold filter.
            store: CorpusStore used to build/validate the index if needed.

        Returns:
            List of ``(item_id, score)`` pairs, sorted by score descending.
        """
        self._ensure_index(store)
        if self._index is None:
            return []

        retriever = self._index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query_text)

        results: list[tuple[str, float]] = []
        for node in nodes:
            item_id = node.metadata.get("id")
            if item_id:
                score = float(node.score) if node.score is not None else 0.0
                results.append((item_id, score))
        return results


# ── module-level singleton ────────────────────────────────────────────

_engine: LlamaIndexEngine | None = None


def get_engine(index_dir: Path | None = None) -> LlamaIndexEngine:
    """Return the process-scoped LlamaIndexEngine singleton."""
    global _engine
    if _engine is None:
        _engine = LlamaIndexEngine(index_dir=index_dir)
    return _engine


def invalidate_engine() -> None:
    """Mark the singleton engine stale after a corpus write."""
    global _engine
    if _engine is not None:
        _engine.invalidate()
