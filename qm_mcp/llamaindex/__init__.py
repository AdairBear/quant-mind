"""LlamaIndex retrieval adapter for QuantMind corpus.

Phase 1 — retrieval only. No LLM. No hosted services.

Deps (optional group ``llamaindex`` in pyproject.toml):
    llama-index-core (MIT / Apache-2.0)
    llama-index-embeddings-huggingface (Apache-2.0)
    sentence-transformers (Apache-2.0, wraps torch)
    BAAI/bge-base-en-v1.5 model (~438 MB, MIT, one-time HF download)

Activate via env flag:
    QM_USE_LLAMA_INDEX=true

When unset the existing OpenAI embedding + numpy cosine path is unchanged.
"""
