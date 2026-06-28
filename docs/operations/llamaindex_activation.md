# LlamaIndex Backend — Activation Guide

PR #1 (`feat/llamaindex-phase-1`) adds an optional LlamaIndex retrieval backend.
When active it swaps OpenAI `text-embedding-3-small` for BAAI/bge-base-en-v1.5
(local, free, MIT) via FastEmbed/ONNX. The `qm_query` MCP tool API is identical
in both modes.

## Install

```bash
uv pip install -e ".[llamaindex]"
```

What this pulls in (beyond the standard deps):

| Package | Runtime | Model |
|---|---|---|
| `llama-index-core` | SimpleVectorStore + retrieval | — |
| `llama-index-embeddings-fastembed` | ONNX bridge | — |
| `fastembed` | ONNX Runtime | BAAI/bge-base-en-v1.5 (~130 MB, downloaded once) |

The ONNX model is cached at `~/.cache/fastembed/` after the first run.
No torch dependency; no numpy ABI conflicts.

## Activate

```bash
export QM_USE_LLAMA_INDEX=true
```

Or set it in `.env`:

```
QM_USE_LLAMA_INDEX=true
```

## Smoke test

```bash
QM_USE_LLAMA_INDEX=true qm_mcp query "limit order book market making"
```

Expected output (first run downloads the ONNX model and builds the index):

```
LlamaIndexEngine: loading embed model BAAI/bge-base-en-v1.5 via FastEmbed/ONNX
LlamaIndexEngine.build: indexing 33 of 33 corpus items
LlamaIndexEngine.build: index persisted to /Users/thomasadair/.quantmind/llama_index
```

Subsequent runs load from disk — no re-embedding unless the corpus changes.

### Verified output (2026-06-28, 33-item corpus)

```
sources returned: 5
  score=0.7748  High-frequency trading in a limit order book
  score=0.7555  Dealing with the Inventory Risk. A solution to the market making …
  score=0.7540  Limit Order Strategic Placement with Adverse Selection Risk …
  score=0.7525  Adaptive Optimal Market Making Strategies with Inventory Liquidation …
  score=0.7487  Dealing with the Inventory Risk. A solution to the market making …
```

## Index persistence

The index lives at `~/.quantmind/llama_index/`. Files written on first build:

```
docstore.json
default__vector_store.json
index_store.json
graph_store.json
image__vector_store.json
qm_index_meta.json   ← freshness marker: {"count": 33, "model": "BAAI/bge-base-en-v1.5"}
```

The engine is stale when `qm_index_meta.json` doesn't exist or the corpus item
count changes. A stale index is rebuilt automatically on the next `qm_query` call.

## Distance threshold

Both backends share the same `distance_threshold` parameter (Ch 14 Gulli 2026):

```
min_cosine_similarity = 1.0 - distance_threshold
```

| threshold | min_similarity | behaviour |
|---|---|---|
| 0.7 (default) | 0.30 | permissive — returns weak matches in the same domain |
| 0.4 | 0.60 | moderate — filters unrelated papers |
| 0.25 | 0.75 | strict — only strong matches pass |

At default threshold, off-domain queries (e.g. "Kelly criterion" against a
microstructure corpus) still return weak matches (~0.53). Tighten to 0.4–0.25
for an honest empty result when the corpus lacks the topic.

## Platform notes

**Intel macOS (x86_64):** torch dropped Intel macOS wheels after 2.2.2. The
`sentence-transformers` path (original design) fails with
`RuntimeError: Numpy is not available` at embedding time because torch 2.2.2
was compiled against NumPy 1.x but the project requires numpy≥2. FastEmbed/ONNX
sidesteps this entirely — no torch, no numpy ABI issue.

**Apple Silicon / Linux / Windows:** FastEmbed ONNX still runs correctly. If you
prefer the PyTorch path (for GPU acceleration or custom fine-tuning), you can
swap back to `llama-index-embeddings-huggingface` + `sentence-transformers` on
platforms where torch≥2.4 is available.

## Rollback

```bash
unset QM_USE_LLAMA_INDEX
```

Or remove it from `.env`. The default OpenAI backend is restored immediately
with no corpus changes required. The LlamaIndex index at `~/.quantmind/llama_index/`
is inert when the flag is off and can be deleted freely.
