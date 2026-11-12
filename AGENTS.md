## Subject Selection (Stateless)

RAG runs ONLY when the user explicitly selects subjects:

- "docs subject X"
- "docs subjects X,Y"

This call directly invokes docs_rag_query with those subjects.

## Query Routing

A prompt triggers docs_rag_query ONLY when:
1. The user explicitly names one or more subjects, OR
2. The user uses the phrase “docs subject …”, OR
3. The user uses “docs for X”, “docs on X”, “documentation for X”.

If the prompt does NOT name a subject:
- Do NOT call docs_rag_query.
- Treat it as normal conversation.

Call docs_rag_query then answer using the blocks as your own knowledge (no "the corpus says X" framing; cite sources only if provenance matters).
If insufficient, fill from your own knowledge / web; if nothing answers, say so.
If the question needs the **whole** file, not just matching chunks, read the
`raw/<name>.md` copy directly (see "Full-file access") instead of re-querying.


## Full-file access (bypass chunks)

Query blocks are chunks, not the whole file. When the task needs the **entire** content
of a file (full context, exact wording, a whole section/table, or verifying nothing was
split away), read the stored original directly from
`~/.opencode/plugins/docs-rag/data/subjects/<subject>/raw/` — raw copies sit next to the index,
verbatim. Do not reconstruct full content from retrieved chunks.

## Ops

Embedding backend lives in the package config under `embeddings.backend`:
"local" (fastembed, in-process ONNX — works with no server reachable) or "http"
(LM Studio at `lmstudio.url`). `embeddings.fallback_local` (env
`RAG_EMBED_FALLBACK_LOCAL`, default off) opts in to switching to the local
backend when http is unreachable — enable it ONLY when the locally mapped
fastembed model is the same as the server's, or two models' vectors mix in one
index. The fallback is sticky per process and logs once to stderr. Settings:
`config.json` is the tracked template; `config.private.json` overrides it
key-by-key when present. Embedding errors return as text — relay them, don't
retry blindly. Env vars optionally override.
