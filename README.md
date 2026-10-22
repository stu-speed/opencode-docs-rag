# docs‑rag — OpenCode Plugin

A self‑contained, prompt‑driven retrieval‑augmented generation (RAG) system for OpenCode.  
Everything lives inside one plugin directory — no MCP, no slash commands, no external services beyond LM Studio for embeddings.

---

## Overview

`docs‑rag` provides:

- Hybrid dense + keyword retrieval
  - FAISS (`IndexIDMap2`, per‑subject)
  - SQLite (metadata, chunk registry, FTS5 keyword index)
- LM Studio embeddings (the main model answers; the tool returns context blocks)
- Local ONNX cross‑encoder reranker (CPU)
- Incremental ingest pipeline (documents + tables → unified rows)
- Natural-language routing via `AGENTS.md` (registered as an OpenCode skill through `package.json`)

Six tools:

| Tool | Purpose | Input |
|------|---------|-------|
| `docs_rag_enable`   | Enable one or more subjects            | `{ "subjects": ["X", ...] }` (required, plural) |
| `docs_rag_disable`  | Disable all subjects                   | `(no arguments)` — clears every session |
| `docs_rag_status`   | Report active subjects                 | `(no arguments)` |
| `docs_rag_ingest`   | Add files to a subject                 | `{ "subject": "X", "files": ["Y.pdf"], "description"?: "..." }` |
| `docs_rag_forget`   | Remove files from a subject            | `{ "subject": "X", "files": ["Y.pdf"] }` (filenames in that subject's raw dir) |
| `docs_rag_query`    | Hybrid search, returns context blocks  | `{ "question": "...", "subject"? }` (singular optional subject) |

The plugin is fully self-contained: code, settings, state, and corpus all live inside the package directory.

---

## Installation

1. Copy (or extract) the built plugin into OpenCode's local plugins dir so that
   `~/.opencode/plugins/docs-rag/` contains a node module with an entry point at its root:

```
~/.opencode/plugins/docs-rag/
├── index.js          # re-exports ./dist/index.js (43-byte stub)
└── dist/, python/, config.json, package.json, AGENTS.md, ...
```

2. That's it — OpenCode v2 auto-discovers plugins in `~/.opencode/plugins/`.  
   No entry is needed in `opencode.json`, no MCP server to configure.

> Legacy (v1) installs registered the plugin via a `"plugins"` array pointing at
> an older path; remove that if migrating — it's ignored by v2 discovery.

### Requirements

- `python3` with:
  - `faiss-cpu`
  - `pandas`
  - `openpyxl`
  - `markitdown`
  - `onnxruntime`
- Optional (recommended for PDFs):
  - `marker-pdf` (`marker_single --mode fast`)
- LM Studio reachable for embeddings (URL/model/token in `config.json`)

Activate your venv before launching OpenCode so the plugin inherits the correct interpreter.

---

## Directory Layout

```
docs-rag/
├── index.js              # entry stub → dist/index.js
├── dist/                 # compiled plugin (index.js, tools.js, router.js)
├── src/                  # TypeScript sources (dev only)
├── python/               # ingest/query/forget engines + rag helpers
├── AGENTS.md             # skill: routing + phrasing rules (auto-registered via package.json "opencode.skills")
├── config.json           # LM Studio + search/rerank settings (chmod 600)
└── data/                 # persistent corpus, indexes, and runtime state
    ├── subjects/<subject>/
    │   ├── raw/          # source files as ingested (markdown/pdf/etc.)
    │   ├── json/         # normalized rows per file (<name>.md.json)
    │   └── index/        # FAISS index for this subject
    ├── models/<model>/   # cached ONNX reranker weights
    └── db/rag.db         # SQLite store (chunks, FTS5, metadata)
```

Active-subject state is kept in the plugin's own storage (`ctx.storage`, key
`active_subjects`) — **it persists across sessions and restarts**. There is no
per-startup reset: once enabled, RAG stays ON until `docs_rag_disable`.

---

## Using the Plugin

Natural language drives everything.  
The skill in `AGENTS.md` maps phrases to tools (store-mutating phrases require the noun **“docs”**).

### Enable / Disable / Status

| Phrase | Tool call |
|--------|-----------|
| “enable docs X”   | `docs_rag_enable { "subjects": ["X"] }`  — plural array, one or more |
| “disable docs”    | `docs_rag_disable {}`                    — no args; clears **all** subjects for every session |
| “docs status”     | `docs_rag_status {}`                     |

### Ingest / Forget

| Phrase | Tool call |
|--------|-----------|
| “ingest docs the-report.pdf into reports”   | `docs_rag_ingest { "subject": "reports", "files": ["/path/to/the-report.pdf"] }` — files are copied into the subject's raw dir; optional `"description"` is stored as doc metadata |
| “forget docs the-report.pdf from reports”   | `docs_rag_forget { "subject": "reports", "files": ["the-report.pdf"] }` — matches by filename in that subject's raw dir (index updated first, then file removed) |

### Query

Search all active subjects — omit `"subject"` entirely. Or pass the **singular, optional** key to narrow one subject (e.g. `{ "subject": "fshd", "question": "..." }`). Note: `additionalProperties` is rejected by every tool call — only documented keys are accepted (`subjects`, plural, exists on `_enable`; query takes singular `"subject"`).

The main model answers using the returned context blocks; with no active
subjects and no explicit `"subject"` it returns an error telling you to enable one first.

---

## Ingest Pipeline

### Stage 1 — Normalize Input

Documents → markdown  
Tables → DataFrame rows

Supported formats:

- Documents: `.md`, `.txt`, `.rst`, `.tex`, `.adoc`, `.org`, `.log`, `.pdf`, `.docx`, `.pptx`, `.html`, `.epub`
- Tables: CSV/TSV, Excel, parquet/feather/ORC, pickle, JSON records

PDFs use:

1. `marker_single --mode fast`
2. fallback: `pymupdf4llm`
3. fallback: `markitdown`

### Stage 2 — Unified Rows

Everything becomes rows stored in SQLite + FAISS:

- `id` = stable content hash  
- `text` = JSON with section or record content  
- `doc_meta` = leading-context metadata (title, abstract, or table columns)

Sections > `chunk_chars` (default 2000) are split; code fences are preserved.
Re-ingesting edited markdown updates only changed chunks (incremental by hash).

---

## Tools — Behavior Notes

### `docs_rag_query`
Hybrid search:

1. Resolve subjects → explicit `"subject"`, else all active  
2. Validate subject embedding model matches config  
3. Embed question via LM Studio  
4. Dense FAISS search (per-subject index, pooled to `pool_k`)  
5. Keyword BM25 search (FTS5) when enabled  
6. Reciprocal rank fusion over the pool  
7. Optional ONNX cross-encoder rerank of top candidates  
8. Doc-hit coverage sort; return up to `top_k` context blocks

### `docs_rag_enable` / `_disable` / `_status`
Manage the persisted `active_subjects` list: enable is a set (deduped, validated), disable removes it entirely — there is no per-subject disable in v2. Subject names must match `[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}` and exist under `data/subjects`.

### `docs_rag_ingest`
Copies sources into the subject's raw dir (refuses to overwrite), then runs the Python ingest engine; returns per-file results. Duplicate paths in one call are rejected.

### `docs_rag_forget`
Updates indexes first, deletes matched files only after success — a failed forget leaves your corpus intact.

---

## Query Behavior

- RAG ON (active subjects) + question → answer from retrieved blocks  
  - No “the corpus says…” framing; cite sources only when provenance matters  
  - Insufficient context → fill from own knowledge or the web  
  - Nothing answers it → say so
- Tasks/instructions that don't need docs → normal conversation, no tool call

---

## Retrieval Internals

### Dense
FAISS `IndexIDMap2` over normalized cosine similarity.

### Keyword
FTS5 BM25 with Porter stemming.  
Rare terms anchor the pool; fallback to OR query.

### Fusion
Reciprocal rank fusion (RRF).

### Rerank
Local ONNX cross-encoder (`ms-marco-MiniLM-L6-v2`).  
Cached under `data/models/<model>/`.

### Doc-hits
Coverage-first stable sort by query-term presence in chunk body + metadata.

---

## SQLite Schema

- `subjects`
- `documents`
- `chunks`
- `chunk_ids`
- `chunks_fts` (FTS5)

Indexes are written atomically.

---

## Configuration (`config.json`)

Single settings source:

```jsonc
{
  "lmstudio": {
    "url": "...",
    "embed_model": "...",
    "token": "..."
  },
  "data_dir": "data",
  "search": { "top_k": 8, "pool_k": 50, "rrf_k": 60, "use_fts": true },
  "rerank": {
    "enabled": true,
    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "candidates": 30,
    "max_tokens": 256
  },
  "http_timeout": 120,
  "embed_batch": 256,
  "chunk_chars": 2000,
  "doc_meta": {
    "max_chars": 1500,
    "preamble_sections": ["Abstract"],
    "per_subject": {}
  }
}
```

Embedding model must match the one used at ingest time.  
`config.json` holds an API token — keep it `chmod 600`.

---

## Development & Packaging (dev folder)

The dev checkout keeps TypeScript sources plus a Makefile pipeline:

- `make build` / `make all` → npm ci, tsc into `dist/`, then JS validation + smoke test (`test/sanity.js`)
- `make package` → clean stage copy of the shippable files (index stub, docs, config, src/, python/) without dev artifacts, installs deps in staging, builds there, drops sources from the staged build, and produces `dist/docs-rag-X.Y.Z.tgz` via npm pack

To deploy a new tarball: extract it over your existing
`~/.opencode/plugins/docs-rag/`, keeping its `data/` directory (corpus + indexes) intact.

---

## Limitations

- PDF conversion quality varies with layout; scanned PDFs need OCR first.
- Word “Title” style is not a heading; use Heading 1.
- Legacy `.doc` unsupported.
- Deleting a subject's index breaks queries until re-ingest.
- ONNX reranker downloads model on first use; offline → rerank skipped.
