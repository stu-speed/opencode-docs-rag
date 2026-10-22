# docs‑rag — OpenCode Plugin v2

Self-contained, prompt-driven RAG over your local document corpus: no MCP server and no external services beyond LM Studio for embeddings. Code, settings, state, and the whole corpus live inside one plugin directory under `~/.opencode/plugins/docs-rag/`.

## What it provides

- Per-subject hybrid retrieval — dense FAISS (`IndexIDMap2`) + SQLite FTS5 BM25 keyword search, fused with reciprocal rank fusion
- Optional local ONNX cross‑encoder rerank (CPU; weights cached under `data/models/<model>/`, skipped offline)
- Incremental ingest pipeline: documents → markdown sections, tables → rows — all stored as unified chunks keyed by content hash. Re-ingesting an edited file updates only changed chunks

## Tools

Three tools in v2. All reject undocumented keys (`additionalProperties: false`). Subject names must match `[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}` and exist under `data/subjects`.

| Tool | Purpose | Input |
|------|---------|-------|
| `docs_rag_ingest`  | Add files to a subject      | `{ "subject": "x", "files": ["/abs/path/a.pdf"], "description"? }` — sources are copied into `<data>/subjects/x/raw/`; refuses overwrites, rejects duplicate paths in one call; optional `"description"` is stored as document metadata. Returns per-file results. |
| `docs_rag_forget`  | Remove files from a subject | `{ "subject": "x", "files": ["a.pdf"] }` — matches by filename inside that subject's raw dir (no paths). Index updated first; the file is deleted only on success, so a failed forget leaves your corpus intact. |
| `docs_rag_query`   | Hybrid search → context blocks | `{ "question": "...", "subject"? }`, or `"subjects"?: ["x","y"]`. **At least one subject is required** — singular string or plural array; neither present returns an error telling you to name subjects. The main model answers from the returned context blocks (up to `top_k`). |

v2 selection is stateless: v1's enable/disable/status tools and persisted *active-subjects* list are gone, so every query names its subject(s) explicitly — which also keeps RAG out of prompts that don't ask for it.

## Installation & Requirements

Copy the built plugin into OpenCode's local plugins dir with an entry point at its root:

```
~/.opencode/plugins/docs-rag/
├── index.js          # re-exports ./dist/index.js (43-byte stub)
└── dist/, python/, config.json, package.json, AGENTS.md, ...
```

That's it — OpenCode auto-discovers `~/.opencode/plugins/*/`. No entry in `opencode.json`, no MCP server. The natural-language routing skill is registered via `"opencode.skills"` in `package.json` (points at `AGENTS.md`). Legacy v1 installs that listed the plugin under a `"plugins"` array can drop it — ignored by discovery.

Python 3 with:
- Required: `faiss-cpu`, `pandas`; plus `openpyxl` for Excel tables, and `onnxruntime` (rerank) if enabled in config
- PDFs: optional but recommended `marker-pdf`. Conversion chain per file: `marker_single --mode fast` → fallback `pymupdf4llm` → last resort `markitdown`; legacy `.doc` is best-effort plain text only

LM Studio must be reachable at the URL in `config.json`. Activate your venv before launching OpenCode so the plugin inherits the correct interpreter.

## Directory Layout

```
docs-rag/
├── index.js              # entry stub → dist/index.js (tools + Python router)
├── dist/                 # compiled plugin (index, tools, router)
├── python/               # ingest/query/forget engines + rag helpers; each tool spawns `python3 <name>.py` with JSON args (128 KiB output cap per call)
├── AGENTS.md             # skill: subject-selection routing rules (auto-registered via package.json "opencode.skills")
├── config.json           # LM Studio + search/rerank settings — holds an API token, keep chmod 600
└── data/                 # persistent corpus, indexes; survives plugin upgrades if you preserve it on deploy
    ├── subjects/<subject>/raw|json|index   # source files as ingested · normalized rows (<name>.md.json) · FAISS index for this subject
    ├── models/<model>/                     # cached ONNX reranker weights
    └── db/rag.db                           # SQLite store: subjects, documents, chunks (+ FTS5 virtual table `chunks_fts`)
```

There is no runtime state outside `data/` — v1's persisted active-subjects key was removed with the enable/disable tools.

## Usage & Routing (per AGENTS.md)

RAG runs only when a subject is explicitly selected in your prompt:

- “docs subject X” / “docs subject X,Y” → query those subjects
- “… docs for/on/for <subject> …”, or naming one or more subjects anywhere, also triggers `docs_rag_query` with them
- No named subject (e.g. “how do I inject METAR”) → **no** tool call; normal conversation

Ingest and forget are store-mutating: use direct requests (“add this PDF to the xplane docs”, “remove a.md from reports”). `docs_rag_ingest` copies your files into the subject's raw dir first, so originals elsewhere stay untouched.

## Query Pipeline (per named subject)

1. Embed question via LM Studio; each subject's index is validated against config's embedding model before searching
2. Dense FAISS search per-subject, pooled to `pool_k` candidates
3. Keyword BM25 search (FTS5, Porter stemming; rare terms anchor the pool with OR fallback) when `use_fts` — then reciprocal rank fusion over the combined pools
4. Optional ONNX cross‑encoder rerank of top `candidates`, and doc-hit coverage sort by query-term presence in chunk body + metadata
5. Return up to `top_k` context blocks

Answering etiquette (from the skill): cite sources only when provenance matters; if retrieved context is insufficient, fill from own knowledge or the web — say so outright when nothing answers it.

## Ingest Pipeline

- **Stage 1 — normalize:** documents → markdown (`.md`, `.txt` read directly), tables → DataFrame rows
  - Documents: `.rst`, `.tex`, `.adoc`, `.org`, `.log`, `.pdf`, `.docx`, `.pptx`, `.html`, `.epub` via markitdown; PDFs through the chain above
  - Tables: CSV/TSV, Excel (openpyxl), parquet/feather/ORC, pickle, JSON records — via pandas
- **Stage 2 — unified rows:** every section or record becomes a row in SQLite + FAISS with `id` = stable content hash, `text` = the section/record as JSON, and leading-context metadata (title, abstract, table columns). Sections longer than `chunk_chars` are split; code fences preserved.

## Configuration (`config.json`)

Single settings source:

```jsonc
{
  "lmstudio": { "url": "...", "embed_model": "...", "token": "..." },
  "data_dir": "data",
  "search":   { "top_k": 8, "pool_k": 50, "rrf_k": 60, "use_fts": true },
  "rerank":   { "enabled": true, "model": "cross-encoder/ms-marco-MiniLM-L6-v2", "candidates": 30, "max_tokens": 256 },
  "http_timeout": 120,       // seconds for LM Studio calls (engine side); Python router itself times out at 30 min per call
  "embed_batch": 256,        // chunks embedded per request during ingest/query
  "chunk_chars": 2000,       // split threshold in stage 2
  "doc_meta":   {            // leading-context metadata captured for each document
    "max_chars": 1500,
    "preamble_sections": ["Abstract"],
    "per_subject": {}        // per-subject overrides of the above
  }
}
```

The embedding model must match what was used at ingest time — changing it invalidates existing indexes. `config.json` holds an API token: keep it `chmod 600`.

## Development & Packaging

The dev checkout adds TypeScript sources (`src/*.ts`) plus a Makefile pipeline; development builds go into this folder's own `dist/`.

- **`make all`** (default) — build + test in the repo root: `npm ci`, then `tsc` → local `dist/`; JS validation that `dist/index.js` loads with a default export whose `setup()` is callable, followed by the smoke test (`node test/sanity.js`)
- **`make build`** / **`make test`** — either half separately (test depends on build)

### Packaging → tarball

```sh
make package    # = clean stage stage-install stage-build stage-pack prune
```

1. `clean` wipes the local `dist/` and any stale staging dir
2. `stage` rsyncs only shippable files (`index.js`, docs, config.json, package*.json, tsconfig.json, src/, python/) into a temp copy with Python caches stripped — no corpus/state leaves this folder
3. inside the staged copy: full install via `npm ci`, TypeScript build, then **sources are deleted** (the tarball ships compiled JS + engines only)
4. `stage-pack` runs `npm pack` there and moves the artifact to local `dist/`; staging dir removed

Artifact: `docs-rag-<version>.tgz` in this folder's `dist/`.

### Deploying a new build

Extract or copy over your existing `~/.opencode/plugins/docs-rag/`, keeping its
`data/` directory (corpus + indexes) intact — tarballs never contain one. Everything else is code/config and safe to replace; restart OpenCode so the updated plugin loads.

## Limitations

- PDF conversion quality varies with layout; scanned PDFs need OCR first
- Word “Title” style isn't a heading — use Heading 1 for structure
- Deleting a subject's index breaks queries until re-ingest of that subject
