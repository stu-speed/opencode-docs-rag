"""
Shared configuration for docs-rag.

Loads <package root>/config.json (located from this file's position:
scripts/rag/config.py -> package root) and exposes typed accessors.
Environment variables override config values:

    LMSTUDIO_URL, LMSTUDIO_EMBED_MODEL, LMSTUDIO_TOKEN,
    DOCS_RAG_DATA_DIR, RAG_TOP_K, RAG_POOL_K, RAG_RRF_K, RAG_USE_FTS,
    RAG_HTTP_TIMEOUT, RAG_EMBED_BATCH, RAG_CHUNK_CHARS,
    RAG_RERANK_ENABLED, RAG_RERANK_MODEL, RAG_RERANK_CANDIDATES,
    RAG_RERANK_MAX_TOKENS, RAG_DOC_META_MAX_CHARS
"""
import json
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PKG_ROOT / "config.json"

_DEFAULTS = {
    "lmstudio.url": "http://localhost:1234/v1",
    "lmstudio.embed_model": "",
    "lmstudio.token": "",
    "data_dir": "data",
    "search.top_k": 8,
    "search.pool_k": 50,
    "search.rrf_k": 60,
    "search.use_fts": True,
    "rerank.enabled": True,
    "rerank.model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "rerank.candidates": 30,
    "rerank.max_tokens": 256,
    "http_timeout": 120,
    "embed_batch": 256,
    "chunk_chars": 2000,
    "doc_meta.max_chars": 1500,
    "doc_meta.preamble_sections": ["Abstract"],
    "doc_meta.per_subject": {},
}

_raw_cache = None


def _raw():
    global _raw_cache
    if _raw_cache is None:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _raw_cache = json.load(f)
        except Exception:
            _raw_cache = {}
    return _raw_cache


def get(dotted, env=None):
    """
    Config value by dotted path (e.g. "search.top_k").
    The env var (if given and set) wins; falls back to built-in defaults.
    """
    if env:
        v = os.environ.get(env)
        if v not in (None, ""):
            return v
    cur = _raw()
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _DEFAULTS.get(dotted)
        cur = cur[part]
    return cur


def data_dir():
    """Root of the RAG data (db, per-subject artifacts). Relative values resolve
    against the package root, so the default keeps everything inside the package."""
    p = Path(os.path.expanduser(str(get("data_dir", env="DOCS_RAG_DATA_DIR"))))
    return p if p.is_absolute() else PKG_ROOT / p


def state_dir():
    """Per-session runtime state (cleared at every opencode startup)."""
    return PKG_ROOT / "state"


def state_path():
    """Per-session active-subjects state file."""
    return state_dir() / "state.json"


def db_path():
    return data_dir() / "db" / "rag.db"


def lmstudio_url():
    return str(get("lmstudio.url", env="LMSTUDIO_URL")).rstrip("/")


def embed_model():
    return str(get("lmstudio.embed_model", env="LMSTUDIO_EMBED_MODEL"))


def lmstudio_token():
    return str(get("lmstudio.token", env="LMSTUDIO_TOKEN"))


def top_k():
    return int(get("search.top_k", env="RAG_TOP_K"))


def pool_k():
    return int(get("search.pool_k", env="RAG_POOL_K"))


def rrf_k():
    return int(get("search.rrf_k", env="RAG_RRF_K"))


def use_fts():
    v = get("search.use_fts", env="RAG_USE_FTS")
    if isinstance(v, bool):
        return v
    return str(v).lower() not in ("0", "false", "no", "")


def rerank_enabled():
    v = get("rerank.enabled", env="RAG_RERANK_ENABLED")
    if isinstance(v, bool):
        return v
    return str(v).lower() not in ("0", "false", "no", "")


def rerank_model():
    return str(get("rerank.model", env="RAG_RERANK_MODEL"))


def rerank_candidates():
    return int(get("rerank.candidates", env="RAG_RERANK_CANDIDATES"))


def rerank_max_tokens():
    return int(get("rerank.max_tokens", env="RAG_RERANK_MAX_TOKENS"))


def http_timeout():
    return int(get("http_timeout", env="RAG_HTTP_TIMEOUT"))


def embed_batch():
    return int(get("embed_batch", env="RAG_EMBED_BATCH"))


def chunk_chars():
    return int(get("chunk_chars", env="RAG_CHUNK_CHARS"))


def doc_meta_max_chars():
    return int(get("doc_meta.max_chars", env="RAG_DOC_META_MAX_CHARS"))


def doc_meta_preamble_sections(subject=None):
    """
    Section-name suffixes folded into a document's doc_meta (in addition to the
    bare-title preamble). A per-subject override (doc_meta.per_subject.<name>
    .preamble_sections) wins over the global list. Returns a list of str.
    """
    if subject:
        per = get("doc_meta.per_subject")
        if isinstance(per, dict) and isinstance(per.get(subject), dict):
            v = per[subject].get("preamble_sections")
            if isinstance(v, list):
                return [str(s) for s in v]
    v = get("doc_meta.preamble_sections")
    if isinstance(v, list):
        return [str(s) for s in v]
    return ["Abstract"]
