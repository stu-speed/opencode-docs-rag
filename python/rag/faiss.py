import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit

import faiss
import numpy as np

from .config import (
    embed_backend,
    embed_batch,
    embed_fallback_local,
    embed_model,
    http_timeout,
    lmstudio_token,
    lmstudio_url,
    local_embed_model,
)

LMSTUDIO_BASE = lmstudio_url()
EMBED_MODEL = embed_model()
LMSTUDIO_TOKEN = lmstudio_token()

HTTP_TIMEOUT = http_timeout()
EMBED_BATCH = embed_batch()
EMBED_BACKEND = embed_backend()

# Pre-flight probe (used by _http_json): seconds to wait for a TCP connect
# before declaring the server unreachable, and how long a successful probe
# stays cached. Deliberately tiny next to HTTP_TIMEOUT: its job is to turn a
# silent-drop "server down" hang into a fast, readable error.
CONNECT_PROBE_TIMEOUT = 3.0
_PROBE_TTL = 60.0
_probe_ok_at = 0.0         # last successful probe, monotonic clock


def _probe_server():
    """
    Pre-flight TCP connect to LMSTUDIO_BASE's host:port before an HTTP call.
    A dead host or firewall that silently drops packets would otherwise hang
    for the full HTTP_TIMEOUT waiting for the TCP handshake; a refused
    connection surfaces here in milliseconds with the same friendly message
    the urlopen handler gives. Successes are cached for _PROBE_TTL so steady
    streaming of batches pays at most one extra probe per window.
    """
    global _probe_ok_at
    if time.monotonic() - _probe_ok_at < _PROBE_TTL:
        return
    parts = urlsplit(LMSTUDIO_BASE)
    host = parts.hostname
    if not host:
        return  # unusual URL — let urlopen handle it
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=CONNECT_PROBE_TIMEOUT):
            pass
    except OSError as e:
        raise RuntimeError(
            f"LM Studio unreachable at {LMSTUDIO_BASE} ({e}). "
            "Start LM Studio and load an embedding model, or set LMSTUDIO_URL."
        ) from e
    _probe_ok_at = time.monotonic()


def _http_json(url, payload=None):
    _probe_server()
    headers = {}
    if LMSTUDIO_TOKEN:
        headers["Authorization"] = f"Bearer {LMSTUDIO_TOKEN}"
    if payload is None:
        req = urllib.request.Request(url, headers=headers)
    else:
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"LM Studio unreachable at {LMSTUDIO_BASE} ({e.reason}). "
            "Start LM Studio and load an embedding model, or set LMSTUDIO_URL."
        ) from e


# ---------------------------------------------------------------------------
# Local backend (fastembed, in-process ONNX — HTTP-free)
# ---------------------------------------------------------------------------
_LOCAL_EMBEDDER = None     # lazy TextEmbedding singleton
_LOCAL_MODEL_ID = ""       # fastembed model id backing the singleton

# Set to True once an http-embed call has failed and been routed to the local
# fallback (embeddings.fallback_local). Sticky for the rest of the process:
# later embeds skip the http probe instead of paying its timeout per batch.
# Note this means a server that comes back mid-process is not picked up —
# consistent within-process embeddings matter more here.
_HTTP_FALLBACK_ACTIVE = False
_FALLBACK_WARNED = False


def _warn_fallback(http_err):
    """One-time stderr note that embeddings switched to the local backend."""
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        print(
            f"docs-rag: LM Studio unreachable ({http_err}); "
            "falling back to local fastembed embeddings for the rest of "
            "this process (embeddings.fallback_local).",
            file=sys.stderr,
        )


def _check_backend():
    if EMBED_BACKEND not in ("http", "local"):
        raise RuntimeError(
            f"Unknown embeddings.backend {EMBED_BACKEND!r} "
            "(config or RAG_EMBED_BACKEND). Use \"http\" or \"local\"."
        )
    return EMBED_BACKEND


def _supported_local_models():
    """Supported fastembed model ids (built-in catalogue)."""
    from fastembed import TextEmbedding
    return [str(m["model"]) for m in TextEmbedding.list_supported_models()]


def _map_local_model(model):
    """
    Map the canonical embedding model name (lmstudio.embed_model, e.g.
    "text-embedding-nomic-embed-text-v1.5") to a fastembed model id.
    embeddings.local_model wins when set; otherwise accept an exact
    id, then match a catalogue id ending in "/<name>".
    """
    configured = local_embed_model()
    if configured:
        return configured
    if not (model or "").strip():
        raise RuntimeError(
            "Embedding model is not configured. Set lmstudio.embed_model in the "
            "docs-rag config.json (or the LMSTUDIO_EMBED_MODEL environment "
            "variable), or set embeddings.local_model to a fastembed id such "
            "as 'nomic-ai/nomic-embed-text-v1.5'."
        )
    candidates = [model]
    stripped = model[len("text-embedding-"):] if model.startswith("text-embedding-") else model
    if stripped != model:
        candidates.append(stripped)
    supported = [s.lower() for s in _supported_local_models()]
    for cand in (c.lower() for c in candidates if c):
        if cand in supported:
            return cand
        for supp in supported:
            if supp.endswith("/" + cand):
                return supp
    raise RuntimeError(
        f"Cannot map embedding model {model!r} to a fastembed model. "
        "Set embeddings.local_model in the docs-rag config.json (or "
        "RAG_EMBED_LOCAL_MODEL) to a fastembed id such as "
        "'nomic-ai/nomic-embed-text-v1.5'."
    )


def _local_embedder(model):
    """Lazy fastembed TextEmbedding for the resolved model id (cached)."""
    global _LOCAL_EMBEDDER, _LOCAL_MODEL_ID
    model_id = _map_local_model(model)
    if _LOCAL_EMBEDDER is None or _LOCAL_MODEL_ID != model_id:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "embeddings.backend is \"local\" but fastembed is not installed. "
                "Run `pip install fastembed` (onnxruntime only, no torch), or "
                "set embeddings.backend to \"http\"."
            ) from e
        _LOCAL_EMBEDDER = TextEmbedding(model_name=model_id)
        _LOCAL_MODEL_ID = model_id
    return _LOCAL_EMBEDDER


def _embed_local(prefixed, model):
    """
    Embed already-prefixed texts in-process via fastembed. fastembed applies
    no task prefixes itself, so the caller's prep_for_embed() output is used
    as-is. Raw vectors are already L2-normalized; normalized again for parity.
    """
    out = np.asarray(list(_local_embedder(model).embed(prefixed, batch_size=EMBED_BATCH)), dtype="float32")
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def resolve_embed_model():
    """
    The configured embedding model. Required, used as-is: there is no
    auto-detection and no fallback across loaded models. On the local backend
    a fastembed model id is accepted as the canonical name.
    """
    if EMBED_MODEL:
        return EMBED_MODEL
    if _check_backend() == "local":
        return _map_local_model("")
    raise RuntimeError(
        "Embedding model is not configured. Set lmstudio.embed_model in the "
        "docs-rag config.json (or the LMSTUDIO_EMBED_MODEL environment "
        "variable) naming the model used for ingest and query."
    )


def task_prefix(model, task):
    """
    Task prefix for models trained with them (nomic-embed): documents are
    embedded as "search_document: ..." and queries as "search_query: ...".
    Empty string for other models.
    """
    if task and "nomic" in (model or "").lower():
        return {"document": "search_document: ", "query": "search_query: "}[task]
    return ""


def prep_for_embed(text, model, task):
    """The exact text that will be embedded (prefix applied)."""
    return task_prefix(model, task) + text


def embed(texts, model=None, task=None):
    """
    Embed one or more texts — over HTTP (LM Studio, OpenAI-compatible
    /embeddings) or in-process via fastembed, per embeddings.backend.

    Returns an L2-normalized float32 array of shape (n, dim). Normalization
    is required because the index uses inner product (IndexFlatIP), so
    scores are equivalent to cosine similarity.

    task: "document" or "query" — applies the model's task prefix if it has
    one (nomic-embed).

    Fallback: when embeddings.fallback_local is enabled (opt-in — only turn
    it on when the locally mapped fastembed model is compatible with the
    server's), an unreachable http backend switches to _embed_local for the
    rest of the process; with the setting off, the http error propagates.
    """
    global _HTTP_FALLBACK_ACTIVE
    if isinstance(texts, str):
        texts = [texts]
    model = model or resolve_embed_model()
    prefix = task_prefix(model, task)
    prefixed = [prefix + t for t in texts]
    if _check_backend() == "local" or _HTTP_FALLBACK_ACTIVE:
        return _embed_local(prefixed, model)
    try:
        return _embed_http(prefixed, model)
    except RuntimeError as e:
        if not embed_fallback_local():
            raise
        _HTTP_FALLBACK_ACTIVE = True
        _warn_fallback(e)
        try:
            return _embed_local(prefixed, model)
        except Exception as local_err:
            raise RuntimeError(
                f"{e}\nembeddings.fallback_local is enabled, but the local "
                f"backend also failed: {local_err}"
            ) from local_err


def _embed_http(prefixed, model):
    """One /embeddings round trip; raises RuntimeError when the server is
    unreachable. Returns an L2-normalized float32 array of shape (n, dim)."""
    data = _http_json(
        f"{LMSTUDIO_BASE}/embeddings",
        {"model": model, "input": prefixed},
    )
    items = sorted(data["data"], key=lambda d: d["index"])
    arr = np.asarray([item["embedding"] for item in items], dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_batched(texts, batch_size=None, task=None):
    """Embed texts in batches to keep individual HTTP payloads bounded."""
    if batch_size is None:
        batch_size = EMBED_BATCH
    if not texts:
        return np.empty((0, 0), dtype="float32")
    if batch_size <= 0 or len(texts) <= batch_size:
        return embed(texts, task=task)
    parts = [embed(texts[i:i + batch_size], task=task) for i in range(0, len(texts), batch_size)]
    return np.vstack(parts)


def embed_one(text, model=None):
    """Embed a single text; returns a normalized float32 vector of shape (dim,)."""
    return embed([text], model=model)[0]


def probe_embed_dim(model=None):
    """Embed a probe string and return the configured model's output dimension."""
    return embed(["dim probe"], model=model).shape[1]


# ---------------------------------------------------------------------------
# Index management (IndexIDMap2 with explicit chunk ids)
# ---------------------------------------------------------------------------
def is_idmap(index):
    return isinstance(index, faiss.IndexIDMap2)


def load_index(index_dir):
    """Load the subject index from disk, or None if it does not exist."""
    path = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


def new_idmap_index(dim):
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def save_index_atomic(index, index_dir):
    """Write to a temp file then os.replace, so a concurrent reader never
    sees a torn index file."""
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, "faiss.index")
    tmp = f"{path}.tmp.{os.getpid()}"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)


def add_with_ids(index, texts, ids, task=None):
    """Embed texts and add them under the given faiss ids. Returns the matrix."""
    if not texts:
        return np.empty((0, index.d), dtype="float32")
    arr = embed_batched(texts, task=task)
    if arr.shape[1] != index.d:
        raise ValueError(
            f"embedding dim {arr.shape[1]} does not match index dim {index.d}. "
            "Set LMSTUDIO_EMBED_MODEL to the model used when this index was built."
        )
    index.add_with_ids(arr, np.asarray(ids, dtype=np.int64))
    return arr


def remove_ids(index, ids):
    """Remove vectors by faiss id. Returns the number removed."""
    if not ids:
        return 0
    return int(index.remove_ids(np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)))
