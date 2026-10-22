import json
import os
import urllib.request
import urllib.error

import faiss
import numpy as np

from .config import (
    embed_batch,
    embed_model,
    http_timeout,
    lmstudio_token,
    lmstudio_url,
)

LMSTUDIO_BASE = lmstudio_url()
EMBED_MODEL = embed_model()
LMSTUDIO_TOKEN = lmstudio_token()

HTTP_TIMEOUT = http_timeout()
EMBED_BATCH = embed_batch()


def _http_json(url, payload=None):
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


def resolve_embed_model():
    """
    The configured embedding model. Required, used as-is: there is no
    auto-detection and no fallback across loaded models.
    """
    if not EMBED_MODEL:
        raise RuntimeError(
            "Embedding model is not configured. Set lmstudio.embed_model in the "
            "docs-rag config.json (or the LMSTUDIO_EMBED_MODEL environment "
            "variable) naming the model used for ingest and query."
        )
    return EMBED_MODEL


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
    Embed one or more texts via LM Studio.

    Returns an L2-normalized float32 array of shape (n, dim). Normalization
    is required because the index uses inner product (IndexFlatIP), so
    scores are equivalent to cosine similarity.

    task: "document" or "query" — applies the model's task prefix if it has
    one (nomic-embed).
    """
    if isinstance(texts, str):
        texts = [texts]
    model = model or resolve_embed_model()
    prefix = task_prefix(model, task)
    data = _http_json(
        f"{LMSTUDIO_BASE}/embeddings",
        {"model": model, "input": [prefix + t for t in texts]},
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
