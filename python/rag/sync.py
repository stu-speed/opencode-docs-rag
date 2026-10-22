"""
Incremental sync of a subject's FAISS index with its SQLite chunks.

SQLite (chunks + chunk_ids) is the source of truth. This module:
  - embeds only new or changed chunk texts (no-op costs zero embeddings)
  - adds vectors under stable integer faiss ids (IndexIDMap2)
  - removes vectors for deleted chunks
  - migrates legacy position-based flat indexes one-shot
"""
import numpy as np

from . import faiss as faiss_mod
from . import sqlite as rag_sqlite


def sync_subject_index(conn, subject, index_dir):
    """
    Bring the subject's index in line with its chunks in SQLite.
    Commits the chunk_ids changes. Returns (index, stats).
    """
    texts = {cid: text for cid, text in rag_sqlite.get_chunk_texts(conn, subject)}
    # Hash model + exact text that gets embedded (task prefix included), so a
    # change of embedding model or prefix policy re-embeds automatically.
    model = faiss_mod.resolve_embed_model()
    hashes = {
        cid: rag_sqlite.content_hash(
            model + "\0" + faiss_mod.prep_for_embed(t, model, "document")
        )
        for cid, t in texts.items()
    }
    rag_sqlite.set_subject_embed_model(conn, subject, model)

    known = rag_sqlite.get_idmap(conn, subject)  # {chunk_id: (faiss_id, hash)}

    to_add = [cid for cid in texts if cid not in known]
    to_update = [cid for cid in texts if cid in known and known[cid][1] != hashes[cid]]
    to_remove = [cid for cid in known if cid not in texts]

    index = faiss_mod.load_index(index_dir)
    stats = {"added": 0, "updated": 0, "removed": 0, "embedded": 0, "migrated": False}

    # Legacy position-based flat index: ids are row order, which is unsafe.
    # Rebuild id-mapped from scratch (one re-embed of the subject).
    if index is not None and not faiss_mod.is_idmap(index):
        index = None
        to_add, to_update, to_remove = list(texts), [], list(known)
        rag_sqlite.delete_chunk_ids(conn, subject, list(known))
        known = {}
        stats["migrated"] = True

    if index is None:
        # New (or migrated) index. Dim comes from the configured embedding
        # model when there is nothing to embed yet.
        embed_ids = to_add + to_update
        if embed_ids:
            arr = faiss_mod.embed_batched([texts[c] for c in embed_ids], task="document")
            index = faiss_mod.new_idmap_index(arr.shape[1])
            new_ids = rag_sqlite.next_faiss_ids(conn, subject, len(to_add))
            ids = new_ids + [known[c][0] for c in to_update]
            index.add_with_ids(arr, np.asarray(ids, dtype="int64"))
            for c, fid in zip(to_add, new_ids):
                rag_sqlite.upsert_chunk_id(conn, subject, c, fid, hashes[c])
            for c in to_update:
                rag_sqlite.upsert_chunk_id(conn, subject, c, known[c][0], hashes[c])
            stats["added"] = len(to_add)
            stats["updated"] = len(to_update)
            stats["embedded"] = len(embed_ids)
        else:
            index = faiss_mod.new_idmap_index(faiss_mod.probe_embed_dim())
        rag_sqlite.delete_chunk_ids(conn, subject, to_remove)
        stats["removed"] = len(to_remove)
        faiss_mod.save_index_atomic(index, index_dir)
        conn.commit()
        return index, stats

    # Incremental path.
    if to_remove or to_update:
        drop = [known[c][0] for c in to_remove] + [known[c][0] for c in to_update]
        faiss_mod.remove_ids(index, drop)
        rag_sqlite.delete_chunk_ids(conn, subject, to_remove)
        stats["removed"] = len(to_remove)

    if to_add or to_update:
        new_ids = rag_sqlite.next_faiss_ids(conn, subject, len(to_add))
        ids = new_ids + [known[c][0] for c in to_update]
        faiss_mod.add_with_ids(index, [texts[c] for c in to_add + to_update], ids, task="document")
        for c, fid in zip(to_add, new_ids):
            rag_sqlite.upsert_chunk_id(conn, subject, c, fid, hashes[c])
        for c in to_update:
            rag_sqlite.upsert_chunk_id(conn, subject, c, known[c][0], hashes[c])
        stats["added"] = len(to_add)
        stats["updated"] = len(to_update)
        stats["embedded"] = len(to_add) + len(to_update)

    faiss_mod.save_index_atomic(index, index_dir)
    conn.commit()
    return index, stats
