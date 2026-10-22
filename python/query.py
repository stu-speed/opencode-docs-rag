#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import faiss
from typing import Dict, List, Tuple, Optional

from rag.faiss import embed, resolve_embed_model
from rag.sqlite import (
    SubjectInfo,
    get_conn,
    get_subjects,
    get_faiss_idmap,
    get_chunk,
    fts_match_from_text,
    fts_search,
    query_terms,
)
from rag.rerank import LocalOnnxReranker
from rag.config import (
    pool_k,
    top_k,
    rrf_k,
    rerank_enabled,
    rerank_candidates,
    use_fts,
)


ChunkKey = Tuple[str, int]          # (subject, chunk_id)
PoolItem = Tuple[str, int, float]   # (subject, chunk_id, score)


POOL_K: int = pool_k()
TOP_K: int = top_k()
RRF_K: int = rrf_k()
USE_FTS: bool = use_fts()
RERANK_ENABLED: bool = rerank_enabled()
RERANK_CANDIDATES: int = rerank_candidates()


# ------------------------------------------------------------
# Embedding
# ------------------------------------------------------------

def embed_question(question: str, dim: int) -> List[float]:
    model: str = resolve_embed_model()
    vec = embed([question], model=model, task="query")[0]

    if vec.shape[0] != dim:
        raise RuntimeError(
            f"Embedding dim {vec.shape[0]} != index dim {dim}. Re-ingest subject."
        )
    return vec


# ------------------------------------------------------------
# Hybrid search (single subject)
# ------------------------------------------------------------

def hybrid_search(
    subject: str,
    index: faiss.Index,
    idmap: Dict[int, int],
    conn,
    question: str,
    qvec
) -> List[PoolItem]:

    signals: List[List[Tuple[str, int]]] = []

    # Dense FAISS
    if index.ntotal > 0:
        distances, indices = index.search(qvec.reshape(1, -1), POOL_K)
        dense: List[Tuple[str, int]] = []

        for dist, fid in zip(distances[0], indices[0]):
            if fid < 0:
                continue
            cid = idmap.get(int(fid))
            if cid is not None:
                dense.append((subject, cid))

        if dense:
            signals.append(dense)

    # Keyword FTS
    if USE_FTS:
        match, fallback = fts_match_from_text(conn, subject, question)
        if match:
            rows = fts_search(conn, subject, match, POOL_K, fallback)
            kw = [(subject, cid) for cid, _score in rows]
            if kw:
                signals.append(kw)

    # RRF fusion
    fused: Dict[ChunkKey, float] = {}
    n_signals: Dict[ChunkKey, int] = {}
    best_rank: Dict[ChunkKey, int] = {}

    for ranked_list in signals:
        for rank, key in enumerate(ranked_list, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            n_signals[key] = n_signals.get(key, 0) + 1
            prev = best_rank.get(key)
            if prev is None or rank < prev:
                best_rank[key] = rank

    ranked = sorted(
        fused.items(),
        key=lambda kv: (-kv[1], -n_signals[kv[0]], best_rank[kv[0]], kv[0])
    )

    return [(subject, chunk_id, score) for (subject, chunk_id), score in ranked]


# ------------------------------------------------------------
# doc-meta term coverage
# ------------------------------------------------------------

def _body_of(text_field: str) -> str:
    try:
        rec = json.loads(text_field)
        if isinstance(rec, dict):
            return rec.get("text", "") or ""
    except Exception:
        pass
    return text_field or ""


def _fetch_text_and_meta(conn, items: List[PoolItem]) -> Dict[ChunkKey, Tuple[str, str]]:
    out: Dict[ChunkKey, Tuple[str, str]] = {}

    # Group by subject
    by_subject: Dict[str, List[int]] = {}
    for subject, cid, _score in items:
        by_subject.setdefault(subject, []).append(cid)

    for subject, cids in by_subject.items():
        marks = ",".join("?" * len(cids))
        rows = conn.execute(
            f"SELECT chunk_id, text, doc_meta FROM chunks "
            f"WHERE subject = ? AND chunk_id IN ({marks})",
            [subject, *cids],
        ).fetchall()

        for cid, text, doc_meta in rows:
            out[(subject, cid)] = (text, doc_meta)

    return out


def count_doc_hits(body: str, doc_meta: str, terms: List[str]) -> int:
    hay = (body + "\n" + (doc_meta or "")).lower()
    return sum(1 for t in terms if t in hay)


def apply_doc_hits(pool: List[PoolItem], conn, terms: List[str]) -> List[PoolItem]:
    if not terms:
        return pool

    fetched = _fetch_text_and_meta(conn, pool)
    scored: List[Tuple[PoolItem, int]] = []

    for item in pool:
        key = (item[0], item[1])
        if key not in fetched:
            continue
        text, doc_meta = fetched[key]
        hits = count_doc_hits(_body_of(text), doc_meta, terms)
        scored.append((item, hits))

    scored.sort(key=lambda it: -it[1])
    return [item for item, _hits in scored]


# ------------------------------------------------------------
# Cross-encoder rerank
# ------------------------------------------------------------

def _fetch_texts(conn, items: List[PoolItem]) -> List[Tuple[str, int, str]]:
    out: Dict[ChunkKey, str] = {}

    # Group by subject
    by_subject: Dict[str, List[int]] = {}
    for subject, cid, _score in items:
        by_subject.setdefault(subject, []).append(cid)

    for subject, cids in by_subject.items():
        marks = ",".join("?" * len(cids))
        rows = conn.execute(
            f"SELECT chunk_id, text FROM chunks "
            f"WHERE subject = ? AND chunk_id IN ({marks})",
            [subject, *cids],
        ).fetchall()

        for cid, text in rows:
            out[(subject, cid)] = text

    final: List[Tuple[str, int, str]] = []
    for s, cid, _score in items:
        text = out.get((s, cid))
        if text is not None:
            final.append((s, cid, text))
    return final


def rerank_pool(
    question: str,
    pool: List[PoolItem],
    conn,
    warnings: List[str]
) -> Tuple[List[PoolItem], Dict[ChunkKey, float]]:

    if not RERANK_ENABLED or len(pool) <= TOP_K:
        return pool, {}

    window = pool[: min(RERANK_CANDIDATES, len(pool))]

    try:
        reranker = LocalOnnxReranker()
        fetched = _fetch_texts(conn, window)
        scores = reranker.rerank(question, [text for _s, _c, text in fetched])
    except Exception as e:
        warnings.append(f"[warn] rerank failed ({e})")
        return pool, {}

    scores_by_key: Dict[ChunkKey, float] = {
        (subject, cid): score
        for (subject, cid, _text), score in zip(fetched, scores)
    }

    scored = [(item, scores_by_key[(item[0], item[1])]) for item in window
              if (item[0], item[1]) in scores_by_key]
    unscored = [item for item in window
                if (item[0], item[1]) not in scores_by_key]

    scored.sort(key=lambda it: -it[1])
    new_pool = [item for item, _score in scored] + unscored + pool[len(window):]

    return new_pool, scores_by_key


# ------------------------------------------------------------
# Build context blocks
# ------------------------------------------------------------

def build_context(
    ranked: List[PoolItem],
    conn,
    rerank_scores: Optional[Dict[ChunkKey, float]] = None
) -> List[Dict[str, object]]:

    blocks: List[Dict[str, object]] = []

    for subject, cid, score in ranked:
        row = get_chunk(conn, subject, cid)
        if row is None:
            continue

        text, meta, doc_meta = row

        block: Dict[str, object] = {
            "subject": subject,
            "chunk_id": cid,
            "score": float(score),
            "text": text,
            "meta": meta,
            "doc_meta": doc_meta or "",
        }

        if rerank_scores and (subject, cid) in rerank_scores:
            block["rerank_score"] = float(rerank_scores[(subject, cid)])

        blocks.append(block)

    return blocks


# ------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------

def build_context_output(question: str, context: List[Dict[str, object]]) -> str:
    if not context:
        return f"No relevant chunks retrieved for: {question}"

    lines: List[str] = [f"Retrieved {len(context)} chunk(s) for: {question}", ""]

    for i, b in enumerate(context):
        header = (
            f"[{i+1}] (subject={b['subject']}, "
            f"source={b['meta'].get('source_file','?')}, "
            f"chunk={b['chunk_id']}, score={b['score']:.4f}"
        )
        if "rerank_score" in b:
            header += f", rerank={b['rerank_score']:.4f}"
        lines.append(header + ")")

        if b.get("doc_meta"):
            lines.append(f"[doc-meta] {b['doc_meta']}")

        lines.append(str(b["text"]))
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------
# Main (multi-subject)
# ------------------------------------------------------------

def main(subjects: List[str], question: str) -> None:
    if not subjects:
        raise ValueError("No subjects provided.")

    conn = get_conn()
    warnings: List[str] = []

    # Strongly typed multi-subject loader
    subject_info_map: Dict[str, SubjectInfo] = get_subjects(subjects)

    fused_all: List[PoolItem] = []

    # Deterministic subject order
    for subject in subjects:
        if subject not in subject_info_map:
            warnings.append(f"[warn] Subject not found: {subject}")
            continue

        info = subject_info_map[subject]

        # FAISS index existence check
        if not os.path.exists(info.index_path):
            warnings.append(f"[warn] Missing FAISS index for subject: {subject}")
            continue

        index: faiss.Index = faiss.read_index(info.index_path)
        idmap: Dict[int, int] = get_faiss_idmap(conn, subject)

        dim: int = index.d
        qvec = embed_question(question, dim)

        fused = hybrid_search(subject, index, idmap, conn, question, qvec)
        fused_all.extend(fused)

    # Nothing retrieved
    if not fused_all:
        print("[warn] No results found across all subjects.")
        conn.close()
        return

    # Global rerank across subjects
    ranked, rerank_scores = rerank_pool(question, fused_all, conn, warnings)

    # Doc-meta term hits
    ranked = apply_doc_hits(ranked, conn, query_terms(question))

    # Build context
    context = build_context(ranked[:TOP_K], conn, rerank_scores)
    conn.close()

    # Output
    out = build_context_output(question, context)

    if warnings:
        out = "\n".join(warnings + [out])

    print(out)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        # JSON input from OpenCode router
        args = json.loads(sys.argv[1])

        subjects = args["subjects"]  # list[str]
        question = args["question"]  # str

        main(subjects, question)
    except Exception as e:
        print(f"[error] {e}")
        raise
