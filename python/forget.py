#!/usr/bin/env python3
"""
Forget a source file from a subject.

Usage: forget.py <subject> <file>

Removes the document's chunks from SQLite (FTS5 follows via triggers),
drops the corresponding vectors from the FAISS index, deletes the stored
raw/json artifacts, and prunes the chunk id mapping. Other files in the
subject are untouched.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag.config import data_dir
from rag.sqlite import get_conn, commit
from rag.sync import sync_subject_index

RAG_ROOT = str(data_dir())


MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def main(subject, files):
    subject = subject.strip()
    if not subject:
        raise ValueError("Subject cannot be empty.")

    msgs_all = []

    for file_path in files:
        src_name = os.path.basename(file_path)
        base = os.path.join(RAG_ROOT, "subjects", subject)
        index_dir = os.path.join(base, "index")

        ext = os.path.splitext(src_name)[1].lower()
        if ext in MARKDOWN_EXTENSIONS:
            names = [src_name]
        else:
            names = [os.path.splitext(src_name)[0] + ".md", src_name]

        conn = get_conn()
        n_chunks = 0
        n_docs = 0

        for name in names:
            doc_id = f"{subject}_{name}"
            n_chunks += conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
            ).rowcount
            n_docs += conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            ).rowcount

        commit(conn)

        if n_chunks == 0 and n_docs == 0:
            conn.close()
            msgs_all.append(
                f"No document '{src_name}' found in subject '{subject}'. Nothing to forget."
            )
            continue

        if not os.path.exists(index_dir):
            conn.close()
            msgs_all.append(
                f"[warn] No FAISS index directory for subject '{subject}'."
            )
            continue

        index, stats = sync_subject_index(conn, subject, index_dir)
        conn.close()

        for name in names:
            raw_path = os.path.join(base, "raw", name)
            json_path = os.path.join(base, "json", name + ".json")

            if os.path.exists(raw_path):
                os.remove(raw_path)
            if os.path.exists(json_path):
                os.remove(json_path)

        msgs_all.append(
            f"Forgot '{src_name}' from subject '{subject}': "
            f"removed {n_chunks} chunks, {stats['removed']} vectors ({index.ntotal} remain)."
        )

    print("\n".join(msgs_all))


if __name__ == "__main__":
    try:
        args = json.loads(sys.argv[1])
        subject = args["subject"]
        files = args["files"]
        main(subject, files)
    except Exception as e:
        print(f"[error] {e}")
        raise
