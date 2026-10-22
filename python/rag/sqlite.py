import faiss
import hashlib
import json
import os
import re
import sqlite3

from .config import db_path, data_dir


DB_PATH = str(db_path())
RAG_ROOT = str(data_dir())

# BM25 keyword index over chunk text + document-level metadata (doc_meta).
# External-content FTS5 kept in sync with the chunks table by the triggers
# below. The porter tokenizer stems both indexed text and query terms (so
# "sibling" matches "siblings"); databases created before porter was added are
# rebuilt by _migrate_fts_tokenizer, and the single-column FTS index is
# upgraded to (text, doc_meta) by _migrate_fts_docmeta — both on open, with
# zero re-embedding.
FTS_TABLE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    doc_meta,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

FTS_TRIGGER_DDL = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, doc_meta) VALUES (new.rowid, new.text, new.doc_meta);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, doc_meta) VALUES ('delete', old.rowid, old.text, old.doc_meta);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, doc_meta) VALUES ('delete', old.rowid, old.text, old.doc_meta);
    INSERT INTO chunks_fts(rowid, text, doc_meta) VALUES (new.rowid, new.text, new.doc_meta);
END;
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    subject TEXT PRIMARY KEY,
    description TEXT,
    created_at TEXT,
    embed_model TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    subject TEXT,
    file_path TEXT,
    meta_json TEXT,
    FOREIGN KEY(subject) REFERENCES subjects(subject)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    subject TEXT,
    text TEXT,
    meta_json TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
    FOREIGN KEY(subject) REFERENCES subjects(subject)
);

-- Stable integer FAISS id per chunk. This replaces the old positional
-- (row-order) mapping, which silently broke on re-ingest.
CREATE TABLE IF NOT EXISTS chunk_ids (
    subject TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    faiss_id INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (subject, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_ids_faiss ON chunk_ids(subject, faiss_id);

-- Tiny key/value store for one-shot migration flags.
CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT
);

""" + FTS_TABLE_DDL + "\n\n" + FTS_TRIGGER_DDL


# ------------------------------------------------------------
# Strongly typed SubjectInfo data class
# ------------------------------------------------------------
class SubjectInfo:
    subject: str
    index_path: str
    dim: int
    ntotal: int
    embed_model: str | None

    def __init__(self, subject: str, index_path: str, dim: int, ntotal: int, embed_model: str | None):
        self.subject = subject
        self.index_path = index_path
        self.dim = dim
        self.ntotal = ntotal
        self.embed_model = embed_model

    def __repr__(self):
        return (
            f"SubjectInfo(subject={self.subject!r}, index_path={self.index_path!r}, "
            f"dim={self.dim}, ntotal={self.ntotal}, embed_model={self.embed_model!r})"
        )

# ------------------------------------------------------------
# get_subject_info — ALL logic lives here, returns SubjectInfo
# ------------------------------------------------------------
def get_subject_info(conn, subject: str) -> SubjectInfo:
    """
    Validate subject, FAISS index, idmap, embed model, and return a strongly typed
    SubjectInfo instance.
    """

    # FAISS index path
    index_dir = os.path.join(RAG_ROOT, "subjects", subject, "index")
    index_path = os.path.join(index_dir, "faiss.index")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found for subject '{subject}'")

    # Load FAISS index — also validates readability. The live vector count is authoritative here: ingest/forget report it too (index.ntotal), so no separate bookkeeping column can drift out of sync with the actual vectors.
    index = faiss.read_index(index_path)
    dim = index.d
    ntotal = index.ntotal

    # Subject existence check from SQLite; embed_model is resolved below via get_subject_embed_model()
    row = conn.execute(
        """
        SELECT 1 FROM subjects WHERE subject = ?""",
        [subject],
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Subject '{subject}' not found in SQLite. Ingest it first.")

    # embed_model override from rag.sqlite
    embed_model = get_subject_embed_model(conn, subject)

    # idmap check (legacy warning)
    idmap = get_faiss_idmap(conn, subject)
    if ntotal > 0 and not idmap:
        print(
            f"[warn] subject '{subject}' has {ntotal} vectors but "
            "no chunk id mapping (legacy index). Re-ingest the subject's "
            "files to migrate it."
        )

    return SubjectInfo(
        subject=subject,
        index_path=index_path,
        dim=dim,
        ntotal=ntotal,
        embed_model=embed_model,
    )

# ------------------------------------------------------------
# NEW: get_subjects wrapper — takes ONLY subjects list
# ------------------------------------------------------------
def get_subjects(subjects: list[str]) -> dict[str, SubjectInfo]:
    """
    Given a list of subject names, return a dict mapping each subject
    to its fully validated SubjectInfo instance.
    This wrapper manages its own SQLite connection.
    """
    conn = get_conn()
    try:
        out: dict[str, SubjectInfo] = {}
        for subject in subjects:
            out[subject] = get_subject_info(conn, subject)
        return out
    finally:
        conn.close()


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    _migrate(conn)
    _migrate_fts_tokenizer(conn)
    _migrate_fts_docmeta(conn)
    _ensure_fts_baseline(conn)
    _backfill_doc_meta(conn)
    conn.commit()
    return conn


def _migrate(conn):
    """One-shot column additions for databases created before the change."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(subjects)")}
    if "embed_model" not in cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN embed_model TEXT")
    chunk_cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    if "doc_meta" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN doc_meta TEXT NOT NULL DEFAULT ''")


def _migrate_fts_tokenizer(conn):
    """One-shot rebuild of chunks_fts when it predates the porter tokenizer.

    The FTS5 tokenizer cannot be altered in place, so the table (and the
    triggers that reference it) are dropped and recreated, then repopulated
    from the external content table. Zero embeddings are involved; only
    re-tokenization of the existing chunk texts.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    if row is None:
        return  # SCHEMA just created it (with porter)
    if "porter" in (row[0] or "").lower():
        return  # already stemmed
    for trig in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.execute("DROP TABLE chunks_fts")
    conn.executescript(FTS_TABLE_DDL + "\n" + FTS_TRIGGER_DDL)
    # Repopulate from the chunks table (external-content rebuild command).
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def _migrate_fts_docmeta(conn):
    """One-shot upgrade of the single-column (text) FTS index to the
    two-column (text, doc_meta) index. The FTS5 schema cannot be altered in
    place, so the table (and the triggers that reference it) are dropped and
    recreated, then repopulated from the external content table. Zero
    embeddings are involved; doc_meta is filled in by _backfill_doc_meta and
    the triggers keep FTS in sync afterwards."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    if row is None:
        return  # SCHEMA just created it (two-column)
    if "doc_meta" in (row[0] or ""):
        return  # already two-column
    for trig in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.execute("DROP TABLE chunks_fts")
    conn.executescript(FTS_TABLE_DDL + "\n" + FTS_TRIGGER_DDL)
    # Repopulate (doc_meta is still '' here; the backfill fills it in).
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def _kv_get(conn, k):
    row = conn.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
    return row[0] if row is not None else None


def _kv_set(conn, k, v):
    conn.execute("INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?)", (k, v))


def _backfill_doc_meta(conn):
    """One-shot: compute and stamp doc_meta for every existing document from
    the chunks that are already stored (documents use section/row_index; tables
    use the document's column list). No files are re-read and no embeddings are
    computed; the UPDATE triggers keep the FTS index in sync."""
    if _kv_get(conn, "docmeta_backfill") == "1":
        return
    from . import docmeta
    for doc_id, subject, meta_json in conn.execute(
        "SELECT doc_id, subject, meta_json FROM documents"
    ).fetchall():
        meta = json.loads(meta_json) if meta_json else {}
        columns = meta.get("columns")
        rows = conn.execute(
            "SELECT text, json_extract(meta_json, '$.row_index') "
            "FROM chunks WHERE doc_id = ? "
            "ORDER BY json_extract(meta_json, '$.row_index')",
            (doc_id,),
        ).fetchall()
        row_dicts = [
            {"text": text, "row_index": (ri if ri is not None else i)}
            for i, (text, ri) in enumerate(rows)
        ]
        dm = docmeta.compute_doc_meta(row_dicts, columns=columns, subject=subject)
        conn.execute(
            "UPDATE chunks SET doc_meta = ? WHERE doc_id = ?", (dm, doc_id)
        )
    _kv_set(conn, "docmeta_backfill", "1")


def _ensure_fts_baseline(conn):
    """One-time FTS population for rows that predate the FTS table."""
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if n_chunks == 0:
        return
    n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    if n_fts == 0:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        conn.commit()


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------
def insert_subject(conn, subject, description, created_at):
    conn.execute(
        "INSERT OR IGNORE INTO subjects(subject, description, created_at) VALUES (?, ?, ?)",
        (subject, description, created_at),
    )


def set_subject_embed_model(conn, subject, model):
    conn.execute(
        "UPDATE subjects SET embed_model = ? WHERE subject = ?",
        (model, subject),
    )


def get_subject_embed_model(conn, subject):
    """The embedding model recorded for the subject, or None (pre-migration)."""
    row = conn.execute(
        "SELECT embed_model FROM subjects WHERE subject = ?",
        (subject,),
    ).fetchone()
    return row[0] if row is not None else None


def insert_document(conn, doc_id, subject, file_path, meta):
    conn.execute(
        "INSERT OR REPLACE INTO documents(doc_id, subject, file_path, meta_json) VALUES (?, ?, ?, ?)",
        (doc_id, subject, file_path, json.dumps(meta)),
    )


def insert_chunks(conn, doc_id, subject, rows, doc_meta=""):
    """Replace the document's chunk set. Deleting first (instead of relying on
    chunk_id collisions) guarantees rows dropped from the file never linger.
    doc_meta is the document-level metadata, stamped on every row of the doc
    so it is indexed and retrievable alongside each chunk."""
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    for row in rows:
        conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, subject, text, doc_meta, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["id"], doc_id, subject, row["text"], doc_meta, json.dumps(row)),
        )


def commit(conn):
    conn.commit()


# ---------------------------------------------------------------------------
# Chunk state
# ---------------------------------------------------------------------------
def get_chunk_texts(conn, subject):
    """[(chunk_id, text)] in chunk_id order (stable)."""
    return conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE subject = ? ORDER BY chunk_id",
        (subject,),
    ).fetchall()


def get_chunk(conn, subject, chunk_id):
    """(text, meta_dict, doc_meta) for one chunk, or None."""
    row = conn.execute(
        "SELECT text, meta_json, doc_meta FROM chunks "
        "WHERE subject = ? AND chunk_id = ?",
        (subject, chunk_id),
    ).fetchone()
    if row is None:
        return None
    return row[0], json.loads(row[1]), row[2]


def prune_stale_chunks(conn):
    """
    Remove chunks that no longer belong to their document:
      - chunks whose doc_id is gone from documents
      - table rows beyond the document's current row count
        (row_index >= n_rows), e.g. after re-ingesting a shorter file
        under the same basename
    Returns the number of rows removed.
    """
    cur = conn.execute(
        """
        DELETE FROM chunks
        WHERE doc_id NOT IN (SELECT doc_id FROM documents)
           OR json_extract(meta_json, '$.row_index') >= (
                SELECT json_extract(d.meta_json, '$.n_rows')
                FROM documents d
                WHERE d.doc_id = chunks.doc_id
           )
        """
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# chunk_ids (faiss id map) accessors
# ---------------------------------------------------------------------------
def get_idmap(conn, subject):
    """{chunk_id: (faiss_id, content_hash)}"""
    rows = conn.execute(
        "SELECT chunk_id, faiss_id, content_hash FROM chunk_ids WHERE subject = ?",
        (subject,),
    ).fetchall()
    return {cid: (fid, h) for cid, fid, h in rows}


def get_faiss_idmap(conn, subject):
    """{faiss_id: chunk_id} for the subject, read live (inverse of get_idmap)."""
    rows = conn.execute(
        "SELECT faiss_id, chunk_id FROM chunk_ids WHERE subject = ?",
        (subject,),
    ).fetchall()
    return {fid: cid for fid, cid in rows}


def next_faiss_ids(conn, subject, n):
    """Allocate n new unique faiss ids for the subject (monotonic)."""
    if n <= 0:
        return []
    base = conn.execute(
        "SELECT COALESCE(MAX(faiss_id), -1) + 1 FROM chunk_ids WHERE subject = ?",
        (subject,),
    ).fetchone()[0]
    return [base + i for i in range(n)]


def upsert_chunk_id(conn, subject, chunk_id, faiss_id, hash_):
    conn.execute(
        "INSERT OR REPLACE INTO chunk_ids(subject, chunk_id, faiss_id, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (subject, chunk_id, faiss_id, hash_),
    )


def delete_chunk_ids(conn, subject, chunk_ids):
    if not chunk_ids:
        return
    conn.executemany(
        "DELETE FROM chunk_ids WHERE subject = ? AND chunk_id = ?",
        [(subject, c) for c in chunk_ids],
    )


# ---------------------------------------------------------------------------
# FTS5 keyword search
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    """
    a an and are as at be but by can could did do does for from had has have he her
    him his how i if in into is it its me may might my no nor not of off on or our
    she should so than that the their them then these they this to too was we were
    what when where which while who will with would you your
    """.split()
)

_FTS_QUERY = """
SELECT c.chunk_id, bm25(chunks_fts) AS score
FROM chunks_fts
JOIN chunks c ON c.rowid = chunks_fts.rowid
WHERE chunks_fts MATCH ? AND c.subject = ?
ORDER BY score
LIMIT ?
"""


def query_terms(text, max_terms=12):
    """Deduped, de-stopped, lowercased tokens from free text — the same terms
    used to build an FTS MATCH expression. Shared with the doc-meta hit
    counting in query.py so both signals agree on what counts as a term."""
    terms = []
    for tok in re.findall(r"[a-z0-9][a-z0-9_\-]*", text.lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in terms:
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def fts_match_from_text(conn, subject, text, max_terms=12, rare_frac=0.10):
    """
    Build a safe FTS5 MATCH expression from free text.

    The porter tokenizer stems both the indexed text and the query terms
    (so "sibling" matches "siblings"), and terms are additionally
    prefix-expanded (trailing *) to cover inflections the stemmer does not
    unify (e.g. digit-mixed tokens like "d4z4"). If at least one term is
    rare (document frequency <= rare_frac of the subject's chunks), the
    candidate pool is anchored on the rare terms: the full OR expression
    must co-occur with at least one rare term, so generic high-frequency
    terms ("study", "fshd", ...) can boost BM25 scores but cannot define
    the pool on their own.

    Returns (match, fallback): match is the (possibly anchored) expression,
    fallback the plain OR of all terms, to be tried when the anchored query
    matches nothing. Returns (None, None) when there are no usable terms.
    """
    terms = query_terms(text, max_terms)
    if not terms:
        return None, None

    full = " OR ".join(f'"{t}*"' for t in terms)
    n_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE subject = ?", (subject,)
    ).fetchone()[0]
    if n_chunks == 0:
        return full, None
    threshold = max(2, int(rare_frac * n_chunks))
    rare = []
    for t in terms:
        df = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts JOIN chunks c "
            "ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.subject = ?",
            (f'"{t}"', subject),
        ).fetchone()[0]
        if df <= threshold:
            rare.append(t)
    if not rare:
        return full, None
    anchor = " OR ".join(f'"{t}*"' for t in rare)
    return f"({full}) AND ({anchor})", full


def fts_search(conn, subject, match, limit, fallback=None):
    """BM25 keyword search within one subject. Returns [(chunk_id, score)].
    If the anchored match yields no rows and a fallback is given, retries
    with the plain OR expression."""
    rows = conn.execute(_FTS_QUERY, (match, subject, limit)).fetchall()
    if not rows and fallback and fallback != match:
        rows = conn.execute(_FTS_QUERY, (fallback, subject, limit)).fetchall()
    return rows
