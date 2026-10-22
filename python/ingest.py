#!/usr/bin/env python3
import csv as csvmod
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from rag.sqlite import (
    get_conn,
    insert_subject,
    insert_document,
    insert_chunks,
    commit,
    prune_stale_chunks,
)
from rag.config import chunk_chars, data_dir
from rag import docmeta
from rag.sync import sync_subject_index


RAG_ROOT = str(data_dir())

def _read_excel_all_sheets(path):
    """
    Load every non-empty sheet of a workbook. With 2+ sheets the frames are
    concatenated with a `__sheet` column; a single-sheet workbook loads exactly
    as before (no extra column), so re-ingests stay content-identical.
    Returns None when every sheet is empty.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    frames = [(name, df) for name, df in sheets.items() if not df.empty]
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0][1]
    combined = []
    for name, df in frames:
        df = df.copy()
        df["__sheet"] = name
        combined.append(df)
    return pd.concat(combined, ignore_index=True)


# Binary/tabular formats: always parsed with their dedicated loader.
BINARY_TABLE_LOADERS = {
    ".xlsx": _read_excel_all_sheets,
    ".xls": _read_excel_all_sheets,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".orc": getattr(pd, "read_orc", None),
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
}

# Text formats that are always documents (never sniffed as tables).
DOCUMENT_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".tex", ".adoc", ".org", ".log",
    ".pdf", ".docx", ".doc", ".epub", ".pptx", ".html", ".htm",
}

# Formats whose own content is already markdown (no conversion).
MARKDOWN_EXTENSIONS = {".md", ".markdown"}

# Formats converted to markdown via MarkItDown.
MARKITDOWN_EXTENSIONS = {".docx", ".pptx", ".html", ".htm", ".epub"}

# Plain text formats (stored as markdown as-is).
PLAIN_TEXT_EXTENSIONS = {".rst", ".tex", ".adoc", ".org", ".log"}

# Document chunking: merge paragraph blocks under the same section heading up
# to this many characters (~500 tokens) before starting a new row.
DOC_CHUNK_MAX_CHARS = chunk_chars()

# Delimiters tried (in order) when sniffing ambiguous text files.
SNIFF_DELIMITERS = ["\t", ",", "|", ";"]

MIN_TABLE_COLUMNS = 2


class IngestError(Exception):
    """Raised when a file cannot be normalized into rows."""


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            return f.read()
    except Exception as e:
        raise IngestError(f"could not read file {path}: {e}") from e


MARKER_TIMEOUT = 900  # seconds per marker_single run (CPU fast mode)


def _marker_single_bin():
    """Locate the marker single-file CLI: next to this interpreter first, then PATH."""
    candidates = [os.path.join(os.path.dirname(sys.executable), "marker_single")]
    which = shutil.which("marker_single")
    if which is not None:
        candidates.append(which)
    for c in dict.fromkeys(candidates):
        if os.access(c, os.X_OK):
            return c
    raise IngestError(
        f"marker_single executable not found (looked next to {sys.executable} and on "
        "PATH); install with: pip install marker-pdf"
    )


def _marker_convert(path):
    """Convert a PDF via the marker CLI in fast mode.

    No LLM is used, so there are no external API calls; layout/table detection
    runs on local models only (no HuggingFace downloads for text-layered PDFs).
    Image extraction is disabled to keep RAG chunks as clean markdown."""
    bin_path = _marker_single_bin()
    with tempfile.TemporaryDirectory(prefix="docs-rag-marker-") as out_dir:
        cmd = [
            bin_path, path,
            "--output_dir", out_dir,
            "--output_format", "markdown",
            "--mode", "fast",
            "--disable_image_extraction",
            "--disable_tqdm",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=MARKER_TIMEOUT
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            raise IngestError(f"marker_single did not complete ({e})") from e
        tail = ((proc.stderr or proc.stdout).strip().splitlines())[-3:]
        if proc.returncode != 0:
            raise IngestError(
                f"marker_single failed (exit {proc.returncode}): {' | '.join(tail)}"
            )
        md_files = sorted(glob.glob(os.path.join(out_dir, "**", "*.md"), recursive=True))
        if not md_files or os.path.getsize(md_files[0]) == 0:
            raise IngestError(
                f"marker_single produced no markdown (exit {proc.returncode}): "
                + (" | ".join(tail) if tail else "(no output)")
            )
        with open(md_files[0], "r", encoding="utf-8") as f:
            text = f.read()
    return text


def _markitdown_convert(path):
    from markitdown import MarkItDown
    return MarkItDown().convert(path).text_content


def convert_document_to_markdown(path):
    """
    Convert a document to markdown text.

    Returns (markdown_text, converted) where converted is True when a format
    converter produced the text (md/txt files are their own markdown).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in MARKDOWN_EXTENSIONS or ext in PLAIN_TEXT_EXTENSIONS:
        text = read_text(path)
        if looks_like_binary(text):
            raise IngestError(f"binary file, cannot read as text: {path}")
        return text, False

    if ext == ".pdf":
        # Primary: marker (fast mode, local models only - no LLM calls). Falls
        # back to pymupdf4llm when the user has not installed marker-pdf.
        errors = []
        try:
            return _marker_convert(path), True
        except Exception as e:
            errors.append(f"marker: {e}")
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(path), True
        except Exception as e:
            errors.append(f"pymupdf4llm: {e}")
        # Last resort.
        try:
            return _markitdown_convert(path), True
        except Exception as e:
            errors.append(f"markitdown: {e}")
        raise IngestError(f"could not convert PDF {path}: " + "; ".join(errors))

    if ext in MARKITDOWN_EXTENSIONS:
        try:
            return _markitdown_convert(path), True
        except Exception as e:
            raise IngestError(f"could not convert {path}: {e}") from e

    # Anything else (e.g. legacy .doc): best-effort plain text.
    text = read_text(path)
    if looks_like_binary(text):
        raise IngestError(f"binary file, cannot ingest as text: {path}")
    return text, False


def looks_like_binary(text):
    return "\x00" in text[:8192]


def sniff_delimited_table(text):
    """
    Strict header+rows detection for plain text.

    Returns a DataFrame only if EVERY non-empty line has exactly the same
    number of fields as the header line, with at least MIN_TABLE_COLUMNS
    fields. Anything ragged or single-column is rejected (it is a document,
    not a table).
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    for delim in SNIFF_DELIMITERS:
        if delim not in lines[0]:
            continue
        rows = list(csvmod.reader(lines, delimiter=delim))
        # Drop empty leading/trailing fields produced by fenced pipe tables.
        if delim == "|":
            rows = [
                r[1:-1] if len(r) >= 2 and r[0].strip() == "" and r[-1].strip() == "" else r
                for r in rows
            ]
            # Drop GFM header separator rows (|---|---|).
            rows = [
                r for r in rows
                if not (
                    len(r) >= MIN_TABLE_COLUMNS
                    and all(_GFM_SEP_RE.fullmatch(f.strip()) for f in r)
                )
            ]
        if len(rows) < 2:
            continue
        n_cols = len(rows[0])
        if n_cols < MIN_TABLE_COLUMNS:
            continue
        if not all(len(r) == n_cols for r in rows):
            continue
        header = [c.strip() for c in rows[0]]
        return pd.DataFrame(rows[1:], columns=header)
    return None


def try_strict_csv(path, sep):
    """
    Parse a file with a known delimited extension. Ragged or single-column
    input is rejected (returns None) so it falls back to document mode.
    """
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception:
        return None
    if df.empty or df.shape[1] < MIN_TABLE_COLUMNS:
        return None
    return df


def detect_and_load(path):
    """
    Detect whether a file is a header+rows table or a single document.

    Returns (mode, data) where mode is "table" (data: DataFrame) or
    "raw" (data: full document text).
    """
    ext = os.path.splitext(path)[1].lower()

    # 1. Known binary/tabular formats.
    if ext in BINARY_TABLE_LOADERS:
        loader = BINARY_TABLE_LOADERS[ext]
        load_err = None
        if loader is not None:
            try:
                df = loader(path)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return "table", df
            except Exception as e:
                load_err = e
        text = read_text(path)
        if looks_like_binary(text):
            raise IngestError(
                f"cannot load {path} as a table"
                + (f" ({load_err})" if load_err is not None else " (no loader available)")
                + " and the file is not readable text"
            )
        return "raw", text

    # 2. JSON: table only if it parses as a multi-column frame.
    if ext == ".json":
        try:
            df = pd.read_json(path)
            if isinstance(df, pd.DataFrame) and df.shape[1] >= MIN_TABLE_COLUMNS:
                return "table", df
        except Exception:
            pass
        text = read_text(path)
        if looks_like_binary(text):
            raise IngestError(f"binary file, cannot ingest as text: {path}")
        return "raw", text

    # 3. Known document formats: convert to markdown (the editable source of truth).
    if ext in DOCUMENT_EXTENSIONS:
        return "raw", convert_document_to_markdown(path)[0]

    # 4. Known delimited text: strict parse, else document.
    if ext == ".csv":
        df = try_strict_csv(path, ",")
        if df is not None:
            return "table", df
        text = read_text(path)
        if looks_like_binary(text):
            raise IngestError(f"binary file, cannot ingest as text: {path}")
        return "raw", text
    if ext in (".tsv", ".tab"):
        df = try_strict_csv(path, "\t")
        if df is not None:
            return "table", df
        text = read_text(path)
        if looks_like_binary(text):
            raise IngestError(f"binary file, cannot ingest as text: {path}")
        return "raw", text

    # 5. Ambiguous text (.txt, no extension, ...): sniff, else document.
    text = read_text(path)
    if looks_like_binary(text):
        raise IngestError(f"binary file, cannot ingest as text: {path}")
    df = sniff_delimited_table(text)
    if df is not None:
        return "table", df
    return "raw", text


def ensure_subject_dirs(subject):
    base = os.path.join(RAG_ROOT, "subjects", subject)
    raw_dir = os.path.join(base, "raw")
    json_dir = os.path.join(base, "json")
    index_dir = os.path.join(base, "index")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    return base, raw_dir, json_dir, index_dir


def build_rows_from_table(df, source_file):
    src = os.path.basename(source_file)
    df = df.reset_index(drop=True)
    records = df.to_json(orient="records", lines=True, force_ascii=False)
    rows = []
    seen = {}
    for i, line in enumerate(records.splitlines()):
        # Content-hashed id (with an occurrence suffix for duplicate rows):
        # reordering rows does not force re-embedding, only rows whose
        # content was added, changed, or removed.
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:12]
        seen[digest] = seen.get(digest, 0) + 1
        rows.append({
            "id": f"{src}_{digest}_{seen[digest]}",
            "text": line,
            "source_file": src,
            "row_index": i,
        })
    return rows


_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_GFM_SEP_RE = re.compile(r":?-+:?")


def markdown_blocks(md_text):
    """
    Split markdown into (section_path, block) pairs.

    Blocks break at blank lines outside code fences (fences stay intact).
    ATX headings set the section path for everything until the next heading
    of the same or higher level.
    """
    blocks = []
    stack = []  # [(level, title), ...]
    cur = []
    in_fence = False
    fence_char = ""

    def flush():
        nonlocal cur
        if cur:
            body = "\n".join(cur).strip("\n")
            if body:
                blocks.append((" > ".join(t for _lvl, t in stack), body))
        cur = []

    for line in md_text.splitlines():
        fm = _FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)
            if not in_fence:
                in_fence, fence_char = True, marker[0]
            elif marker[0] == fence_char:
                in_fence, fence_char = False, ""
            cur.append(line)
            continue
        if in_fence:
            cur.append(line)
            continue
        if not line.strip():
            flush()
            continue
        hm = _HEADING_RE.match(line)
        if hm:
            flush()
            level = len(hm.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, hm.group(2).strip()))
            continue
        cur.append(line)
    flush()
    return blocks


def _split_oversized(block, max_chars):
    """
    Split one oversized block into pieces of at most max_chars, cutting at
    sentence or line boundaries. Fenced code blocks are never split.
    """
    stripped = block.lstrip()
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return [block]
    parts = []
    while len(block) > max_chars:
        window = block[:max_chars]
        cuts = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", window)]
        cuts.append(window.rfind("\n"))
        cut = max(cuts)
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(block[:cut].strip())
        block = block[cut:].strip()
    if block:
        parts.append(block)
    return parts


def chunk_document_blocks(blocks, max_chars=None):
    """
    Merge consecutive blocks that share a section into rows of at most
    max_chars, never crossing a section boundary. Oversized single blocks
    (e.g. long paragraphs) are split at sentence/line boundaries; fenced
    code blocks are never split.
    Returns [(section, content)].
    """
    if max_chars is None:
        max_chars = DOC_CHUNK_MAX_CHARS
    rows = []
    cur_sec = None
    cur_parts = []
    cur_len = 0

    def flush():
        nonlocal cur_sec, cur_parts, cur_len
        if cur_parts:
            rows.append((cur_sec, "\n\n".join(cur_parts)))
        cur_sec, cur_parts, cur_len = None, [], 0

    for sec, block in blocks:
        pieces = _split_oversized(block, max_chars) if len(block) > max_chars else [block]
        for piece in pieces:
            if sec != cur_sec or (cur_parts and cur_len + len(piece) + 2 > max_chars):
                flush()
                cur_sec = sec
            cur_parts.append(piece)
            cur_len += len(piece) + 2
    flush()
    return rows


def build_rows_from_document(md_text, md_name):
    """
    Document rows: (section, content) pairs from the markdown chunker, run
    through the table row builder so documents and tables process identically.
    """
    pairs = chunk_document_blocks(markdown_blocks(md_text))
    if not pairs:
        pairs = [("", md_text.strip())]
    df = pd.DataFrame(pairs, columns=["section", "text"])
    return build_rows_from_table(df, md_name)


def write_json(subject, file_path, description, mode, data):
    base, raw_dir, json_dir, index_dir = ensure_subject_dirs(subject)
    src_name = os.path.basename(file_path)
    converted_from = None

    if mode == "table":
        # copy raw file into subject/raw
        raw_target = os.path.join(raw_dir, src_name)
        if os.path.abspath(file_path) != os.path.abspath(raw_target):
            try:
                with open(file_path, "rb") as src, open(raw_target, "wb") as dst:
                    dst.write(src.read())
            except Exception:
                pass
        id_name = src_name
        rows = build_rows_from_table(data, src_name)
        columns = list(data.columns)
    else:
        # Documents are stored as markdown: the editable source of truth.
        md_name = (
            src_name
            if os.path.splitext(src_name)[1].lower() in MARKDOWN_EXTENSIONS
            else os.path.splitext(src_name)[0] + ".md"
        )
        md_target = os.path.join(raw_dir, md_name)
        if md_name != src_name:
            converted_from = src_name
            # Never clobber an existing (possibly user-edited) conversion on
            # re-ingest; delete the .md to force a fresh conversion.
            if os.path.exists(md_target):
                try:
                    with open(md_target, "r", encoding="utf-8") as f:
                        data = f.read()
                except Exception:
                    pass
        try:
            with open(md_target, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass
        id_name = md_name
        rows = build_rows_from_document(data, md_name)
        columns = ["section", "text"]

    json_target = os.path.join(json_dir, id_name + ".json")

    meta = {
        "subject": subject,
        "description": description or "",
        "source_file": id_name,
        "columns": columns,
        "n_rows": len(rows),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode,
    }
    if converted_from:
        meta["converted_from"] = converted_from

    payload = {
        "meta": meta,
        "rows": rows,
    }

    with open(json_target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload, json_target, index_dir


def update_sqlite_and_faiss(subject, payload, index_dir):
    """
    Combined updater: SQLite + FAISS.

    SQLite is committed first (source of truth), then the FAISS index is
    synced incrementally: only new or changed chunk texts are embedded,
    removed chunks have their vectors dropped, unchanged chunks cost nothing.
    """
    conn = get_conn()

    meta = payload["meta"]
    rows = payload["rows"]

    # Unique document ID
    doc_id = f"{subject}_{meta['source_file']}"

    insert_subject(conn, subject, meta["description"], meta["created_at"])
    insert_document(conn, doc_id, subject, meta["source_file"], meta)
    doc_meta = docmeta.compute_doc_meta(
        rows, columns=meta.get("columns"), subject=subject
    )
    insert_chunks(conn, doc_id, subject, rows, doc_meta)
    pruned = prune_stale_chunks(conn)

    commit(conn)
    suffix = f" (pruned {pruned} stale chunks)" if pruned else ""
    msgs = [f"[sqlite] Updated rag.db for subject '{subject}'{suffix}"]

    index, stats = sync_subject_index(conn, subject, index_dir)
    conn.close()

    suffix = " [migrated legacy index]" if stats["migrated"] else ""
    msgs.append(
        f"[faiss] Synced subject '{subject}': +{stats['added']} added, "
        f"{stats['updated']} updated, -{stats['removed']} removed, "
        f"{stats['embedded']} embedded ({index.ntotal} total){suffix}"
    )
    return msgs


def main(subject, files, description):
    msgs_all = []

    subject = subject.strip()
    if not subject:
        raise ValueError("Subject cannot be empty.")

    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        mode, data = detect_and_load(file_path)
        if mode not in ("csv", "tsv", "json", "txt", "raw"):
            raise ValueError(f"Unsupported ingest mode: {mode}")

        payload, json_target, index_dir = write_json(
            subject, file_path, description, mode, data
        )

        msgs = update_sqlite_and_faiss(subject, payload, index_dir)
        msgs.append(
            f"Ingested {file_path} into subject '{subject}' as {mode}. JSON: {json_target}"
        )

        msgs_all.extend(msgs)

    print("\n".join(msgs_all))


if __name__ == "__main__":
    try:
        args = json.loads(sys.argv[1])

        subject = args["subject"]
        files = args["files"]
        description = args.get("description", "")

        main(subject, files, description)

    except Exception as e:
        print(f"[error] {e}")
        raise
