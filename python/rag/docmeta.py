"""
Generic document-level metadata (doc_meta) derivation.

doc_meta is a document's leading context: its title and the first chunk(s)
(typically the metadata/abstract) plus any later Abstract/Summary section.
It is derived purely from the chunk structure ingest already produces (the
per-chunk {"section","text"} JSON + row_index) — no format parsing, no
DOI/year extraction, no model calls — so it behaves identically for papers,
READMEs, and any other document. Table rows carry no "section", so they fall
back to their column list.
"""
import json

from .config import doc_meta_max_chars, doc_meta_preamble_sections


def _row_body(row_text):
    """(section, body) parsed from a chunk's stored JSON text, or (None, "")."""
    try:
        rec = json.loads(row_text)
    except (TypeError, ValueError):
        return None, ""
    if not isinstance(rec, dict):
        return None, ""
    return rec.get("section"), rec.get("text", "")


def _cap(text, max_chars):
    """Truncate to max_chars, preferring a clean newline boundary."""
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars // 2:
        cut = cut[:nl]
    return cut


def compute_doc_meta(rows, columns=None, subject=None,
                     preamble_sections=None, max_chars=None):
    """
    Build the document-level metadata string for one document.

    rows:       the document's chunk rows; each dict's "text" is the stored
                {"section","text"} JSON for documents or a record for tables.
    columns:    table column names, used when rows carry no "section".
    subject:    used to look up a per-subject preamble-sections override.

    Returns a capped plain-text string (may be "").
    """
    if max_chars is None:
        max_chars = doc_meta_max_chars()
    if preamble_sections is None:
        preamble_sections = doc_meta_preamble_sections(subject)

    parsed = []
    for i, row in enumerate(rows):
        sec, body = _row_body(row.get("text"))
        parsed.append((row.get("row_index", i), sec, body))

    # Document vs table: a document row's JSON has a "section" key.
    has_section = any(sec is not None for _ri, sec, _b in parsed)
    if not has_section:
        return _cap(" | ".join(str(c) for c in (columns or [])), max_chars)

    ordered = sorted(parsed, key=lambda t: t[0])
    suffixes = [p.strip().lower() for p in preamble_sections if p.strip()]

    # Leading context = the first two chunks (title/preamble + the abstract or
    # first section) plus any later Abstract/Summary chunk. Generic: it does
    # not rely on a specific heading name to find the title.
    parts = []
    for idx, (_ri, sec, body) in enumerate(ordered):
        sec = (sec or "").strip()
        if idx < 2:
            if sec and len(sec) <= 80 and ">" not in sec:
                parts.append(sec)
            if body:
                parts.append(body)
        else:
            last = sec.split(">")[-1].strip().lower()
            if body and any(last.endswith(n) for n in suffixes):
                parts.append(body)
    return _cap("\n\n".join(p for p in parts if p), max_chars)
