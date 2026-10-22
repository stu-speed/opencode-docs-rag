#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.config import data_dir
from rag.sqlite import (
    SubjectInfo,
    get_subject_info,
    get_subjects,
)

RAG_ROOT = str(data_dir())

REPORT_NOTE = (
    "Note for the assistant: report ONLY what this command printed to the user — "
    "relay it in chat text verbatim (tool outputs are not visible). Do not add "
    "details that do not appear here, such as file counts."
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args: list[str]) -> None:
    # No args → clear active subjects
    if len(args) == 0:
        print("Cleared all active subjects.")
        print(REPORT_NOTE)
        return

    subjects = args
    subject_info = get_subjects(subjects)

    # Print exactly the original success format
    active_line = ", ".join(
        f"{subject} ({subject_info[subject].ntotal} chunks)"
        for subject in subjects
    )

    print(f"RAG is now enabled. Active subject(s): {active_line}.")
    print(REPORT_NOTE)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"[error] {e}")
        raise
