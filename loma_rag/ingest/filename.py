"""Parse LOMA-style course filenames (e.g. LOMA281_M2L3_Foo.docx)."""
from __future__ import annotations

import re

FILENAME_RE = re.compile(r"LOMA(?P<course>\d+)_M(?P<module>\d+)L(?P<lesson>\d+)_")


def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.search(name)
    if not m:
        return None
    return {
        "course": f"LOMA{m['course']}",
        "module": f"M{m['module']}",
        "lesson": f"L{m['lesson']}",
        "lesson_id": f"LOMA{m['course']}_M{m['module']}L{m['lesson']}",
    }
