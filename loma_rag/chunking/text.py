"""Plain-text normalisation helpers used during chunking."""
from __future__ import annotations

# Some Knowledge Node filenames use a non-breaking space ("Knowledge\xa0Node.xlsx").
def norm_ws(s: str) -> str:
    return s.replace("\xa0", " ")

# UI/transcript artifacts that bleed into the body text.
SKIP_PARAS = {"open transcript", "image description"}
