"""JSONL read/write helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(p: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Returns [] if the file is missing."""
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(items: Iterable, path: Path) -> None:
    """Write an iterable of dataclass-or-dict items to JSONL.

    Dataclass instances are serialised via dataclasses.asdict; dicts pass through.
    """
    from dataclasses import asdict, is_dataclass
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            if is_dataclass(it):
                obj = asdict(it)
            else:
                obj = it
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
