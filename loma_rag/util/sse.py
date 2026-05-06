"""Server-Sent Events frame encoder."""
from __future__ import annotations

import json


def sse_event(payload: dict) -> bytes:
    """Format a payload as a single SSE frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
