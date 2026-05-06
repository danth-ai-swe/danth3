"""Sentinel-detect streaming + non-streaming helpers used by the answer pipeline."""
from __future__ import annotations

from loma_rag.config.settings import azure, rag
from loma_rag.constant.tokens import INSUFFICIENT_TOKEN


def stream_with_sentinel_detect(
    client,
    system: str,
    user: str,
    sentinel: str = INSUFFICIENT_TOKEN,
    sniff_chars: int | None = None,
    model: str | None = None,
) -> tuple[str, bool]:
    """Stream the LLM response, but if `sentinel` appears in the early
    buffer the stream is silently aborted and (text, True) is returned —
    signalling the caller should fallback. Otherwise text is printed
    progressively and (text, False) is returned.
    """
    if sniff_chars is None:
        sniff_chars = rag.sniff_chars
    stream = client.chat.completions.create(
        model=model or azure.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        stream=True,
    )
    buf: list[str] = []
    decided = False
    insufficient = False
    for ev in stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta.content or ""
        if not delta:
            continue
        buf.append(delta)
        text = "".join(buf)
        if not decided:
            if sentinel in text:
                insufficient = True
                decided = True
                break
            # commit to streaming once we have enough chars to be confident
            # the sentinel won't appear (the sentinel is short enough that
            # any prefix of it will fit in sniff_chars).
            if len(text) >= sniff_chars or "\n" in text:
                if sentinel in text:
                    insufficient = True
                    decided = True
                    break
                decided = True
                print(text, end="", flush=True)
        else:
            print(delta, end="", flush=True)
    if not decided:
        text = "".join(buf)
        if sentinel in text:
            insufficient = True
        else:
            print(text, end="", flush=True)
    if not insufficient:
        print()
    return "".join(buf), insufficient


def complete_with_sentinel_detect(
    client, system: str, user: str, sentinel: str = INSUFFICIENT_TOKEN,
    model: str | None = None,
) -> tuple[str, bool]:
    resp = client.chat.completions.create(
        model=model or azure.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    insufficient = sentinel in text
    return text, insufficient
