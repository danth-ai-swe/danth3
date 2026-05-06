"""Build user-message bodies: LOMA chunks payload, web docs payload."""
from __future__ import annotations

from loma_rag.constant.tokens import INSUFFICIENT_TOKEN
from loma_rag.model.domain import RetrievalResult, WebDoc
from loma_rag.util.language import language_name


def build_loma_user_prompt(result: RetrievalResult, user_lang: str = "en") -> str:
    lang_label = language_name(user_lang)
    parts: list[str] = [
        f"# REQUIRED OUTPUT LANGUAGE: {lang_label}",
        f"You MUST write the entire answer in {lang_label}, regardless of the "
        f"language(s) used in the LOMA excerpts (some excerpts contain bilingual "
        f"annotations — ignore that and write in {lang_label} only).",
        "",
        f"# Question\n{result.query}\n",
        "# LOMA excerpts",
    ]
    for c in result.chunks:
        header = (
            f"[{c.chunk_id}]  course={c.course}  lesson={c.lesson_id}  "
            f"section={c.section!r}  subsection={c.subsection!r}"
        )
        parts.append(f"\n## {header}\n{c.text.strip()}")

    if result.related_nodes:
        parts.append("\n# Related concepts (from knowledge graph)")
        for n in result.related_nodes:
            parts.append(f"- **{n.name}** ({n.category}): {n.definition.strip()}")

    parts.append(
        "\n# Instructions\n"
        "Answer using ONLY the excerpts. Cite inline like [LOMA281_M1L1_C033]. "
        f"If the excerpts don't contain enough info, output exactly {INSUFFICIENT_TOKEN}. "
        f"Write the answer in {lang_label}."
    )
    return "\n".join(parts)


def build_web_user_prompt(query: str, docs: list[WebDoc], user_lang: str = "en") -> str:
    lang_label = language_name(user_lang)
    parts = [
        f"# REQUIRED OUTPUT LANGUAGE: {lang_label}",
        f"You MUST write the entire answer in {lang_label}, regardless of the "
        f"language(s) used in the web results below.",
        "",
        f"# Question\n{query}\n",
        "# Web search results",
    ]
    for i, d in enumerate(docs, 1):
        body = (d.content or d.snippet).strip()
        parts.append(f"\n## [WEB_{i}] {d.title}\nURL: {d.url}\n{body}")
    parts.append(
        "\n# Instructions\n"
        "Answer using these web results. Cite sources inline like [WEB_1], [WEB_2]. "
        f"Write the answer in {lang_label}."
    )
    return "\n".join(parts)
