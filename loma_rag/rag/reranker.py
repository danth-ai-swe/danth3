"""Cross-encoder reranker: lazy-load Xenova/ms-marco-MiniLM-L-6-v2 via fastembed."""
from __future__ import annotations

from loma_rag.config.settings import rag as rag_cfg


class Reranker:
    """Lazy-loading cross-encoder reranker wrapper.

    The model is downloaded on first call to _ensure(); construction is cheap.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or rag_cfg.rerank_model
        self._model = None

    def _ensure(self) -> None:
        """Lazy-load the cross-encoder model + tokenizer on first use."""
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(model_name=self._model_name)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage against `query`. Returns one float per passage."""
        self._ensure()
        return list(self._model.rerank(query, passages))
