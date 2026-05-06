"""FastAPI app factory + lifespan (eager retriever/client warm-up)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from loma_rag.api.routes.chat import router as chat_router
from loma_rag.api.routes.health import router as health_router
from loma_rag.llm.embedding import embed_sparse, embed_colbert
from loma_rag.llm.openai_client import make_chat_client
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] initialising retriever and clients…")
    retriever = Retriever(rerank=True, expand_graph=True)
    chat_client = make_chat_client()
    web_fb = WebFallback(dense_client=chat_client)

    print("[startup] warming up sparse model + colbert + reranker + KG…")
    retriever._ensure_clients()
    # Sparse and ColBERT singletons live in loma_rag.llm.embedding;
    # call them directly to trigger lazy load.
    embed_sparse("warmup")
    embed_colbert("warmup")
    if retriever._reranker is not None:
        list(retriever._reranker.rerank("warmup", ["warmup passage"]))

    app.state.retriever = retriever
    app.state.chat_client = chat_client
    app.state.web_fallback = web_fb
    print("[startup] ready.")
    yield
    print("[shutdown] bye.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LOMA RAG Chatbot",
        description="Hybrid RAG (Qdrant) + GraphRAG (KG) + HyDE + SearXNG/FAISS fallback",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()
