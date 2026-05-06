"""POST /chat (non-streaming) and /chat/stream (SSE)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from loma_rag.model.api_models import ChatData, ChatRequest, ChatResponse, Citation, GraphNode
from loma_rag.rag.pipeline import run_query, stream_query
from loma_rag.util.sse import sse_event

router = APIRouter()


_PATH_TO_INTENT = {
    "off_topic": "off_topic",
    "unsupported_language": "unsupported_language",
    "quiz": "quiz",
}


def _classify_intent(path: str) -> str:
    return _PATH_TO_INTENT.get(path, "insurance")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    state = request.app.state
    try:
        result = run_query(
            state.retriever, state.chat_client, state.web_fallback,
            req.query, top_k=req.top_k, web_k=req.web_k,
        )
    except Exception as e:  # noqa: BLE001
        return ChatResponse(success=False, data=None, error=f"{type(e).__name__}: {e}")

    # Build citations: only chunks/docs actually cited in the answer.
    citations: list[Citation] = []
    if result.used_path == "loma":
        for c in result.loma_chunks:
            if c.chunk_id and c.chunk_id in (result.answer_text or ""):
                citations.append(Citation(
                    label=c.chunk_id,
                    source=c.source or None,
                    lesson_id=c.lesson_id,
                    course=c.course,
                    section=c.section,
                    subsection=c.subsection or "",
                    rerank_score=float(c.rerank_score),
                ))
    elif result.used_path == "web":
        for i, d in enumerate(result.web_docs, 1):
            citations.append(Citation(
                label=f"WEB_{i}",
                title=d.title,
                url=d.url,
                score=float(d.score),
            ))

    # Build the graph-node block from KG-expanded nodes (chunk-derived AND
    # direct semantic hits). Mark which ones were direct hits.
    direct_ids = set()
    if hasattr(result, "loma_chunks") and result.loma_chunks:
        # `direct_node_ids` lives on RetrievalResult; expose via result if present.
        direct_ids = set(getattr(result, "direct_node_ids", []) or [])
    related_nodes: list[GraphNode] = []
    for n in (result.related_nodes or []):
        related_nodes.append(GraphNode(
            node_id=n.node_id,
            name=n.name,
            category=n.category,
            definition=n.definition,
            lesson_id=n.lesson_id,
            direct_hit=(n.node_id in direct_ids),
        ))

    data = ChatData(
        path=result.used_path,
        answer=result.answer_text,
        citations=citations,
        related_nodes=related_nodes,
        en_search_query=result.en_search_query,
        intent=_classify_intent(result.used_path),
        web_search_used=(result.used_path == "web"),
    )
    success = not result.error and result.used_path != "error"
    return ChatResponse(success=success, data=data, error=result.error or "")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    state = request.app.state

    def gen():
        try:
            for event in stream_query(
                    state.retriever, state.chat_client, state.web_fallback,
                    req.query, top_k=req.top_k, web_k=req.web_k,
            ):
                yield sse_event(event)
        except Exception as e:  # noqa: BLE001
            yield sse_event({"type": "done", "path": "error",
                             "error": f"{type(e).__name__}: {e}",
                             "answer_text": "", "citations": [], "timings": {}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
