"""POST /quiz/chat (non-streaming) and POST /quiz/chat/stream (SSE)."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from loma_rag.model.api_models import (
    Citation, GraphNode, QuizChatData, QuizChatRequest, QuizChatResponse,
)
from loma_rag.rag.quiz_chat import run_quiz_chat, stream_quiz_chat
from loma_rag.util.sse import sse_event

router = APIRouter()


def _build_loma_citations(answer_text: str, chunks) -> list[Citation]:
    cited: list[Citation] = []
    for c in chunks:
        if c.chunk_id and c.chunk_id in (answer_text or ""):
            cited.append(Citation(
                label=c.chunk_id,
                source=c.source or None,
                lesson_id=c.lesson_id,
                course=c.course,
                section=c.section,
                subsection=c.subsection or "",
                rerank_score=float(c.rerank_score),
            ))
    return cited


def _build_web_citations(web_docs) -> list[Citation]:
    return [
        Citation(label=f"WEB_{i}", title=d.title, url=d.url, score=float(d.score))
        for i, d in enumerate(web_docs, 1)
    ]


def _build_related_nodes(related, direct_ids) -> list[GraphNode]:
    direct = set(direct_ids or [])
    return [
        GraphNode(
            node_id=n.node_id, name=n.name, category=n.category,
            definition=n.definition, lesson_id=n.lesson_id,
            direct_hit=(n.node_id in direct),
        )
        for n in (related or [])
    ]


@router.post("/quiz/chat", response_model=QuizChatResponse)
async def quiz_chat(req: QuizChatRequest, request: Request) -> QuizChatResponse:
    state = request.app.state
    try:
        result = run_quiz_chat(
            state.retriever, state.chat_client, state.web_fallback,
            req.question, req.options, req.query,
            top_k=req.top_k, web_k=req.web_k,
        )
    except Exception as e:  # noqa: BLE001
        return QuizChatResponse(success=False, data=None,
                                error=f"{type(e).__name__}: {e}")

    if result.path == "loma":
        citations = _build_loma_citations(result.message, result.loma_chunks)
    elif result.path == "web":
        citations = _build_web_citations(result.web_docs)
    else:
        citations = []

    data = QuizChatData(
        path=result.path,
        intent=result.intent,
        answer=result.answer,
        message=result.message,
        citations=citations,
        related_nodes=_build_related_nodes(result.related_nodes, result.direct_node_ids),
        en_search_query=result.en_search_query,
        web_search_used=result.web_search_used,
    )
    success = not result.error and result.path != "error"
    return QuizChatResponse(success=success, data=data, error=result.error or "")


@router.post("/quiz/chat/stream")
async def quiz_chat_stream(req: QuizChatRequest, request: Request) -> StreamingResponse:
    state = request.app.state

    def gen():
        try:
            for event in stream_quiz_chat(
                state.retriever, state.chat_client, state.web_fallback,
                req.question, req.options, req.query,
                top_k=req.top_k, web_k=req.web_k,
            ):
                if event.get("type") == "done":
                    if event["path"] == "loma":
                        cits = _build_loma_citations(event["message"], event["chunks"])
                    elif event["path"] == "web":
                        cits = _build_web_citations(event["web_docs"])
                    else:
                        cits = []
                    rels = _build_related_nodes(event["related_nodes"], event["direct_ids"])
                    yield sse_event({
                        "type": "done",
                        "intent": event["intent"],
                        "path": event["path"],
                        "answer": event.get("answer"),
                        "message": event["message"],
                        "citations": [c.model_dump() for c in cits],
                        "related_nodes": [r.model_dump() for r in rels],
                        "en_search_query": event.get("en_search_query", ""),
                        "web_search_used": event.get("web_search_used", False),
                        "error": event.get("error", ""),
                    })
                else:
                    yield sse_event(event)
        except Exception as e:  # noqa: BLE001
            yield sse_event({
                "type": "done",
                "intent": "error",
                "path": "error",
                "answer": None,
                "message": "",
                "citations": [],
                "related_nodes": [],
                "en_search_query": "",
                "web_search_used": False,
                "error": f"{type(e).__name__}: {e}",
            })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
