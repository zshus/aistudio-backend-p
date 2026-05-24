import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.application import chat_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ConversationMessage(BaseModel):
    role: str
    content: str


class ChatQueryRequest(BaseModel):
    query: str
    conversation_history: list[ConversationMessage] = []
    folder_ids: list[int] | None = None
    top_k: int = 5


@router.post("/query")
def chat_query(req: ChatQueryRequest):
    logger.info("채팅 쿼리: query=%r, folder_ids=%s", req.query[:50], req.folder_ids)

    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]

    import json as _json

    def event_stream():
        try:
            yield from chat_service.stream_response(
                query=req.query,
                conversation_history=history,
                folder_ids=req.folder_ids,
                top_k=req.top_k,
            )
        except Exception as e:
            logger.error("event_stream 오류: %s", e, exc_info=True)
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
