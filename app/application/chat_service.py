import json
import logging
from typing import Generator

import anthropic

from app.application.routing_service import RouteCandidate, find_candidates
from app.application.search_service import similarity_search
from app.config import settings

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _build_system_prompt(candidates: list[RouteCandidate], rag_context: str) -> str:
    parts = ["당신은 사용자의 질문에 답변하는 AI 어시스턴트입니다."]

    if rag_context:
        parts.append(
            "\n다음 문서 내용을 참고하여 답변하세요:\n"
            "=== 참고 문서 ===\n"
            f"{rag_context}\n"
            "=== 끝 ===\n"
            "문서에 없는 내용은 모른다고 답변하세요."
        )
    else:
        parts.append("관련 문서를 찾지 못했습니다. 일반적인 지식으로 답변하세요.")

    return "\n".join(parts)


def _fetch_rag_context(
    candidates: list[RouteCandidate],
    query: str,
    top_k: int,
) -> tuple[str, list[dict]]:
    file_candidates = [c for c in candidates if c.target_type == "file"]
    if not file_candidates:
        return "", []

    folder_ids = list({c.folder_id for c in file_candidates if c.folder_id is not None})
    results = similarity_search(query=query, top_k=top_k, folder_ids=folder_ids)

    sources = []
    chunks = []
    seen_files = set()
    for r in results:
        chunks.append(f"[{r.file_name}]\n{r.chunk_text}")
        if r.file_id not in seen_files:
            seen_files.add(r.file_id)
            sources.append({"file_name": r.file_name, "file_id": r.file_id, "score": r.score})

    return "\n\n".join(chunks), sources


def stream_response(
    query: str,
    conversation_history: list[dict],
    folder_ids: list[int] | None = None,
    top_k: int = 5,
) -> Generator[str, None, None]:
    # 1. 라우팅 후보 탐색
    candidates = find_candidates(query)

    # folder_ids 필터 (지정된 경우 해당 폴더 후보만)
    if folder_ids:
        folder_id_set = set(folder_ids)
        candidates = [
            c for c in candidates
            if c.target_type != "file" or c.folder_id in folder_id_set
        ]

    routing_event = [
        {"name": c.name, "type": c.target_type, "score": round(c.score, 4)}
        for c in candidates
    ]
    yield f"data: {json.dumps({'type': 'routing', 'targets': routing_event}, ensure_ascii=False)}\n\n"

    # 2. 파일 RAG 컨텍스트 수집
    rag_context, sources = _fetch_rag_context(candidates, query, top_k)

    # 3. LLM 메시지 구성
    messages = []
    for h in conversation_history:
        role = h.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": h.get("content", "")})
    messages.append({"role": "user", "content": query})

    system_prompt = _build_system_prompt(candidates, rag_context)

    # 4. Claude API 스트리밍
    try:
        client = _get_client()
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error("LLM 스트리밍 오류: %s", e, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        return

    # 5. done 이벤트
    yield f"data: {json.dumps({'type': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
