import json
import logging
from collections import defaultdict
from typing import Generator

import numpy as np
from google import genai
from google.genai import types

from app.application.routing_service import RouteCandidate, find_candidates
from app.application.search_service import similarity_search_by_files
from app.infrastructure.embedder import embedder
from app.infrastructure.opensearch_adapter import get_all_tools
from app.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


_ROUTING_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="general_chat",
        description="문서 검색 없이 일반 지식으로 답변합니다. 문서와 무관한 질문이나 일반 대화에 사용합니다.",
    ),
    types.FunctionDeclaration(
        name="rag_search",
        description="관련 문서를 검색하여 답변합니다. 파일·폴더 내용에 대한 질문에 사용합니다.",
    ),
    types.FunctionDeclaration(
        name="web_search",
        description="인터넷에서 최신 정보를 검색합니다. 실시간 정보·뉴스·외부 정보가 필요할 때 사용합니다.",
    ),
    types.FunctionDeclaration(
        name="list_capabilities",
        description="시스템이 제공하는 기능 목록을 반환합니다. '뭘 할 수 있어', '기능이 뭐야', '어떤 기능 있어' 처럼 시스템 기능을 직접 물어볼 때 사용합니다.",
    ),
])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _filter_relevant_history(
    conversation_history: list[dict],
    query: str,
    current_tool: str = "general_chat",
    top_k: int = 3,
    min_score: float = 0.25,
) -> list[dict]:
    """현재 질문과 관련된 대화 이력만 선별한다.

    - tool 타입이 다른 이력(RAG ↔ general_chat)은 임계값을 높여 오염 방지
    - 임베딩 유사도 top_k 쌍 중 min_score 이상인 쌍만 포함
    """
    if not conversation_history:
        return []

    # (user, bot) 쌍으로 묶기
    pairs: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(conversation_history):
        user_turn = conversation_history[i]
        bot_turn = conversation_history[i + 1] if i + 1 < len(conversation_history) else None
        pairs.append((user_turn, bot_turn))
        i += 2

    if not pairs:
        return []

    # 임베딩 유사도로 모든 쌍 점수화
    query_emb = np.array(embedder.encode_one(query))
    scored: list[tuple[float, int]] = []
    for idx, (user_turn, bot_turn) in enumerate(pairs):
        text = user_turn.get("content", "")
        if bot_turn:
            text += " " + bot_turn.get("content", "")[:200]
        pair_emb = np.array(embedder.encode_one(text[:500]))
        score = _cosine_similarity(query_emb, pair_emb)

        # tool 타입이 다른 이력은 유사도 임계값을 높임 (오염 방지)
        bot_tool = (bot_turn or {}).get("tool", "general_chat")
        effective_min = min_score if bot_tool == current_tool else min_score + 0.25

        scored.append((score, idx, effective_min))

    scored.sort(reverse=True)

    relevant_indices = set()
    for score, idx, effective_min in scored[:top_k]:
        if score >= effective_min:
            relevant_indices.add(idx)

    selected = [p for idx, p in enumerate(pairs) if idx in relevant_indices]
    result: list[dict] = []
    for user_turn, bot_turn in selected:
        result.append(user_turn)
        if bot_turn:
            result.append(bot_turn)

    logger.info(
        "히스토리 필터링: 전체=%d쌍 → 선택=%d쌍 (current_tool=%s)",
        len(pairs), len(selected), current_tool,
    )
    return result


def _route_with_llm(candidates: list[RouteCandidate], query: str) -> str:
    """LLM이 어떤 도구를 쓸지만 결정한다. 파일 선택은 OpenSearch 결과를 직접 사용."""
    has_file_candidates = any(c.target_type == "file" for c in candidates)
    candidate_section = f"관련 문서 후보 존재: {'예' if has_file_candidates else '아니오'}\n\n"

    prompt = (
        f"사용자 질문: {query}\n\n"
        f"{candidate_section}"
        "질문에 가장 적합한 tool을 선택하세요.\n"
        "- 인사, 감사, 잡담 등 단순 대화이면 반드시 general_chat\n"
        "- 문서·파일 내용에 대한 질문이면 rag_search\n"
        "- 최신 정보·뉴스·외부 정보가 필요하면 web_search\n"
        "- 일반 지식 질문이면 general_chat\n"
        "주의: 문서 후보가 존재하더라도 질문이 인사말이거나 문서와 무관하면 general_chat을 선택하세요."
    )

    try:
        response = _get_client().models.generate_content(
            model=settings.llm_model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                tools=[_ROUTING_TOOL],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY"),
                ),
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.function_call:
                return part.function_call.name
    except Exception as e:
        logger.error("LLM 라우팅 실패, general_chat으로 fallback: %s", e)

    return "general_chat"


def _execute_rag(
    candidates: list[RouteCandidate],
    query: str,
    top_k: int,
) -> tuple[str, list[dict]]:
    """OpenSearch가 선별한 file 후보를 folder별로 묶어 Milvus RAG 검색을 실행한다."""
    folder_file_map: dict[int, list[int]] = defaultdict(list)
    for c in candidates:
        if c.target_type == "file" and c.folder_id is not None:
            try:
                file_id = int(c.target_id.replace("file_", ""))
                folder_file_map[c.folder_id].append(file_id)
            except ValueError:
                logger.warning("file_id 파싱 실패: target_id=%s", c.target_id)

    if not folder_file_map:
        return "", []

    results = similarity_search_by_files(query=query, top_k=top_k, folder_file_map=dict(folder_file_map))
    chunks, sources, seen = [], [], set()
    for r in results:
        chunks.append(f"[{r.file_name}]\n{r.chunk_text}")
        if r.file_id not in seen:
            seen.add(r.file_id)
            sources.append({"file_name": r.file_name, "file_id": r.file_id, "score": r.score})

    return "\n\n".join(chunks), sources


def _extract_web_sources(chunk) -> list[dict]:
    """스트리밍 마지막 청크에서 Google Search grounding 출처를 추출한다."""
    try:
        gm = chunk.candidates[0].grounding_metadata
        if not gm or not hasattr(gm, "grounding_chunks"):
            return []
        sources = []
        for gc in gm.grounding_chunks:
            if hasattr(gc, "web") and gc.web:
                sources.append({"title": gc.web.title or "", "url": gc.web.uri or ""})
        return sources
    except (IndexError, AttributeError):
        return []


def _tool_info() -> str:
    tools = get_all_tools()
    lines = "\n".join(
        f"- {t['name']}: {t['description']}"
        for t in tools
        if not t.get("hidden")
    )
    return f"사용 가능한 기능:\n{lines}"


def _build_system_prompt(tool_name: str, rag_context: str) -> str:
    base = "당신은 사용자의 질문에 답변하는 AI 어시스턴트입니다."
    if tool_name == "rag_search" and rag_context:
        return (
            f"{base}\n"
            "다음 문서 내용을 참고하여 답변하세요:\n"
            "=== 참고 문서 ===\n"
            f"{rag_context}\n"
            "=== 끝 ===\n"
            "문서에 없는 내용은 모른다고 답변하세요."
        )
    if tool_name == "web_search":
        return f"{base} 검색된 최신 정보를 바탕으로 정확하게 답변하세요."
    return f"{base} 일반적인 지식으로 답변하세요."


def stream_response(
    query: str,
    conversation_history: list[dict],
    folder_ids: list[int] | None = None,
    top_k: int = 5,
) -> Generator[str, None, None]:
    # 1. OpenSearch 라우팅 후보 탐색
    candidates = find_candidates(query)
    if folder_ids:
        folder_id_set = set(folder_ids)
        candidates = [
            c for c in candidates
            if c.target_type != "file" or c.folder_id in folder_id_set
        ]

    yield f"data: {json.dumps({'type': 'routing', 'targets': [{'name': c.name, 'type': c.target_type, 'score': round(c.score, 4)} for c in candidates]}, ensure_ascii=False)}\n\n"

    # 2. LLM 라우팅 결정 (도구 선택만, 파일 선택은 OpenSearch 결과 직접 사용)
    tool_name = _route_with_llm(candidates, query)
    yield f"data: {json.dumps({'type': 'llm_decision', 'tool': tool_name}, ensure_ascii=False)}\n\n"

    # 3. Tool 실행
    if tool_name == "list_capabilities":
        yield f"data: {json.dumps({'type': 'token', 'content': _tool_info()}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'source_type': ''}, ensure_ascii=False)}\n\n"
        return

    rag_context, sources, source_type = "", [], ""
    if tool_name == "rag_search":
        rag_context, sources = _execute_rag(candidates, query, top_k)
        source_type = "file"

    # 4. 관련 대화 이력 필터링 후 Gemini 포맷 변환
    filtered_history = _filter_relevant_history(conversation_history, query, current_tool=tool_name)
    contents = []
    for h in filtered_history:
        role = "model" if h.get("role") == "assistant" else h.get("role", "user")
        if role in ("user", "model"):
            contents.append(
                types.Content(role=role, parts=[types.Part(text=h.get("content", ""))])
            )
    contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

    # 5. Gemini 스트리밍 답변 (web_search는 Google Search Grounding 활성화)
    if tool_name == "web_search":
        gen_config = types.GenerateContentConfig(
            system_instruction=_build_system_prompt(tool_name, ""),
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=2048,
        )
    else:
        gen_config = types.GenerateContentConfig(
            system_instruction=_build_system_prompt(tool_name, rag_context),
            max_output_tokens=2048,
        )

    try:
        last_chunk = None
        for chunk in _get_client().models.generate_content_stream(
            model=settings.llm_model,
            contents=contents,
            config=gen_config,
        ):
            last_chunk = chunk
            if chunk.text:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.text}, ensure_ascii=False)}\n\n"

        if tool_name == "web_search" and last_chunk:
            sources = _extract_web_sources(last_chunk)
            source_type = "web"

    except Exception as e:
        logger.error("LLM 스트리밍 오류: %s", e, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        return

    # 6. 완료
    yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'source_type': source_type}, ensure_ascii=False)}\n\n"
