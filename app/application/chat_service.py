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


def _rewrite_query(query: str, conversation_history: list[dict]) -> str:
    """직전 대화 맥락을 반영해 OpenSearch 검색용 독립형 쿼리로 재작성한다.
    주제 전환이 감지되거나 불확실하면 원본 쿼리를 그대로 반환한다."""
    if not conversation_history:
        return query

    recent = conversation_history[-6:]  # 최대 3쌍
    context_lines = []
    for h in recent:
        role = "사용자" if h.get("role") == "user" else "AI"
        context_lines.append(f"{role}: {h.get('content', '')[:150]}")
    context = "\n".join(context_lines)

    prompt = (
        f"직전 대화:\n{context}\n\n"
        f"현재 질문: {query}\n\n"
        "규칙:\n"
        "1. 현재 질문이 직전 대화의 연속(지시대명사·생략·후속 질문)이면 맥락을 포함한 독립형 검색 쿼리로 재작성\n"
        "2. 완전히 다른 주제이거나 '이 파일', '그거 말고' 같은 명시적 전환이면 현재 질문을 그대로 반환\n"
        "3. 판단이 불확실하면 현재 질문을 그대로 반환\n"
        "결과만 한 줄로 출력하세요."
    )

    try:
        response = _get_client().models.generate_content(
            model=settings.llm_model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(max_output_tokens=80),
        )
        rewritten = (response.text or "").strip()
        if rewritten and rewritten != query:
            logger.info("쿼리 재작성: '%s' → '%s'", query, rewritten)
        return rewritten or query
    except Exception as e:
        logger.warning("쿼리 재작성 실패, 원본 사용: %s", e)
        return query


def _filter_relevant_history(
    conversation_history: list[dict],
    query: str,
    current_tool: str = "general_chat",
    same_tool_recent: int = 5,
    cross_tool_top_k: int = 2,
    cross_tool_min_score: float = 0.5,
) -> list[dict]:
    """대화 이력을 선별한다.

    - same-tool: 최신 N쌍을 유사도 무관하게 보장 (대화 연속성 유지)
    - cross-tool: 유사도 상위 K쌍만 엄격한 임계값으로 허용
    """
    if not conversation_history:
        return []

    pairs: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(conversation_history):
        user_turn = conversation_history[i]
        bot_turn = conversation_history[i + 1] if i + 1 < len(conversation_history) else None
        pairs.append((user_turn, bot_turn))
        i += 2

    if not pairs:
        return []

    # same-tool: 최신순 same_tool_recent쌍 보장
    same_tool_indexed = [
        idx for idx, (_, bot_turn) in enumerate(pairs)
        if (bot_turn or {}).get("tool", "general_chat") == current_tool
    ]
    selected_indices = set(same_tool_indexed[-same_tool_recent:])

    # cross-tool: 유사도 상위 cross_tool_top_k쌍 (엄격한 임계값)
    cross_tool_indexed = [idx for idx in range(len(pairs)) if idx not in selected_indices]
    if cross_tool_indexed:
        query_emb = np.array(embedder.encode_one(query))
        scored_cross: list[tuple[float, int]] = []
        for idx in cross_tool_indexed:
            user_turn, bot_turn = pairs[idx]
            text = user_turn.get("content", "")
            if bot_turn:
                text += " " + bot_turn.get("content", "")[:200]
            pair_emb = np.array(embedder.encode_one(text[:500]))
            score = _cosine_similarity(query_emb, pair_emb)
            scored_cross.append((score, idx))

        scored_cross.sort(reverse=True)
        for score, idx in scored_cross[:cross_tool_top_k]:
            if score >= cross_tool_min_score:
                selected_indices.add(idx)

    # 원래 시간순 유지
    selected = [pairs[idx] for idx in sorted(selected_indices)]
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
        if not gm or not getattr(gm, "grounding_chunks", None):
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
    # 1. 쿼리 재작성: 대화 맥락 반영 (OpenSearch·RAG·히스토리 필터링에 사용)
    search_query = _rewrite_query(query, conversation_history)

    # 2. OpenSearch 라우팅 후보 탐색 (재작성 쿼리 사용)
    candidates = find_candidates(search_query)
    if folder_ids:
        folder_id_set = set(folder_ids)
        candidates = [
            c for c in candidates
            if c.target_type != "file" or c.folder_id in folder_id_set
        ]

    yield f"data: {json.dumps({'type': 'routing', 'targets': [{'name': c.name, 'type': c.target_type, 'score': round(c.score, 4)} for c in candidates]}, ensure_ascii=False)}\n\n"

    # 3. LLM 라우팅 결정 (재작성 쿼리 사용)
    tool_name = _route_with_llm(candidates, search_query)
    yield f"data: {json.dumps({'type': 'llm_decision', 'tool': tool_name}, ensure_ascii=False)}\n\n"

    # 4. Tool 실행
    if tool_name == "list_capabilities":
        yield f"data: {json.dumps({'type': 'token', 'content': _tool_info()}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'source_type': ''}, ensure_ascii=False)}\n\n"
        return

    rag_context, sources, source_type = "", [], ""
    if tool_name == "rag_search":
        rag_context, sources = _execute_rag(candidates, search_query, top_k)
        source_type = "file"

    # 5. 관련 대화 이력 필터링 (재작성 쿼리 기준 유사도 산정)
    filtered_history = _filter_relevant_history(conversation_history, search_query, current_tool=tool_name)
    contents = []
    for h in filtered_history:
        role = "model" if h.get("role") == "assistant" else h.get("role", "user")
        if role in ("user", "model"):
            contents.append(
                types.Content(role=role, parts=[types.Part(text=h.get("content", ""))])
            )
    contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

    # 6. Gemini 스트리밍 답변 (web_search는 Google Search Grounding 활성화)
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

    # 7. 완료
    yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'source_type': source_type}, ensure_ascii=False)}\n\n"
