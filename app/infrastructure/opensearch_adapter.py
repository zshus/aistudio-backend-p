import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
from app.config import settings

logger = logging.getLogger(__name__)

INDEX = settings.opensearch_keyword_index
DIM = settings.embedding_dim


def _client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        connection_class=RequestsHttpConnection,
        use_ssl=False,
        verify_certs=False,
        timeout=10,
    )


def ensure_index():
    client = _client()
    if client.indices.exists(index=INDEX):
        return
    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "target_id":          {"type": "keyword"},
                "target_type":        {"type": "keyword"},
                "name":               {"type": "text"},
                "description":        {"type": "text"},
                "keywords_text":      {"type": "text", "analyzer": "standard"},
                "keywords_embedding": {
                    "type":      "knn_vector",
                    "dimension": DIM,
                    "method": {
                        "name":       "hnsw",
                        "engine":     "lucene",
                        "space_type": "cosinesimil",
                    },
                },
                "folder_id":  {"type": "integer"},
                "enabled":    {"type": "boolean"},
                "created_at": {"type": "date"},
            }
        },
    }
    client.indices.create(index=INDEX, body=body)
    logger.info("OpenSearch 인덱스 생성 완료: %s", INDEX)


def upsert(
    target_id: str,
    target_type: str,
    name: str,
    keywords: list[str],
    embedding: list[float],
    description: str = "",
    folder_id: int | None = None,
    enabled: bool = True,
):
    client = _client()
    doc = {
        "target_id":          target_id,
        "target_type":        target_type,
        "name":               name,
        "description":        description,
        "keywords_text":      " ".join(keywords),
        "keywords_embedding": embedding,
        "folder_id":          folder_id,
        "enabled":            enabled,
    }
    client.index(index=INDEX, id=target_id, body=doc, refresh=True)
    logger.info("키워드 사전 upsert: target_id=%s, keywords=%s", target_id, keywords)


def delete(target_id: str):
    try:
        client = _client()
        client.delete(index=INDEX, id=target_id, refresh=True)
        logger.info("키워드 사전 삭제: target_id=%s", target_id)
    except NotFoundError:
        pass


def hybrid_search(query_text: str, query_embedding: list[float], k: int = 10) -> list[dict]:
    client = _client()
    body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "keywords_embedding": {
                                "vector": query_embedding,
                                "k": k,
                            }
                        }
                    },
                    {
                        "match": {
                            "keywords_text": {
                                "query": query_text,
                                "boost": 0.3,
                            }
                        }
                    },
                ],
                "filter": [{"term": {"enabled": True}}],
            }
        },
    }
    try:
        resp = client.search(index=INDEX, body=body)
    except Exception as e:
        logger.error("OpenSearch 검색 실패: %s", e)
        return []

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "target_id":   src["target_id"],
            "target_type": src["target_type"],
            "name":        src["name"],
            "description": src.get("description", ""),
            "keywords":    src.get("keywords_text", "").split(),
            "folder_id":   src.get("folder_id"),
            "score":       hit["_score"],
        })
    return results


def get_keywords(target_id: str) -> list[str]:
    try:
        client = _client()
        doc = client.get(index=INDEX, id=target_id)
        return doc["_source"].get("keywords_text", "").split()
    except NotFoundError:
        return []


_TOOL_DEFINITIONS = [
    {
        "target_id":   "general_chat",
        "target_type": "tool",
        "name":        "일반 채팅",
        "description": "문서 검색 없이 일반 지식으로 답변",
        "keywords":    ["안녕", "인사", "대화", "설명", "도움", "질문", "답변", "알려줘", "뭐야", "어떻게", "왜", "무엇", "기능", "할 수 있어", "뭘 할", "도움말", "사용법", "소개", "능력"],
    },
    {
        "target_id":   "rag_search",
        "target_type": "tool",
        "name":        "문서 검색",
        "description": "업로드된 파일·폴더 내용을 검색하여 답변",
        "keywords":    ["문서", "파일", "자료", "내용", "보고서", "매뉴얼", "계약서", "찾아줘", "검색", "있어", "업로드"],
    },
    {
        "target_id":   "web_search",
        "target_type": "tool",
        "name":        "웹 검색",
        "description": "인터넷에서 최신 정보를 검색하여 답변",
        "keywords":    ["날씨", "뉴스", "최신", "실시간", "지금", "오늘", "현재", "주가", "환율", "인터넷", "최근"],
    },
    {
        "target_id":   "list_capabilities",
        "target_type": "tool",
        "name":        "기능 목록",
        "description": "시스템이 제공하는 기능 목록을 안내",
        "keywords":    ["기능", "할 수 있어", "뭘 할", "도움말", "사용법", "소개", "능력", "어떤 기능", "무슨 기능", "뭐 할 수"],
        "hidden":      True,
    },
]


def seed_tools(embedder) -> int:
    """general_chat / rag_search / web_search tool을 OpenSearch에 upsert한다."""
    for tool in _TOOL_DEFINITIONS:
        embedding = embedder.encode_one(" ".join(tool["keywords"]))
        upsert(
            target_id=tool["target_id"],
            target_type=tool["target_type"],
            name=tool["name"],
            keywords=tool["keywords"],
            embedding=embedding,
            description=tool["description"],
        )
    logger.info("tool seed 완료: %d개", len(_TOOL_DEFINITIONS))
    return len(_TOOL_DEFINITIONS)
