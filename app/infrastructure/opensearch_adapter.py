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
