from app.infrastructure import milvus_adapter
from app.infrastructure.embedder import embedder
from app.domain.schema import SearchResult


def similarity_search(
    query: str,
    top_k: int,
    folder_ids: list[int],
) -> list[SearchResult]:
    query_embedding = embedder.encode_one(query)
    all_hits = milvus_adapter.search_multiple(query_embedding, top_k, folder_ids)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return _to_results(all_hits)


def similarity_search_by_files(
    query: str,
    top_k: int,
    folder_file_map: dict[int, list[int]],
) -> list[SearchResult]:
    """OpenSearch가 선별한 folder→file_ids 기반으로 Milvus 검색"""
    query_embedding = embedder.encode_one(query)
    all_hits = milvus_adapter.search_by_files(query_embedding, top_k, folder_file_map)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return _to_results(all_hits)


def _to_results(hits: list[dict]) -> list[SearchResult]:
    return [
        SearchResult(
            file_id=h["file_id"],
            file_name=h["file_name"],
            folder_id=h["folder_id"],
            chunk_text=h["chunk_text"],
            chunk_index=h["chunk_index"],
            score=h["score"],
        )
        for h in hits
    ]
