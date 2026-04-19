from app.infrastructure import milvus_adapter
from app.infrastructure.embedder import embedder
from app.domain.schema import SearchResult


def similarity_search(
    query: str,
    top_k: int,
    folder_ids: list[int],
) -> list[SearchResult]:
    query_embedding = embedder.encode_one(query)

    # 폴더별 검색 후 결과 합산
    all_hits = milvus_adapter.search_multiple(query_embedding, top_k, folder_ids)

    # score 내림차순 정렬
    all_hits.sort(key=lambda h: h["score"], reverse=True)

    return [
        SearchResult(
            file_id=h["file_id"],
            file_name=h["file_name"],
            folder_id=h["folder_id"],
            chunk_text=h["chunk_text"],
            chunk_index=h["chunk_index"],
            score=h["score"],
        )
        for h in all_hits
    ]
