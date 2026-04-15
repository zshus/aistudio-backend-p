from app.infrastructure import milvus_adapter
from app.infrastructure.embedder import embedder
from app.domain.schema import SearchResult


def similarity_search(
    query: str,
    top_k: int,
    folder_id: int,
) -> list[SearchResult]:
    query_embedding = embedder.encode_one(query)
    hits = milvus_adapter.search(query_embedding, top_k, folder_id)

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
