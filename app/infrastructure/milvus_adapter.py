from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from app.config import settings


DIM = settings.embedding_dim


def connect():
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)


def collection_name(folder_id: int) -> str:
    return f"_{folder_id}"


def ensure_collection(col_name: str) -> Collection:
    if utility.has_collection(col_name):
        col = Collection(col_name)
        col.load()
        return col

    fields = [
        FieldSchema("id",          DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema("file_id",     DataType.INT64),
        FieldSchema("file_name",   DataType.VARCHAR,      max_length=255),
        FieldSchema("folder_id",   DataType.INT64),
        FieldSchema("chunk_text",  DataType.VARCHAR,      max_length=2000),
        FieldSchema("chunk_index", DataType.INT32),
        FieldSchema("embedding",   DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="Document vector store")
    col = Collection(col_name, schema)

    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    col.load()
    return col


def drop_collection(col_name: str) -> bool:
    if utility.has_collection(col_name):
        utility.drop_collection(col_name)
        return True
    return False


def insert_chunks(
    file_id: int,
    file_name: str,
    folder_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    col = ensure_collection(collection_name(folder_id))
    data = [
        [file_id] * len(chunks),
        [file_name] * len(chunks),
        [folder_id] * len(chunks),
        [c[:2000] for c in chunks],
        list(range(len(chunks))),
        embeddings,
    ]
    result = col.insert(data)
    col.flush()
    return len(result.primary_keys)


def delete_by_file_id(file_id: int, folder_id: int) -> int:
    col_name = collection_name(folder_id)
    if not utility.has_collection(col_name):
        return 0
    col = Collection(col_name)
    col.load()
    result = col.delete(f"file_id == {file_id}")
    col.flush()
    return result.delete_count


def search(
    query_embedding: list[float],
    top_k: int,
    folder_id: int,
) -> list[dict]:
    col_name = collection_name(folder_id)
    if not utility.has_collection(col_name):
        return []

    col = ensure_collection(col_name)
    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=f"folder_id == {folder_id}",
        output_fields=["file_id", "file_name", "folder_id", "chunk_text", "chunk_index"],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "file_id":     hit.entity.get("file_id"),
            "file_name":   hit.entity.get("file_name"),
            "folder_id":   hit.entity.get("folder_id"),
            "chunk_text":  hit.entity.get("chunk_text"),
            "chunk_index": hit.entity.get("chunk_index"),
            "score":       hit.score,
        })
    return hits


def query_chunks_by_file_id(file_id: int, folder_id: int) -> list[dict]:
    col_name = collection_name(folder_id)
    if not utility.has_collection(col_name):
        return []
    col = Collection(col_name)
    col.load()
    results = col.query(
        expr=f"file_id == {file_id}",
        output_fields=["chunk_text", "chunk_index"],
        limit=16384,
    )
    return sorted(results, key=lambda x: x["chunk_index"])


def search_multiple(
    query_embedding: list[float],
    top_k: int,
    folder_ids: list[int],
) -> list[dict]:
    """여러 폴더(컬렉션)를 순회하며 검색 후 결과를 합산하여 반환"""
    all_hits = []
    for folder_id in folder_ids:
        try:
            hits = search(query_embedding, top_k, folder_id)
            all_hits.extend(hits)
        except Exception as e:
            # 특정 폴더 검색 실패 시 해당 폴더만 건너뜀
            print(f"[WARN] folderId={folder_id} 검색 실패: {e}")
    return all_hits
