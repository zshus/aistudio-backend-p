from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from app.config import settings


COLLECTION_NAME = settings.collection_name
DIM = settings.embedding_dim


def connect():
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)


def ensure_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

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
    collection = Collection(COLLECTION_NAME, schema)

    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    collection.load()
    return collection


def insert_chunks(
    file_id: int,
    file_name: str,
    folder_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    collection = ensure_collection()
    data = [
        [file_id] * len(chunks),
        [file_name] * len(chunks),
        [folder_id] * len(chunks),
        [c[:2000] for c in chunks],
        list(range(len(chunks))),
        embeddings,
    ]
    result = collection.insert(data)
    collection.flush()
    return len(result.primary_keys)


def delete_by_file_id(file_id: int) -> int:
    collection = ensure_collection()
    result = collection.delete(f"file_id == {file_id}")
    collection.flush()
    return result.delete_count


def search(
    query_embedding: list[float],
    top_k: int,
    folder_id: int | None = None,
) -> list[dict]:
    collection = ensure_collection()

    expr = f"folder_id == {folder_id}" if folder_id is not None else None

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
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
