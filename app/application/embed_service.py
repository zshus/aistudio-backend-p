import logging
from app.config import settings
from app.infrastructure import milvus_adapter
from app.infrastructure.text_extractor import extract_text_from_bytes, split_into_chunks
from app.infrastructure.embedder import embedder

logger = logging.getLogger(__name__)


def embed_and_store(
    file_id: int,
    file_name: str,
    folder_id: int,
    file_bytes: bytes,
) -> int:
    text = extract_text_from_bytes(file_bytes, file_name)
    logger.info("텍스트 추출 완료: file_id=%s, text_length=%s", file_id, len(text))

    chunks = split_into_chunks(text, settings.chunk_size, settings.chunk_overlap)
    logger.info("청크 분할 완료: file_id=%s, chunks=%s", file_id, len(chunks))

    if not chunks:
        raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")

    embeddings = embedder.encode(chunks)
    logger.info("임베딩 생성 완료: file_id=%s, embeddings=%s", file_id, len(embeddings))

    inserted = milvus_adapter.insert_chunks(
        file_id=file_id,
        file_name=file_name,
        folder_id=folder_id,
        chunks=chunks,
        embeddings=embeddings,
    )
    return inserted


def delete_embeddings(file_id: int, folder_id: int) -> int:
    return milvus_adapter.delete_by_file_id(file_id, folder_id)


def delete_collection(folder_id: int) -> bool:
    col_name = milvus_adapter.collection_name(folder_id)
    return milvus_adapter.drop_collection(col_name)
