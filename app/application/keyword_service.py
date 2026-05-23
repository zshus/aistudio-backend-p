import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from app.infrastructure import milvus_adapter, opensearch_adapter
from app.infrastructure.embedder import embedder

logger = logging.getLogger(__name__)

MAX_KEYWORDS = 5
SAMPLE_FRONT = 5
SAMPLE_MID = 4
SAMPLE_BACK = 3


def _sample_chunks(chunks: list[str]) -> list[str]:
    n = len(chunks)
    if n <= SAMPLE_FRONT + SAMPLE_MID + SAMPLE_BACK:
        return chunks

    front = chunks[:SAMPLE_FRONT]
    mid_start = n // 2 - SAMPLE_MID // 2
    mid = chunks[mid_start: mid_start + SAMPLE_MID]
    back = chunks[-SAMPLE_BACK:]
    return front + mid + back


def _tfidf_keywords(chunks: list[str], n: int) -> list[str]:
    if not chunks:
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features=n * 5,
            min_df=1,
            max_df=0.95,
            token_pattern=r"(?u)\b\w{2,}\b",
        )
        X = vectorizer.fit_transform(chunks)
        scores = np.asarray(X.sum(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(scores)[::-1][:n]
        return [feature_names[i] for i in top_indices]
    except Exception as e:
        logger.warning("TF-IDF 추출 실패: %s", e)
        return []


def extract_and_save(file_id: int, folder_id: int, file_name: str) -> list[str]:
    chunks_data = milvus_adapter.query_chunks_by_file_id(file_id, folder_id)
    if not chunks_data:
        logger.warning("키워드 추출 대상 chunk 없음: file_id=%s, folder_id=%s", file_id, folder_id)
        return []

    chunks = [c["chunk_text"] for c in chunks_data]
    sampled = _sample_chunks(chunks)
    keywords = _tfidf_keywords(sampled, MAX_KEYWORDS)

    if not keywords:
        logger.warning("키워드 추출 결과 없음: file_id=%s", file_id)
        return []

    keyword_text = " ".join(keywords)
    embedding = embedder.encode_one(keyword_text)

    target_id = f"file_{file_id}"
    opensearch_adapter.upsert(
        target_id=target_id,
        target_type="file",
        name=file_name,
        keywords=keywords,
        embedding=embedding,
        folder_id=folder_id,
    )
    logger.info("키워드 사전 저장 완료: file_id=%s, keywords=%s", file_id, keywords)
    return keywords


def save_keywords(file_id: int, folder_id: int, file_name: str, keywords: list[str]) -> list[str]:
    if not keywords:
        return []
    keyword_text = " ".join(keywords)
    embedding = embedder.encode_one(keyword_text)
    opensearch_adapter.upsert(
        target_id=f"file_{file_id}",
        target_type="file",
        name=file_name,
        keywords=keywords,
        embedding=embedding,
        folder_id=folder_id,
    )
    logger.info("키워드 수동 저장 완료: file_id=%s, keywords=%s", file_id, keywords)
    return keywords


def delete_keywords(file_id: int):
    opensearch_adapter.delete(f"file_{file_id}")


def get_keywords(file_id: int) -> list[str]:
    return opensearch_adapter.get_keywords(f"file_{file_id}")
