from pydantic import BaseModel
from typing import Optional


class EmbedRequest(BaseModel):
    file_id: int
    file_name: str
    folder_id: int


class EmbedResponse(BaseModel):
    file_id: int
    chunks_inserted: int
    message: str


class DeleteResponse(BaseModel):
    file_id: int
    deleted_count: int
    message: str


class CollectionDeleteResponse(BaseModel):
    folder_id: int
    deleted: bool
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    folder_ids: list[int]  # useYn=true 폴더 id 리스트 (backend-j에서 필터링 후 전달)


class SearchResult(BaseModel):
    file_id: int
    file_name: str
    folder_id: int
    chunk_text: str
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
