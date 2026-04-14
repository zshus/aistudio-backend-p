from fastapi import APIRouter, HTTPException
from app.domain.schema import SearchRequest, SearchResponse
from app.application import search_service

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.top_k < 1 or req.top_k > 100:
        raise HTTPException(status_code=422, detail="top_k는 1~100 사이여야 합니다.")

    results = search_service.similarity_search(
        query=req.query,
        top_k=req.top_k,
        folder_id=req.folder_id,
    )
    return SearchResponse(query=req.query, results=results)
