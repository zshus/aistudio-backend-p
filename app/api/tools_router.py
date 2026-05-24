from fastapi import APIRouter
from app.infrastructure import opensearch_adapter
from app.infrastructure.embedder import embedder

router = APIRouter(prefix="/v1/tools", tags=["tools"])


@router.post("/seed")
def seed_tools():
    count = opensearch_adapter.seed_tools(embedder)
    return {"message": f"tool seed 완료: {count}개"}
