from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.infrastructure import opensearch_adapter
from app.infrastructure.embedder import embedder

router = APIRouter(prefix="/v1/tools", tags=["tools"])


class ToolUpsertRequest(BaseModel):
    tool_id: str
    name: str
    description: str = ""
    keywords: list[str]
    use_yn: bool = True
    hidden: bool = False


class ToolSyncRequest(BaseModel):
    tools: list[ToolUpsertRequest]


@router.post("/upsert")
def upsert_tool(req: ToolUpsertRequest):
    embedding = embedder.encode_one(" ".join(req.keywords))
    opensearch_adapter.upsert(
        target_id=req.tool_id,
        target_type="tool",
        name=req.name,
        keywords=req.keywords,
        embedding=embedding,
        description=req.description,
        enabled=req.use_yn,
    )
    return {"message": f"tool upsert 완료: {req.tool_id}"}


@router.delete("/{tool_id}")
def delete_tool(tool_id: str):
    opensearch_adapter.delete(tool_id)
    return {"message": f"tool 삭제 완료: {tool_id}"}


@router.post("/sync")
def sync_tools(req: ToolSyncRequest):
    for tool in req.tools:
        embedding = embedder.encode_one(" ".join(tool.keywords))
        opensearch_adapter.upsert(
            target_id=tool.tool_id,
            target_type="tool",
            name=tool.name,
            keywords=tool.keywords,
            embedding=embedding,
            description=tool.description,
            enabled=tool.use_yn,
        )
    return {"message": f"tool sync 완료: {len(req.tools)}개"}
