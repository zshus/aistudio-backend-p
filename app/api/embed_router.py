from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.domain.schema import EmbedResponse, DeleteResponse
from app.application import embed_service

router = APIRouter(prefix="/api/v1/embed", tags=["embed"])


@router.post("", response_model=EmbedResponse)
async def embed(
    file: UploadFile = File(...),
    file_id: int = Form(...),
    file_name: str = Form(...),
    folder_id: int = Form(...),
):
    try:
        file_bytes = await file.read()
        inserted = embed_service.embed_and_store(
            file_id=file_id,
            file_name=file_name,
            folder_id=folder_id,
            file_bytes=file_bytes,
        )
        return EmbedResponse(
            file_id=file_id,
            chunks_inserted=inserted,
            message="임베딩 저장 완료",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/{file_id}", response_model=DeleteResponse)
def delete_embed(file_id: int):
    deleted = embed_service.delete_embeddings(file_id)
    return DeleteResponse(
        file_id=file_id,
        deleted_count=deleted,
        message="벡터 삭제 완료",
    )
