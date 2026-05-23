import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, Query
from app.domain.schema import EmbedResponse, DeleteResponse, CollectionDeleteResponse
from app.application import embed_service, keyword_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/embed", tags=["embed"])


def _background_extract_keywords(file_id: int, folder_id: int, file_name: str):
    try:
        keyword_service.extract_and_save(file_id=file_id, folder_id=folder_id, file_name=file_name)
    except Exception as e:
        logger.warning("백그라운드 키워드 추출 실패: file_id=%s, error=%s", file_id, e)


@router.post("", response_model=EmbedResponse)
async def embed(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_id: int = Form(...),
    file_name: str = Form(...),
    folder_id: int = Form(...),
):
    logger.info("임베딩 요청: file_id=%s, file_name=%s, folder_id=%s, size=%s bytes",
                file_id, file_name, folder_id, file.size)
    try:
        file_bytes = await file.read()
        inserted = embed_service.embed_and_store(
            file_id=file_id,
            file_name=file_name,
            folder_id=folder_id,
            file_bytes=file_bytes,
        )
        logger.info("임베딩 완료: file_id=%s, chunks_inserted=%s, collection=_%s",
                    file_id, inserted, folder_id)
        background_tasks.add_task(
            _background_extract_keywords, file_id, folder_id, file_name
        )
        return EmbedResponse(
            file_id=file_id,
            chunks_inserted=inserted,
            message="임베딩 저장 완료",
        )
    except ValueError as e:
        logger.warning("임베딩 실패 (텍스트 추출 불가): file_id=%s, error=%s", file_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("임베딩 오류: file_id=%s, error=%s", file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="임베딩 처리 중 오류가 발생했습니다.")


@router.delete("/collection/{folder_id}", response_model=CollectionDeleteResponse)
def delete_collection(folder_id: int):
    deleted = embed_service.delete_collection(folder_id)
    return CollectionDeleteResponse(
        folder_id=folder_id,
        deleted=deleted,
        message="컬렉션 삭제 완료" if deleted else "컬렉션이 존재하지 않습니다",
    )


@router.delete("/{file_id}", response_model=DeleteResponse)
def delete_embed(file_id: int, folder_id: int = Query(...)):
    deleted = embed_service.delete_embeddings(file_id, folder_id)
    keyword_service.delete_keywords(file_id)
    return DeleteResponse(
        file_id=file_id,
        deleted_count=deleted,
        message="벡터 삭제 완료",
    )
