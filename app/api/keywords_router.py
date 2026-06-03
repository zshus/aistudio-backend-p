import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.application import keyword_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/keywords", tags=["keywords"])


class KeywordExtractRequest(BaseModel):
    file_id: int
    folder_id: int
    file_name: str = ""


class KeywordExtractResponse(BaseModel):
    file_id: int
    keywords: list[str]


class KeywordGetResponse(BaseModel):
    file_id: int
    keywords: list[str]


@router.post("/extract", response_model=KeywordExtractResponse)
def extract_keywords(req: KeywordExtractRequest):
    logger.info("키워드 추출 요청: file_id=%s, folder_id=%s", req.file_id, req.folder_id)
    try:
        keywords = keyword_service.extract_and_save(
            file_id=req.file_id,
            folder_id=req.folder_id,
            file_name=req.file_name,
        )
        return KeywordExtractResponse(file_id=req.file_id, keywords=keywords)
    except Exception as e:
        logger.error("키워드 추출 오류: file_id=%s, error=%s", req.file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="키워드 추출 중 오류가 발생했습니다.")


class KeywordSaveRequest(BaseModel):
    file_id: int
    folder_id: int
    file_name: str = ""
    keywords: list[str]


@router.post("/save", response_model=KeywordExtractResponse)
def save_keywords(req: KeywordSaveRequest):
    logger.info("키워드 저장 요청: file_id=%s, keywords=%s", req.file_id, req.keywords)
    try:
        keywords = keyword_service.save_keywords(
            file_id=req.file_id,
            folder_id=req.folder_id,
            file_name=req.file_name,
            keywords=req.keywords,
        )
        return KeywordExtractResponse(file_id=req.file_id, keywords=keywords)
    except Exception as e:
        logger.error("키워드 저장 오류: file_id=%s, error=%s", req.file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="키워드 저장 중 오류가 발생했습니다.")


class KeywordEnabledRequest(BaseModel):
    file_id: int
    enabled: bool


@router.patch("/enabled")
def update_enabled(req: KeywordEnabledRequest):
    logger.info("enabled 업데이트 요청: file_id=%s, enabled=%s", req.file_id, req.enabled)
    try:
        keyword_service.update_enabled(file_id=req.file_id, enabled=req.enabled)
        return {"file_id": req.file_id, "enabled": req.enabled}
    except Exception as e:
        logger.error("enabled 업데이트 오류: file_id=%s, error=%s", req.file_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="enabled 업데이트 중 오류가 발생했습니다.")


class FolderEnabledRequest(BaseModel):
    folder_id: int
    enabled: bool


@router.patch("/folder/enabled")
def update_enabled_by_folder(req: FolderEnabledRequest):
    logger.info("폴더 enabled 업데이트 요청: folder_id=%s, enabled=%s", req.folder_id, req.enabled)
    try:
        keyword_service.update_enabled_by_folder_id(folder_id=req.folder_id, enabled=req.enabled)
        return {"folder_id": req.folder_id, "enabled": req.enabled}
    except Exception as e:
        logger.error("폴더 enabled 업데이트 오류: folder_id=%s, error=%s", req.folder_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="폴더 enabled 업데이트 중 오류가 발생했습니다.")


@router.delete("/folder/{folder_id}")
def delete_keywords_by_folder(folder_id: int):
    logger.info("폴더 키워드 삭제 요청: folder_id=%s", folder_id)
    try:
        keyword_service.delete_by_folder_id(folder_id)
        return {"folder_id": folder_id, "deleted": True}
    except Exception as e:
        logger.error("폴더 키워드 삭제 오류: folder_id=%s, error=%s", folder_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="폴더 키워드 삭제 중 오류가 발생했습니다.")


@router.delete("/{file_id}")
def delete_keywords(file_id: int):
    logger.info("키워드 삭제 요청: file_id=%s", file_id)
    keyword_service.delete_keywords(file_id)
    return {"file_id": file_id, "deleted": True}


@router.get("/{file_id}", response_model=KeywordGetResponse)
def get_keywords(file_id: int):
    keywords = keyword_service.get_keywords(file_id)
    return KeywordGetResponse(file_id=file_id, keywords=keywords)
