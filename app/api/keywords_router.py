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


@router.get("/{file_id}", response_model=KeywordGetResponse)
def get_keywords(file_id: int):
    keywords = keyword_service.get_keywords(file_id)
    return KeywordGetResponse(file_id=file_id, keywords=keywords)
