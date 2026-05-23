import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
from app.api.embed_router import router as embed_router
from app.api.search_router import router as search_router
from app.api.keywords_router import router as keywords_router
from app.api.chat_router import router as chat_router
from app.infrastructure import milvus_adapter, opensearch_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    milvus_adapter.connect()
    opensearch_adapter.ensure_index()
    yield


app = FastAPI(title="backend-p vector service", lifespan=lifespan)
app.include_router(embed_router)
app.include_router(search_router)
app.include_router(keywords_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
