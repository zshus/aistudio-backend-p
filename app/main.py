from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.embed_router import router as embed_router
from app.api.search_router import router as search_router
from app.infrastructure import milvus_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    milvus_adapter.connect()
    milvus_adapter.ensure_collection()
    yield


app = FastAPI(title="backend-p vector service", lifespan=lifespan)
app.include_router(embed_router)
app.include_router(search_router)


@app.get("/health")
def health():
    return {"status": "ok"}
