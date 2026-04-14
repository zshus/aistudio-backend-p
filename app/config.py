from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    collection_name: str = "document_vectors"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    uploads_path: str = "/app/uploads"
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
