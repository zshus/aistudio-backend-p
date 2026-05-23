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

    opensearch_host: str = "opensearch"
    opensearch_port: int = 9200
    opensearch_keyword_index: str = "routing_keywords"
    routing_score_threshold: float = 0.4

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"


settings = Settings()
