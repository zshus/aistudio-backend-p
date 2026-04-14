from sentence_transformers import SentenceTransformer
from app.config import settings


class Embedder:
    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


embedder = Embedder()
