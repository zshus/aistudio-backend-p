import logging
from dataclasses import dataclass
from app.infrastructure import opensearch_adapter
from app.infrastructure.embedder import embedder
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RouteCandidate:
    target_id: str
    target_type: str
    name: str
    description: str
    keywords: list[str]
    folder_id: int | None
    score: float


def find_candidates(query: str, top_k: int = 10) -> list[RouteCandidate]:
    embedding = embedder.encode_one(query)
    raw = opensearch_adapter.hybrid_search(
        query_text=query,
        query_embedding=embedding,
        k=top_k,
    )

    threshold = settings.routing_score_threshold
    candidates = []
    for item in raw:
        if item["score"] < threshold:
            continue
        candidates.append(RouteCandidate(
            target_id=item["target_id"],
            target_type=item["target_type"],
            name=item["name"],
            description=item.get("description", ""),
            keywords=item.get("keywords", []),
            folder_id=item.get("folder_id"),
            score=item["score"],
        ))

    logger.info("라우팅 후보: query=%r, candidates=%d", query[:30], len(candidates))
    return candidates
