"""Category-filtered vector search with an unfiltered fallback for weak matches."""

from dataclasses import dataclass

from qdrant_client import models

from sungrid_copilot.config import get_settings
from sungrid_copilot.embeddings import embed_query
from sungrid_copilot.vectorstore import get_client

TOP_K = 5
# If the best filtered hit is weaker than this, the filter probably excluded the answer
# (misclassification), so we search again without it.
MIN_TOP_SCORE = 0.45


@dataclass
class RetrievedChunk:
    chunk_id: str
    header: str
    text: str
    categories: list[str]
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    filter_categories: list[str]  # the filter that was requested
    used_fallback: bool  # True if the unfiltered search was used instead


def _search(vector: list[float], categories: list[str] | None, k: int) -> list[RetrievedChunk]:
    query_filter = None
    if categories:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="categories", match=models.MatchAny(any=categories))]
        )
    hits = get_client().query_points(
        get_settings().qdrant_collection, query=vector, query_filter=query_filter, limit=k
    ).points
    return [
        RetrievedChunk(
            chunk_id=h.payload["chunk_id"],
            header=h.payload["header"],
            text=h.payload["text"],
            categories=h.payload["categories"],
            score=h.score,
        )
        for h in hits
    ]


def retrieve(query: str, categories: list[str], k: int = TOP_K) -> RetrievalResult:
    vector = embed_query(query)
    chunks = _search(vector, categories, k)
    if chunks and chunks[0].score >= MIN_TOP_SCORE:
        return RetrievalResult(chunks, categories, used_fallback=False)
    return RetrievalResult(_search(vector, None, k), categories, used_fallback=True)
