"""Category-filtered vector search with an unfiltered fallback for weak matches."""

from dataclasses import dataclass

from qdrant_client import models

from sungrid_copilot.config import get_settings
from sungrid_copilot.embeddings import embed_query
from sungrid_copilot.vectorstore import get_client

TOP_K = 5
# The filter is overridden (unfiltered results used) when either:
# - absolute: the best filtered hit is weaker than MIN_TOP_SCORE, or
# - relative: the best hit outside the filter beats the best filtered hit by OVERRIDE_MARGIN.
# Found by the golden set: "What rebate will the new multi-family tier get?" is confidently
# classified incentive_rebate (0.98), but the answer is in Company Updates (0.669 vs 0.530).
MIN_TOP_SCORE = 0.45
OVERRIDE_MARGIN = 0.10


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


def should_override_filter(filtered: list[RetrievedChunk], unfiltered: list[RetrievedChunk]) -> bool:
    if not filtered or filtered[0].score < MIN_TOP_SCORE:
        return True
    return bool(unfiltered) and unfiltered[0].score - filtered[0].score >= OVERRIDE_MARGIN


def retrieve(query: str, categories: list[str], k: int = TOP_K) -> RetrievalResult:
    vector = embed_query(query)  # embedded once; both searches reuse it
    filtered = _search(vector, categories, k)
    unfiltered = _search(vector, None, k)
    if should_override_filter(filtered, unfiltered):
        return RetrievalResult(unfiltered, categories, used_fallback=True)
    return RetrievalResult(filtered, categories, used_fallback=False)
