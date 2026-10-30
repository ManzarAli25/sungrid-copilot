"""Query classifier: assigns ONE primary category, plus the categories to filter retrieval on.

Two interchangeable backends read the same category descriptions from categories.yaml:
- Jev (TypeSafe): a typed Choice question with calibrated per-category probabilities.
- LLM fallback (OpenRouter): structured output, used when no TYPESAFE_API_KEY is set.

The classifier never routes the pipeline. Its output only narrows retrieval.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, Field

from sungrid_copilot.config import get_settings, load_category_config

NON_RELEVANT = "non_relevant"
# A category joins the retrieval filter if its probability reaches this. Picked from probes:
# "my rebate hasn't shown up on my bill" → incentive_rebate 0.84 / billing_account 0.16, and
# the correct answer lives in a billing_account doc. A top-only filter would hide it.
SECONDARY_MIN_PROB = 0.10
MAX_FILTER_CATEGORIES = 3

JEV_INSTRUCTIONS = "Which knowledge-base category does this SunGrid Cooperative member query belong to"


@dataclass
class Classification:
    primary: str  # the single primary category required by the brief
    filter_categories: list[str]  # what retrieval filters on; empty = skip retrieval
    probabilities: dict[str, float] = field(default_factory=dict)  # empty for the LLM backend
    backend: str = ""

    @property
    def is_relevant(self) -> bool:
        return self.primary != NON_RELEVANT


def category_descriptions() -> dict[str, str]:
    return {name: c["description"].strip() for name, c in load_category_config()["categories"].items()}


def select_filter_categories(probabilities: dict[str, float], mode: str = "widened") -> list[str]:
    """Primary category plus any plausible runner-ups ("widened"), or primary only ("strict").

    non_relevant is never a filter value.
    """
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if ranked[0][0] == NON_RELEVANT:
        return []
    if mode == "strict":
        return [ranked[0][0]]
    picked = [cat for cat, p in ranked if p >= SECONDARY_MIN_PROB and cat != NON_RELEVANT]
    return picked[:MAX_FILTER_CATEGORIES] or [ranked[0][0]]


# --- Jev backend --------------------------------------------------------------------------


def classify_with_jev(query: str) -> Classification:
    from typesafe_sdk import Choice, TypeSafeClient

    s = get_settings()
    with TypeSafeClient(model=s.jev_model, api_key=s.typesafe_api_key) as client:
        response = client.system_one(
            state=query,
            questions={"category": Choice(instructions=JEV_INSTRUCTIONS, criteria=category_descriptions())},
        )
    answer = response.choices["category"]
    return Classification(
        primary=answer.choice,
        filter_categories=select_filter_categories(answer.probabilities, s.filter_mode),
        probabilities=dict(answer.probabilities),
        backend="jev",
    )


# --- LLM fallback backend -----------------------------------------------------------------


class _LLMClassification(BaseModel):
    primary: str = Field(description="The single best-matching category name")
    also_relevant: list[str] = Field(
        default_factory=list,
        description="Other categories that could plausibly hold the answer (often empty)",
    )


def classify_with_llm(query: str) -> Classification:
    from sungrid_copilot.llm import get_chat_model

    descriptions = category_descriptions()
    menu = "\n".join(f"- {name}: {desc}" for name, desc in descriptions.items())
    prompt = (
        "Classify a SunGrid Cooperative member query into the knowledge-base categories below.\n"
        "Pick one primary category. List other categories in also_relevant only if the answer "
        "could plausibly live there too.\n\n"
        f"Categories:\n{menu}\n\nQuery: {query}"
    )
    result = get_chat_model().with_structured_output(_LLMClassification).invoke(prompt)

    primary = result.primary if result.primary in descriptions else NON_RELEVANT
    if primary == NON_RELEVANT:
        filters: list[str] = []
    elif get_settings().filter_mode == "strict":
        filters = [primary]
    else:
        extras = [c for c in result.also_relevant if c in descriptions and c not in (primary, NON_RELEVANT)]
        filters = ([primary] + extras)[:MAX_FILTER_CATEGORIES]
    return Classification(primary=primary, filter_categories=filters, backend="llm")


# --- entry point --------------------------------------------------------------------------


@lru_cache
def _backend() -> str:
    s = get_settings()
    if s.classifier_backend in ("jev", "llm"):
        return s.classifier_backend
    return "jev" if s.typesafe_api_key else "llm"


def classify(query: str) -> Classification:
    return classify_with_jev(query) if _backend() == "jev" else classify_with_llm(query)
