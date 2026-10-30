"""Intent detection: should this turn run the eligibility check, or be answered from documents?

This is the agent's routing decision (Part B). It is separate from the category classifier,
which only filters retrieval (Part A).
"""

from typing import Literal

from pydantic import BaseModel

from sungrid_copilot.config import get_settings

ELIGIBILITY = "eligibility_check"
GENERAL = "general_question"

INTENTS = {
    ELIGIBILITY: (
        "The member wants to know whether THEIR OWN household qualifies for the Rooftop Rebate "
        "Program, or how much rooftop rebate THEY would receive, or is supplying personal details "
        "(ZIP code, income, system size, installer) for that check."
    ),
    GENERAL: (
        "Any other question, including general questions about how rebate rules work, other "
        "programs (battery add-on, low-income credit, commercial incentive), billing, policies, "
        "installation, or company news."
    ),
}


def _state_text(query: str, awaiting_fields: list[str], known_params: dict) -> str:
    """The member's message plus conversation context, so follow-ups like "what if it were
    9 kW?" are recognised as eligibility re-checks rather than general questions."""
    context = []
    if known_params:
        stated = ", ".join(f"{k}={v}" for k, v in known_params.items())
        context.append(
            "Earlier in this conversation the member gave Rooftop Rebate eligibility details: "
            f"{stated}. A message that changes or adds one of these details is an eligibility check."
        )
    if awaiting_fields:
        context.append(
            "The assistant just asked the member for these eligibility details: "
            f"{', '.join(awaiting_fields)}."
        )
    if not context:
        return query
    return "\n".join(context) + f"\nMember's message: {query}"


def _detect_with_jev(text: str) -> str:
    from typesafe_sdk import Choice, TypeSafeClient

    s = get_settings()
    with TypeSafeClient(model=s.jev_model, api_key=s.typesafe_api_key) as client:
        response = client.system_one(
            state=text,
            questions={"intent": Choice(instructions="What does the member want in this turn", criteria=INTENTS)},
        )
    return response.choices["intent"].choice


class _LLMIntent(BaseModel):
    intent: Literal["eligibility_check", "general_question"]


def _detect_with_llm(text: str) -> str:
    from sungrid_copilot.llm import get_chat_model

    menu = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    prompt = f"Classify the member's intent.\n\nIntents:\n{menu}\n\nConversation:\n{text}"
    return get_chat_model().with_structured_output(_LLMIntent).invoke(prompt).intent


def detect_intent(
    query: str, awaiting_fields: list[str] | None = None, known_params: dict | None = None
) -> str:
    from sungrid_copilot.classifier.classifier import _backend

    text = _state_text(query, awaiting_fields or [], known_params or {})
    return _detect_with_jev(text) if _backend() == "jev" else _detect_with_llm(text)
