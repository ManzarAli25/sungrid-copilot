"""Rewrite a follow-up message into a standalone question using recent conversation history.

"yes they're approved" or "how long do I have to appeal that?" mean little on their own to the
category classifier or to vector search. The standalone rewrite is used for classification,
intent and retrieval; eligibility extraction still reads the member's raw message.
"""

HISTORY_TURNS = 3  # user+assistant pairs given to the rewriter

PROMPT = """Rewrite the member's latest message as a single standalone question or statement that
can be understood without the conversation. Resolve pronouns and references ("that", "it",
"they") from the conversation. Keep the member's meaning and any details they state; do not add
facts, numbers or answers. If the message is already standalone, return it unchanged.
Return only the rewritten text.

Conversation:
{history}

Latest member message: {query}"""


def format_history(history: list[dict]) -> str:
    recent = history[-2 * HISTORY_TURNS :]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def contextualize(query: str, history: list[dict]) -> str:
    if not history:
        return query  # first turn: nothing to resolve, skip the LLM call
    from sungrid_copilot.llm import get_chat_model

    prompt = PROMPT.format(history=format_history(history), query=query)
    rewritten = get_chat_model().invoke(prompt).content.strip()
    return rewritten or query
