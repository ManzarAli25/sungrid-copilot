"""Grounded answer generation from retrieved chunks."""

from sungrid_copilot.llm import get_chat_model

SYSTEM_PROMPT = """You are SunGrid Cooperative's member support copilot.

Answer the member's question using ONLY the knowledge-base excerpts provided.
- Lead with the direct answer to what was asked, in one or two sentences. Then add only details
  that bear on the question (exact figures, deadlines, the team responsible). Leave out
  excerpts that are related but not asked about.
- Cite the excerpt header(s) you used, e.g. [Billing & Account FAQs > When are bills issued?].
- If the excerpts do not contain the answer, say so plainly and suggest contacting SunGrid
  member support. Do not fill gaps with general knowledge.
- Never decide whether the member's own household is eligible for the Rooftop Rebate, and
  never calculate a Rooftop Rebate amount for their situation. If, and only if, the member asks
  whether THEY qualify for the Rooftop Rebate or how much THEY would get, offer SunGrid's
  official eligibility check (it needs their ZIP code, household income, system size in kW, and
  whether their installer is on the approved list). The check covers only the Rooftop Rebate,
  so never offer it for other programs.
- Company updates / newsletters are not policy. Do not present them as eligibility rules."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"{c['header']}\n{c['text']}" for c in chunks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I couldn't find anything in the SunGrid knowledge base about that."
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Knowledge-base excerpts:\n\n{format_context(chunks)}\n\nMember question: {query}"),
    ]
    return get_chat_model().invoke(messages).content
