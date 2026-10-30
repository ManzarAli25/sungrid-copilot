"""Eligibility parameter extraction and the hard-gate verdict.

Every parameter is optional during extraction: a value the member did not state stays None
and is asked for. It is never guessed, because a guessed input yields a confident but
unfounded eligibility verdict.
"""

import re

from pydantic import BaseModel, Field

from sungrid_copilot.integrations.eligibility_tool import EligibilityResult

REQUIRED_FIELDS = ("household_zip", "annual_income_usd", "system_size_kw", "installer_approved")

FIELD_QUESTIONS = {
    "household_zip": "your 5-digit ZIP code",
    "annual_income_usd": "your household's annual income (USD)",
    "system_size_kw": "the size of your solar system in kW",
    "installer_approved": "whether your installer is on SunGrid's approved installer list",
}


class EligibilityParams(BaseModel):
    household_zip: str | None = Field(None, description="5-digit US ZIP code, exactly as stated")
    annual_income_usd: float | None = Field(
        None, description="Annual household income in US dollars, e.g. '90k' -> 90000"
    )
    system_size_kw: float | None = Field(
        None, description="Solar system size in kilowatts; convert watts to kW (6000 W -> 6.0)"
    )
    installer_approved: bool | None = Field(
        None,
        description=(
            "true ONLY if the member explicitly says their installer is SunGrid-approved / on the "
            "approved list; false if they say it is not approved or its certification lapsed; "
            "null otherwise. Never infer this from praise of the installer."
        ),
    )


def extract_params(query: str, awaiting_fields: list[str] | None = None) -> EligibilityParams:
    """Extract from the member's raw message. If the assistant just asked for specific details,
    that question is included so short replies ("yes they're approved", "about 90k") resolve.
    Previously stated values are deliberately NOT included, so they cannot be re-extracted."""
    from sungrid_copilot.llm import get_chat_model

    asked = ""
    if awaiting_fields:
        asked = (
            "The assistant just asked the member for: "
            f"{', '.join(FIELD_QUESTIONS[f] for f in awaiting_fields)}.\n"
        )
    prompt = (
        "Extract Rooftop Rebate eligibility details from the member's message. Only fill a field "
        "if the member explicitly states it; otherwise leave it null. Do not guess.\n\n"
        f"{asked}Member's message: {query}"
    )
    params = get_chat_model().with_structured_output(EligibilityParams).invoke(prompt)
    return sanitize(params)


def sanitize(params: EligibilityParams) -> EligibilityParams:
    """Drop values that cannot be valid, so they are asked for again rather than sent on."""
    data = params.model_dump()
    zip_code = (data["household_zip"] or "").strip()
    data["household_zip"] = zip_code if re.fullmatch(r"\d{5}", zip_code) else None
    if data["annual_income_usd"] is not None and data["annual_income_usd"] < 0:
        data["annual_income_usd"] = None
    if data["system_size_kw"] is not None and data["system_size_kw"] <= 0:
        data["system_size_kw"] = None
    return EligibilityParams(**data)


def merge(known: dict, new: EligibilityParams) -> dict:
    """Values stated in this turn override earlier ones; unstated ones keep earlier values."""
    return {**known, **{k: v for k, v in new.model_dump().items() if v is not None}}


def missing_fields(params: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if params.get(f) is None]


def ask_for_missing_message(missing: list[str]) -> str:
    items = [FIELD_QUESTIONS[f] for f in missing]
    listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return (
        "I can check your Rooftop Rebate eligibility with SunGrid's Incentive Program service. "
        f"To do that I still need {listed}."
    )


def verdict_message(result: EligibilityResult) -> str:
    """The hard gate: a failed check yields the reason only, never an amount or a 'maybe'."""
    if not result["eligible"]:
        return (
            "Based on SunGrid's Incentive Program eligibility service, this household is **not "
            f"eligible** for the Rooftop Rebate Program.\n\nReason: {result['reason']}\n\n"
            "No rebate amount applies. If you believe this determination is wrong, you can ask me "
            "about the rebate appeals process."
        )
    return (
        "Based on SunGrid's Incentive Program eligibility service, this household is **eligible** "
        "for the Rooftop Rebate Program.\n\n"
        f"Estimated rebate: **${result['estimated_rebate_usd']:,.2f}**\n\n"
        "This is an estimate from the eligibility check; the final amount is confirmed when your "
        "application is processed."
    )
