"""LangChain tool wrapper around the eligibility service stub.

The stub is used exactly as provided: same function, same signature, same return shape.
Swapping in the real service later means changing only the import below.
"""

from langchain_core.tools import StructuredTool

from sungrid_copilot.integrations.stub_tools import EligibilityResult, check_rebate_eligibility

eligibility_tool = StructuredTool.from_function(
    func=check_rebate_eligibility,
    name="check_rebate_eligibility",
    description=(
        "SunGrid Incentive Program service: authoritative Rooftop Rebate Program eligibility "
        "check and rebate estimate for one household."
    ),
)

__all__ = ["eligibility_tool", "EligibilityResult"]
