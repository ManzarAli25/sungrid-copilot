# integrations/

`stub_tools.py` is copied **byte-for-byte** from the assessment starter kit. It stands in
for SunGrid's Incentive Program eligibility service, which another team owns.

Do not edit it. The agent calls `check_rebate_eligibility` through its public signature
only, so swapping in the real HTTP client later shouldn't need changes elsewhere.
