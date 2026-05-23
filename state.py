"""
Shared state that flows through every node in the LangGraph.

Key concept: in LangGraph, agents don't pass messages to each other directly.
They all READ from and WRITE to a shared typed state object.
The graph decides which node runs next based on the state.
"""

from typing import TypedDict, Optional


class ClaimState(TypedDict):
    # Input
    claim_id: str
    clinical_notes: str          # raw doctor notes
    patient_id: str
    provider_npi: str            # National Provider Identifier
    service_date: str

    # Filled by claims_validator
    validation_errors: list[str]
    is_valid: bool

    # Filled by billing_coder
    suggested_icd10_codes: list[str]   # diagnosis codes
    suggested_cpt_codes: list[str]     # procedure codes
    coding_confidence: str             # high / medium / low

    # Filled by denial_analyzer (only if claim was denied)
    denial_reason_code: Optional[str]
    denial_explanation: str
    appeal_recommended: bool
    correction_steps: list[str]

    # Filled by supervisor — controls routing
    next_agent: str              # which agent to call next
    iteration: int               # prevent infinite loops

    # Final output
    recommendation: str          # APPROVE / CORRECT / APPEAL / REJECT
    summary: str                 # human-readable summary
