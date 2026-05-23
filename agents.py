"""
Specialist agents for the RCM (Revenue Cycle Management) pipeline.

Each agent is a pure function: ClaimState → ClaimState
They read what they need from state, do their work, return updated state.
The graph decides when to call each agent — agents don't call each other.
"""

import os
import json
import re
from langchain_ollama import OllamaLLM
from state import ClaimState

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

llm = OllamaLLM(model="llama3.2", base_url=OLLAMA_HOST, temperature=0)

REQUIRED_FIELDS = ["claim_id", "clinical_notes", "patient_id", "provider_npi", "service_date"]

# Valid NPI format: 10 digits
NPI_PATTERN = re.compile(r"^\d{10}$")

# Common ICD-10 prefixes for validation
VALID_ICD10_PREFIXES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def claims_validator(state: ClaimState) -> ClaimState:
    """
    Validates that all required claim fields are present and correctly formatted.
    Runs first — no point coding a claim that's missing data.
    """
    errors = []

    for field in REQUIRED_FIELDS:
        if not state.get(field):
            errors.append(f"Missing required field: {field}")

    if state.get("provider_npi") and not NPI_PATTERN.match(str(state["provider_npi"])):
        errors.append("Invalid NPI format — must be 10 digits")

    if state.get("service_date"):
        try:
            from datetime import datetime
            datetime.strptime(state["service_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("Invalid service_date format — use YYYY-MM-DD")

    if not errors and state.get("clinical_notes"):
        if len(state["clinical_notes"]) < 20:
            errors.append("Clinical notes too short — insufficient documentation")

    return {
        **state,
        "validation_errors": errors,
        "is_valid": len(errors) == 0
    }


def billing_coder(state: ClaimState) -> ClaimState:
    """
    Extracts ICD-10 (diagnosis) and CPT (procedure) codes from clinical notes.
    Uses LLM to understand medical terminology and suggest appropriate codes.
    """
    prompt = f"""You are a medical billing coder. Extract billing codes from these clinical notes.

Clinical Notes:
{state['clinical_notes']}

Respond with ONLY valid JSON in this exact format:
{{
  "icd10_codes": ["code1", "code2"],
  "cpt_codes": ["code1", "code2"],
  "confidence": "high|medium|low",
  "reasoning": "brief explanation"
}}

ICD-10 codes start with a letter followed by digits (e.g. J18.9 for pneumonia, E11.9 for diabetes).
CPT codes are 5 digits (e.g. 99213 for office visit, 93000 for ECG)."""

    response = llm.invoke(prompt)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        return {
            **state,
            "suggested_icd10_codes": parsed.get("icd10_codes", []),
            "suggested_cpt_codes": parsed.get("cpt_codes", []),
            "coding_confidence": parsed.get("confidence", "low")
        }
    except (json.JSONDecodeError, ValueError):
        return {
            **state,
            "suggested_icd10_codes": [],
            "suggested_cpt_codes": [],
            "coding_confidence": "low"
        }


def denial_analyzer(state: ClaimState) -> ClaimState:
    """
    Analyzes a claim denial reason and recommends corrective action.
    Called when a previously submitted claim came back denied.
    """
    prompt = f"""You are a healthcare claim denial specialist.

Denial Reason Code: {state.get('denial_reason_code', 'Unknown')}
Clinical Notes: {state['clinical_notes']}
Suggested Codes: ICD-10: {state.get('suggested_icd10_codes', [])} | CPT: {state.get('suggested_cpt_codes', [])}

Respond with ONLY valid JSON:
{{
  "explanation": "plain English explanation of why this was denied",
  "appeal_recommended": true|false,
  "correction_steps": ["step 1", "step 2", "step 3"]
}}"""

    response = llm.invoke(prompt)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        return {
            **state,
            "denial_explanation": parsed.get("explanation", ""),
            "appeal_recommended": parsed.get("appeal_recommended", False),
            "correction_steps": parsed.get("correction_steps", [])
        }
    except (json.JSONDecodeError, ValueError):
        return {
            **state,
            "denial_explanation": "Could not parse denial reason",
            "appeal_recommended": False,
            "correction_steps": []
        }


def supervisor(state: ClaimState) -> ClaimState:
    """
    Decides which agent to call next based on current state.
    This is the brain of the graph — all routing logic lives here.

    Routing logic:
    1. Always validate first
    2. If invalid → END (can't process invalid claim)
    3. If valid and no codes → billing_coder
    4. If denial_reason_code present and no analysis → denial_analyzer
    5. Otherwise → END (generate final recommendation)
    """
    iteration = state.get("iteration", 0)

    # Safety: prevent infinite loops
    if iteration >= 4:
        return {**state, "next_agent": "END", "iteration": iteration + 1}

    # Step 1: validation hasn't run yet
    if "is_valid" not in state:
        return {**state, "next_agent": "claims_validator", "iteration": iteration + 1}

    # Step 2: invalid claim — stop here
    if not state.get("is_valid"):
        return {**state, "next_agent": "END", "iteration": iteration + 1}

    # Step 3: valid but not coded yet
    if not state.get("suggested_icd10_codes"):
        return {**state, "next_agent": "billing_coder", "iteration": iteration + 1}

    # Step 4: denial present but not analyzed
    if state.get("denial_reason_code") and not state.get("denial_explanation"):
        return {**state, "next_agent": "denial_analyzer", "iteration": iteration + 1}

    # Step 5: all done
    return {**state, "next_agent": "END", "iteration": iteration + 1}


def generate_recommendation(state: ClaimState) -> ClaimState:
    """
    Final node — synthesizes everything into a recommendation and summary.
    Always runs last before the graph ends.
    """
    if not state.get("is_valid"):
        errors = ", ".join(state.get("validation_errors", []))
        return {
            **state,
            "recommendation": "REJECT",
            "summary": f"Claim rejected due to validation errors: {errors}"
        }

    if state.get("denial_reason_code"):
        action = "APPEAL" if state.get("appeal_recommended") else "CORRECT"
        steps = "; ".join(state.get("correction_steps", []))
        return {
            **state,
            "recommendation": action,
            "summary": f"{action} recommended. {state.get('denial_explanation', '')} Steps: {steps}"
        }

    confidence = state.get("coding_confidence", "low")
    icd10 = ", ".join(state.get("suggested_icd10_codes", []))
    cpt = ", ".join(state.get("suggested_cpt_codes", []))

    if confidence == "high":
        rec = "APPROVE"
        summary = f"Claim approved. ICD-10: {icd10} | CPT: {cpt} (confidence: {confidence})"
    else:
        rec = "CORRECT"
        summary = f"Review required — coding confidence is {confidence}. Suggested ICD-10: {icd10} | CPT: {cpt}"

    return {**state, "recommendation": rec, "summary": summary}
