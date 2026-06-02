"""
PII masker node — runs before billing_coder.

Why this node exists:
  clinical_notes contain PHI (Protected Health Information): patient names,
  dates of birth, addresses, phone numbers. Sending raw PHI to an LLM violates
  HIPAA and is a security risk.

  This node replaces PHI with tokens before the LLM ever sees the data.
  The token → real value mapping is stored in ClaimState["pii_mapping"] so
  the final output can be de-tokenized if needed.

What it masks:
  - Patient ID references in notes
  - Provider NPI in notes
  - Dates (service dates, DOBs)
  - Phone numbers
  - Email addresses
  - SSN patterns
  - Credit card patterns

What it does NOT mask (intentionally):
  - Medical terminology (ICD-10 codes, diagnoses, procedures)
    — billing_coder needs these to extract codes
  - claim_id — needed for tracing
"""

import re
from state import ClaimState


# ── PII patterns ──────────────────────────────────────────────────────────────
PII_PATTERNS = [
    # Emails
    (r'\b[\w.+-]+@[\w-]+\.\w+\b',              "EMAIL"),
    # Phone numbers (US formats: 10 digits, dashes, dots, parens)
    (r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "PHONE"),
    # SSN
    (r'\b\d{3}-\d{2}-\d{4}\b',                 "SSN"),
    # Dates — MM/DD/YYYY or YYYY-MM-DD or Month DD YYYY
    (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',    "DATE"),
    (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b', "DATE"),
    # NPI (10 digits — same pattern as provider_npi field)
    (r'\bNPI[:\s#]*\d{10}\b',                  "NPI"),
    # Patient ID patterns (common formats: P-XXXX, PT-XXXX, MRN XXXX)
    (r'\b(P|PT|MRN|PAT)[-:\s]?\d{4,10}\b',    "PATIENT_ID"),
    # Credit card
    (r'\b4[0-9]{15}\b|\b5[1-5][0-9]{14}\b',   "CC"),
]


def mask_pii(text: str) -> tuple:
    """
    Replace PII tokens in text with placeholders.
    Returns (masked_text, mapping_dict).

    mapping = {"[EMAIL_0]": "john@example.com", "[PHONE_0]": "555-1234"}
    """
    mapping = {}
    counters = {}

    for pattern, label in PII_PATTERNS:
        def make_sub(lbl):
            def sub(match):
                idx = counters.get(lbl, 0)
                counters[lbl] = idx + 1
                key = f"[{lbl}_{idx}]"
                mapping[key] = match.group()
                return key
            return sub
        text = re.sub(pattern, make_sub(label), text, flags=re.IGNORECASE)

    return text, mapping


def restore_pii(text: str, mapping: dict) -> str:
    """Reverse masking — replace tokens with original values."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


# ── LangGraph node ────────────────────────────────────────────────────────────
def pii_masker(state: ClaimState) -> ClaimState:
    """
    Masks PHI in clinical_notes before billing_coder sees them.
    Stores token → real value mapping in state for downstream de-tokenization.

    Position in graph: START → supervisor → claims_validator → pii_masker → billing_coder
    Only runs after validation passes (no point masking an invalid claim).
    """
    masked_notes, mapping = mask_pii(state["clinical_notes"])

    return {
        **state,
        "clinical_notes": masked_notes,
        "pii_mapping": mapping
    }
