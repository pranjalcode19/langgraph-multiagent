import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from graph import rcm_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = FastAPI(title="LangGraph RCM Multi-Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ClaimRequest(BaseModel):
    claim_id: str
    clinical_notes: str
    patient_id: str
    provider_npi: str
    service_date: str
    denial_reason_code: Optional[str] = None   # only if reprocessing a denied claim


class ClaimResponse(BaseModel):
    claim_id: str
    recommendation: str          # APPROVE / CORRECT / APPEAL / REJECT
    summary: str
    validation_errors: list[str]
    suggested_icd10_codes: list[str]
    suggested_cpt_codes: list[str]
    coding_confidence: str
    appeal_recommended: bool
    correction_steps: list[str]
    duration_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-claim", response_model=ClaimResponse)
def process_claim(req: ClaimRequest):
    """
    Run the full RCM agent pipeline on a single claim.
    The LangGraph supervisor routes through validator → coder → (denial analyzer if needed)
    and generates a final recommendation.
    """
    start = time.time()

    initial_state = {
        "claim_id": req.claim_id,
        "clinical_notes": req.clinical_notes,
        "patient_id": req.patient_id,
        "provider_npi": req.provider_npi,
        "service_date": req.service_date,
        "denial_reason_code": req.denial_reason_code,
        "validation_errors": [],
        "is_valid": None,
        "suggested_icd10_codes": [],
        "suggested_cpt_codes": [],
        "coding_confidence": "",
        "denial_explanation": "",
        "appeal_recommended": False,
        "correction_steps": [],
        "next_agent": "",
        "iteration": 0,
        "recommendation": "",
        "summary": ""
    }

    final_state = rcm_graph.invoke(initial_state)

    duration = round((time.time() - start) * 1000)
    logging.info(
        f"claim={req.claim_id} recommendation={final_state['recommendation']} "
        f"duration={duration}ms iterations={final_state['iteration']}"
    )

    return ClaimResponse(
        claim_id=final_state["claim_id"],
        recommendation=final_state["recommendation"],
        summary=final_state["summary"],
        validation_errors=final_state.get("validation_errors", []),
        suggested_icd10_codes=final_state.get("suggested_icd10_codes", []),
        suggested_cpt_codes=final_state.get("suggested_cpt_codes", []),
        coding_confidence=final_state.get("coding_confidence", ""),
        appeal_recommended=final_state.get("appeal_recommended", False),
        correction_steps=final_state.get("correction_steps", []),
        duration_ms=duration
    )


@app.post("/process-claim/stream")
def process_claim_stream(req: ClaimRequest):
    """
    Stream state updates as each agent node completes.
    Shows: which agent ran, what it found, before final answer.
    Useful for building a UI that shows agent reasoning step by step.
    """
    from fastapi.responses import StreamingResponse
    import json

    initial_state = {
        "claim_id": req.claim_id,
        "clinical_notes": req.clinical_notes,
        "patient_id": req.patient_id,
        "provider_npi": req.provider_npi,
        "service_date": req.service_date,
        "denial_reason_code": req.denial_reason_code,
        "validation_errors": [],
        "is_valid": None,
        "suggested_icd10_codes": [],
        "suggested_cpt_codes": [],
        "coding_confidence": "",
        "denial_explanation": "",
        "appeal_recommended": False,
        "correction_steps": [],
        "next_agent": "",
        "iteration": 0,
        "recommendation": "",
        "summary": ""
    }

    def generate():
        for step in rcm_graph.stream(initial_state):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            yield json.dumps({"node": node_name, "state": node_state}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
