import time
import logging
import uuid as uuid_lib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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
        "pii_mapping": {},
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
        "pii_mapping": {},
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


# ── OpenAI-compatible endpoint ────────────────────────────────────────────────
# Accepts clinical notes as the message content.
# Format: "CLAIM_ID: C001 | PATIENT: P123 | NPI: 1234567890 | DATE: 2024-01-15 | NOTES: <clinical notes>"
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "rcm-claims-agent"
    messages: List[ChatMessage]

@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    content = body.messages[-1].content
    start = time.time()

    # Parse structured input or use defaults for demo
    import re
    claim_id   = re.search(r"CLAIM_ID:\s*(\S+)", content)
    patient_id = re.search(r"PATIENT:\s*(\S+)", content)
    npi        = re.search(r"NPI:\s*(\S+)", content)
    date       = re.search(r"DATE:\s*(\S+)", content)

    initial_state = {
        "claim_id":        claim_id.group(1) if claim_id else f"C-{uuid_lib.uuid4().hex[:6]}",
        "clinical_notes":  content,
        "patient_id":      patient_id.group(1) if patient_id else "P-DEMO",
        "provider_npi":    npi.group(1) if npi else "1234567890",
        "service_date":    date.group(1) if date else "2024-01-15",
        "denial_reason_code": None,
        "validation_errors": [], "is_valid": None,
        "suggested_icd10_codes": [], "suggested_cpt_codes": [], "coding_confidence": "",
        "denial_explanation": "", "appeal_recommended": False, "correction_steps": [],
        "next_agent": "", "iteration": 0, "recommendation": "", "summary": ""
    }

    final_state = rcm_graph.invoke(initial_state)
    answer = f"**Recommendation: {final_state['recommendation']}**\n\n{final_state['summary']}"
    if final_state.get("suggested_icd10_codes"):
        answer += f"\n\nICD-10: {', '.join(final_state['suggested_icd10_codes'])}"
        answer += f"\nCPT: {', '.join(final_state['suggested_cpt_codes'])}"
        answer += f"\nCoding Confidence: {final_state['coding_confidence']}"

    duration = round((time.time() - start) * 1000)
    logging.info(f"openai-compat claim={initial_state['claim_id']} recommendation={final_state['recommendation']} duration={duration}ms")
    return {
        "id": f"chatcmpl-{uuid_lib.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": "rcm-claims-agent",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }]
    }

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "rcm-claims-agent", "object": "model"}]}
