# langgraph-multiagent

A LangGraph-based multi-agent system for healthcare Revenue Cycle Management (RCM). A supervisor orchestrates specialist agents through a typed state graph — validating claims, extracting billing codes, and analyzing denials.

## Architecture

```
START
  └── supervisor (decides next step based on state)
        ├── claims_validator  → checks required fields, NPI format, date format
        │     └── supervisor
        ├── billing_coder     → extracts ICD-10 + CPT codes from clinical notes
        │     └── supervisor
        ├── denial_analyzer   → explains denial, recommends appeal or correction
        │     └── supervisor
        └── generate_recommendation → APPROVE / CORRECT / APPEAL / REJECT
              └── END
```

## Key concept: shared state

Agents don't call each other. They all read from and write to a shared `ClaimState` TypedDict. The supervisor reads the state after each agent and decides what to do next — this is what makes LangGraph different from a simple chain.

```python
class ClaimState(TypedDict):
    claim_id: str
    clinical_notes: str        # raw doctor notes
    is_valid: bool             # set by claims_validator
    suggested_icd10_codes: list  # set by billing_coder
    denial_explanation: str    # set by denial_analyzer
    recommendation: str        # set by generate_recommendation
    next_agent: str            # set by supervisor — controls routing
```

## Routing logic (supervisor)

```
No validation yet?          → claims_validator
Invalid claim?              → END (generate REJECT recommendation)
Valid but no codes?         → billing_coder
Denial present, no analysis? → denial_analyzer
Everything done?            → END (generate final recommendation)
```

## Agents

| Agent | Input | Output |
|---|---|---|
| `claims_validator` | claim fields | `validation_errors`, `is_valid` |
| `billing_coder` | `clinical_notes` | `suggested_icd10_codes`, `suggested_cpt_codes`, `coding_confidence` |
| `denial_analyzer` | `denial_reason_code` + codes | `denial_explanation`, `appeal_recommended`, `correction_steps` |
| `generate_recommendation` | all state | `recommendation`, `summary` |

## Run locally

**Prerequisites:** [Ollama](https://ollama.com) running with `llama3.2` pulled.

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

**Process a new claim:**
```bash
curl -X POST http://localhost:8000/process-claim \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-001",
    "clinical_notes": "Patient presents with type 2 diabetes and hypertension. Routine office visit, medication review, BP check.",
    "patient_id": "PAT-12345",
    "provider_npi": "1234567890",
    "service_date": "2026-05-23"
  }'
```

**Reprocess a denied claim:**
```bash
curl -X POST http://localhost:8000/process-claim \
  -d '{
    "claim_id": "CLM-001",
    "clinical_notes": "...",
    "patient_id": "PAT-12345",
    "provider_npi": "1234567890",
    "service_date": "2026-05-23",
    "denial_reason_code": "CO-4"
  }'
```

**Stream agent steps (see which agent is running in real time):**
```bash
curl -X POST http://localhost:8000/process-claim/stream \
  -H "Content-Type: application/json" \
  -d '{...}' 
# Returns NDJSON — one line per agent node as it completes
```

## Run with Docker

```bash
docker build -t langgraph-multiagent .
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  langgraph-multiagent
```

## LangGraph vs basic multi-agent (like multi-agent-devops)

| | multi-agent-devops | langgraph-multiagent |
|---|---|---|
| State | passed as strings | typed `ClaimState` TypedDict |
| Routing | single coordinator LLM call | supervisor node with explicit logic |
| Observability | print statements | stream() yields state after each node |
| Loop prevention | none | `iteration` counter in state |
| Type safety | none | TypedDict catches wrong field names |
| Extensibility | add if/else in coordinator | add node + edge in graph |

## Adding a new agent

```python
# 1. Add to agents.py
def prior_auth_checker(state: ClaimState) -> ClaimState:
    """Check if procedure requires prior authorization."""
    ...
    return {**state, "prior_auth_required": True}

# 2. Register in graph.py
graph.add_node("prior_auth_checker", prior_auth_checker)
graph.add_edge("prior_auth_checker", "supervisor")

# 3. Update supervisor routing logic
if not state.get("prior_auth_checked"):
    return {**state, "next_agent": "prior_auth_checker"}
```
