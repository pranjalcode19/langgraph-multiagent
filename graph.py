"""
LangGraph state machine for RCM claim processing.

Graph structure:
  START → supervisor → claims_validator → supervisor
                    → pii_masker       → supervisor   (masks PHI before LLM)
                    → billing_coder    → supervisor
                    → denial_analyzer  → supervisor
                    → END → generate_recommendation

Key LangGraph concepts:
- StateGraph: a directed graph where nodes are functions and edges are transitions
- add_conditional_edges: supervisor's return value determines next node
- compile(): locks the graph and returns a runnable
- invoke(): runs the full graph synchronously
- stream(): runs the graph and yields state after each node (useful for UI)
"""

from langgraph.graph import StateGraph, END, START
from state import ClaimState
from agents import supervisor, claims_validator, billing_coder, denial_analyzer, generate_recommendation
from pii_masker import pii_masker


def route_from_supervisor(state: ClaimState) -> str:
    """
    Edge function — reads state["next_agent"] set by supervisor,
    returns the name of the next node to execute.
    """
    return state["next_agent"]


def build_graph():
    graph = StateGraph(ClaimState)

    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("claims_validator", claims_validator)
    graph.add_node("pii_masker", pii_masker)
    graph.add_node("billing_coder", billing_coder)
    graph.add_node("denial_analyzer", denial_analyzer)
    graph.add_node("generate_recommendation", generate_recommendation)

    # Entry point: always start with supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor has conditional edges — it decides where to go
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "claims_validator": "claims_validator",
            "pii_masker":       "pii_masker",
            "billing_coder":    "billing_coder",
            "denial_analyzer":  "denial_analyzer",
            "END":              "generate_recommendation"
        }
    )

    # After each worker agent, return to supervisor for next decision
    graph.add_edge("claims_validator", "supervisor")
    graph.add_edge("pii_masker",       "supervisor")   # supervisor routes to billing_coder next
    graph.add_edge("billing_coder",    "supervisor")
    graph.add_edge("denial_analyzer",  "supervisor")

    # generate_recommendation is always the terminal node
    graph.add_edge("generate_recommendation", END)

    return graph.compile()


rcm_graph = build_graph()
