"""
Multi-agent cost-optimization pipeline (LangGraph orchestration).

Team split:
  * THIS file owns the graph structure, state, routing, and wiring to the
    model-selection Router.
  * router.py / model_catalog.py own the provider-neutral model selection
    (semantic-router intent + RouteLLM-style offline difficulty + cost rank
    + a QA-fed learning loop).
  * LLM generation is MOCKED (colleagues own the real calls against the
    chosen model_id).
  * Node A (Pre-Flight Cost) and Node F (Post-Run Ledger) are empty
    pass-through placeholders (colleagues own the cost math).

Install (one-time):
    pip install langgraph fastembed "semantic-router==0.0.72"

Run:
    python cost_optimization_graph.py
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from router import ROUTER

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pipeline")


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Reducer: merge partial dict updates from multiple nodes."""
    return {**(left or {}), **(right or {})}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    requires_planning: bool
    plan: str
    qa_passed: bool

    # Token / cost tracking placeholders (owned by Node A / Node F).
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    actual_input_tokens: int
    actual_output_tokens: int
    actual_cost: float

    # Audit trail: which model each layer used (merge, don't overwrite).
    model_per_layer: Annotated[dict[str, str], merge_dict]

    # Guards the Node E -> Node C replanning cycle.
    replan_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _user_query(state: PipelineState) -> str:
    """Return the original USER request to route on.

    Must skip assistant/tool messages — otherwise later layers (planner, QA)
    would classify the executor's output text instead of the task, producing
    nonsense routing signals.
    """
    def is_user(m) -> bool:
        # LangChain message objects expose .type ("human"); raw dicts use role.
        t = getattr(m, "type", None)
        if t is not None:
            return t == "human"
        return isinstance(m, dict) and m.get("role") == "user"

    def content(m) -> str:
        return getattr(m, "content", None) or (
            m.get("content", "") if isinstance(m, dict) else str(m)
        )

    for m in reversed(state.get("messages", [])):
        if is_user(m):
            return content(m)
    return ""


def mock_llm_call(model_id: str, prompt: str) -> str:
    """Stand-in for the real generation (colleagues own this).
    Deterministic dummy so the graph runs with no API key / network."""
    return f"[mock output from {model_id}] -> {prompt[:60]}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_a_preflight(state: PipelineState) -> PipelineState:
    """Node A — Pre-Flight Cost Estimator. PLACEHOLDER (pass-through)."""
    log.info("[Node A] Pre-Flight Estimator (placeholder — pass-through)")
    return state


def node_b_triage(state: PipelineState) -> PipelineState:
    """Node B — Triage. Classify whether the query needs planning."""
    log.info("[Node B] Triage")
    query = _latest_user_text(state)
    model = ROUTER.choose("triage", query, state)

    _ = mock_llm_call(model, "Classify: does this query require planning?")
    requires_planning = len(query) > 20  # dummy heuristic for the demo
    log.info("          -> requires_planning = %s", requires_planning)
    return {"requires_planning": requires_planning,
            "model_per_layer": {"triage": model}}


def node_c_planner(state: PipelineState) -> PipelineState:
    """Node C — Planner. Produce a step-by-step plan."""
    log.info("[Node C] Planner")
    query = _latest_user_text(state)
    model = ROUTER.choose("planner", query, state)

    plan = mock_llm_call(model, "Generate a step-by-step plan for the task.")
    log.info("          -> plan drafted")
    return {"plan": f"1. understand  2. act  3. verify  ({plan})",
            "model_per_layer": {"planner": model}}


def node_d_execution(state: PipelineState) -> PipelineState:
    """Node D — Execution. Carry out the task (or plan)."""
    log.info("[Node D] Execution")
    query = _latest_user_text(state)
    model = ROUTER.choose("execution", query, state)

    plan = state.get("plan", "(no plan — direct execution)")
    result = mock_llm_call(model, f"Execute against plan: {plan}")
    log.info("          -> task executed")
    return {"messages": [{"role": "assistant", "content": result}],
            "model_per_layer": {"execution": model}}


def node_e_qa(state: PipelineState) -> PipelineState:
    """Node E — QA & Auditor. Did execution succeed?"""
    log.info("[Node E] QA & Auditor")
    query = _latest_user_text(state)
    model = ROUTER.choose("qa", query, state)

    _ = mock_llm_call(model, "Audit the execution output. Pass or fail?")

    # Dummy audit: fail once to exercise the replan cycle, then pass.
    replan_count = state.get("replan_count", 0)
    qa_passed = replan_count >= 1
    log.info("          -> qa_passed = %s (replan_count=%d)", qa_passed, replan_count)

    # Feed the verdict back into the Router's learning loop. In production this
    # closes the self-improving feedback loop (the executor's model is what we
    # judged; here we credit the execution-layer choice).
    exec_model = state.get("model_per_layer", {}).get("execution", model)
    ROUTER.record_outcome(exec_model, qa_passed)

    update: PipelineState = {"qa_passed": qa_passed, "model_per_layer": {"qa": model}}
    if not qa_passed:
        update["replan_count"] = replan_count + 1
    return update


def node_f_ledger(state: PipelineState) -> PipelineState:
    """Node F — Post-Run Ledger. PLACEHOLDER (pass-through)."""
    log.info("[Node F] Post-Run Ledger (placeholder — pass-through)")
    return state


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def route_after_triage(state: PipelineState) -> Literal["planner", "execution"]:
    return "planner" if state.get("requires_planning") else "execution"


def route_after_qa(state: PipelineState) -> Literal["ledger", "replan"]:
    # Pure read — the QA node already committed the incremented replan_count.
    return "ledger" if state.get("qa_passed") else "replan"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("preflight", node_a_preflight)   # A
    graph.add_node("triage", node_b_triage)          # B
    graph.add_node("planner", node_c_planner)        # C
    graph.add_node("execution", node_d_execution)    # D
    graph.add_node("qa", node_e_qa)                  # E
    graph.add_node("ledger", node_f_ledger)          # F

    graph.add_edge(START, "preflight")
    graph.add_edge("preflight", "triage")
    graph.add_conditional_edges(
        "triage", route_after_triage,
        {"planner": "planner", "execution": "execution"},
    )
    graph.add_edge("planner", "execution")
    graph.add_edge("execution", "qa")
    graph.add_conditional_edges(
        "qa", route_after_qa,
        {"ledger": "ledger", "replan": "planner"},
    )
    graph.add_edge("ledger", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()
    log.info("Router backend: %s\n", ROUTER.backend)

    # A few varied requests to show per-request, provider-neutral routing
    # (NOT a fixed model-per-layer map).
    demos = [
        "hi",
        "Write and debug a Python function to parse CSV files.",
        "Prove that there are infinitely many primes, rigorously, with full detail.",
    ]
    for q in demos:
        log.info("=" * 70)
        log.info("USER: %s", q)
        final = app.invoke({"messages": [{"role": "user", "content": q}]})
        log.info("-> models used: %s", final.get("model_per_layer"))
        log.info("-> qa_passed=%s  replans=%s",
                 final.get("qa_passed"), final.get("replan_count", 0))


