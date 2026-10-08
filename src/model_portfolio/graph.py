from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .litellm_client import GraphState, call_model, estimate_messages_tokens


Route = Literal["execution", "planner"]
QaRoute = Literal["post_run_ledger", "planner"]


def preflight_estimator(state: GraphState) -> GraphState:
    prompt = state.get("messages", [{}])[-1].get("content", "")
    estimated_tokens = estimate_messages_tokens([{"role": "user", "content": prompt}], "gpt-4o")
    return {
        **state,
        "estimated_cost": estimated_tokens * 0.000005,
    }


def triage(state: GraphState) -> GraphState:
    prompt = state["messages"][-1]["content"]
    response, updated = call_model(
        prompt=prompt,
        model_tier="triage",
        state=state,
        system_prompt=(
            "Classify the user request as Simple or Complex. "
            'Return only JSON: {"complexity":"Simple"} or {"complexity":"Complex"}.'
        ),
    )
    parsed = _parse_json(response, default={"complexity": "Complex"})
    updated["complexity"] = parsed.get("complexity", "Complex")
    return updated


def route_after_triage(state: GraphState) -> Route:
    return "execution" if state.get("complexity") == "Simple" else "planner"


def planner(state: GraphState) -> GraphState:
    prompt = state["messages"][-1]["content"]
    response, updated = call_model(
        prompt=prompt,
        model_tier="planner",
        state=state,
        system_prompt=(
            "Create a concise step-by-step plan. "
            'Return only JSON: {"steps":["step 1","step 2"]}.'
        ),
    )
    parsed = _parse_json(response, default={"steps": [prompt]})
    updated["current_plan"] = parsed.get("steps", [prompt])
    return updated


def execution(state: GraphState) -> GraphState:
    prompt = state["messages"][-1]["content"]
    plan = state.get("current_plan") or [prompt]
    response, updated = call_model(
        prompt=f"User request:\n{prompt}\n\nPlan:\n{json.dumps(plan)}",
        model_tier="execution",
        state=state,
        system_prompt="Execute the plan and produce the best final answer.",
    )
    updated["draft_answer"] = response
    return updated


def qa_auditor(state: GraphState) -> GraphState:
    prompt = state["messages"][-1]["content"]
    draft = state.get("draft_answer", "")
    response, updated = call_model(
        prompt=f"User request:\n{prompt}\n\nDraft answer:\n{draft}",
        model_tier="qa",
        state=state,
        system_prompt=(
            "Judge whether the draft fully solves the request. "
            'Return only JSON: {"verdict":"PASS","reason":"..."} or '
            '{"verdict":"FAIL","reason":"..."}'
        ),
    )
    parsed = _parse_json(response, default={"verdict": "FAIL", "reason": "Could not parse QA."})
    updated["qa_verdict"] = parsed.get("verdict", "FAIL")
    updated["qa_reason"] = parsed.get("reason", "")
    updated["replan_count"] = int(updated.get("replan_count", 0))
    if updated["qa_verdict"] != "PASS":
        updated["replan_count"] += 1
    return updated


def route_after_qa(state: GraphState) -> QaRoute:
    if state.get("qa_verdict") == "PASS" or int(state.get("replan_count", 0)) >= 2:
        return "post_run_ledger"
    return "planner"


def post_run_ledger(state: GraphState) -> GraphState:
    totals = state.get("token_totals", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    market_rate_per_token = 0.000005
    actual_cost = totals["total_tokens"] * market_rate_per_token
    return {
        **state,
        "actual_cost": actual_cost,
        "final_answer": state.get("draft_answer", ""),
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("preflight_estimator", preflight_estimator)
    graph.add_node("triage", triage)
    graph.add_node("planner", planner)
    graph.add_node("execution", execution)
    graph.add_node("qa_auditor", qa_auditor)
    graph.add_node("post_run_ledger", post_run_ledger)

    graph.add_edge(START, "preflight_estimator")
    graph.add_edge("preflight_estimator", "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"execution": "execution", "planner": "planner"},
    )
    graph.add_edge("planner", "execution")
    graph.add_edge("execution", "qa_auditor")
    graph.add_conditional_edges(
        "qa_auditor",
        route_after_qa,
        {"post_run_ledger": "post_run_ledger", "planner": "planner"},
    )
    graph.add_edge("post_run_ledger", END)
    return graph.compile()


def run_portfolio_graph(user_prompt: str) -> GraphState:
    app = build_graph()
    return app.invoke({"messages": [{"role": "user", "content": user_prompt}]})


def _parse_json(raw: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return default
