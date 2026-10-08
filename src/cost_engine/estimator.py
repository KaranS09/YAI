"""Node A: Pre-Flight Estimator + policy gate."""
import uuid
from pricing import cost_for, BASELINE_MODEL, BASELINES, normalize_route, to_p2_names
from calibration import total_runs, mape
from tokens import count_tokens
from router_advisor import recommend_route, route_cost, guess_complexity
from governance import POLICIES, check_policy, spent_today


def _p2_route(query: str, mode: str):
    try:
        from router import expected_route   # <- P2's function: expected_route(query, mode) -> {node: model}
        return normalize_route(expected_route(query, mode))
    except Exception:
        return None


def estimate(query: str, mode: str = "balanced", max_cost: float | None = None) -> dict:
    if mode not in POLICIES:
        mode = "balanced"
    q = count_tokens(query)
    cx = guess_complexity(query)
    rec = recommend_route(query, mode)                      # also gives the Pareto frontier

    plan, source = _p2_route(query, mode), "p2_router"
    if not plan:
        plan, source = rec["route"], "cost_engine_advisor"
    rc = route_cost(plan, q)

    # ---- policy gate: APPROVED / DOWNGRADED / BLOCKED ----
    cap = max_cost if max_cost is not None else POLICIES[mode]["max_cost_per_query"]
    pol = check_policy(rc["worst_cost"], mode, cap, spent_today())
    decision, note = ("APPROVED", pol["reason"]) if pol["ok"] else (None, pol["reason"])
    if not pol["ok"]:
        if pol["kind"] == "per_query":
            cheaper = recommend_route(query, mode, max_cost=cap)
            rc2 = route_cost(cheaper["route"], q)
            if rc2["worst_cost"] <= cap:
                plan, rc, source, decision = cheaper["route"], rc2, "cost_engine_downgrade", "DOWNGRADED"
                note += f" -> downgraded to a cheaper route (${rc2['worst_cost']:.5f} worst case)"
            else:
                decision = "BLOCKED"; note += " -> no route fits the cap"
        else:
            decision = "BLOCKED"

    baselines = {m: round(cost_for(m, rc["prompt_tokens"], rc["completion_tokens"]), 6) for m in BASELINES}
    base = baselines[BASELINE_MODEL]
    runs = total_runs()
    return {
        "run_id": uuid.uuid4().hex[:12],
        "mode": mode, "complexity": cx, "query_tokens": q,
        "route_source": source, "route_plan": plan, "route_plan_p2": to_p2_names(plan), "layers": rc["layers"],
        "estimated_cost": round(rc["cost"], 6),
        "estimated_cost_worst_case": round(rc["worst_cost"], 6),
        "baseline_model": BASELINE_MODEL, "baseline_estimate": base, "baselines": baselines,
        "estimated_savings_pct": round((base - rc["cost"]) / base * 100, 1) if base else 0.0,
        "decision": decision, "decision_note": note, "policy": pol,
        "confidence": "low" if runs < 3 else "medium" if runs < 10 else "high",
        "calibration_runs": runs, "recent_error_pct": mape(),
        "frontier": rec["frontier"],
    }


def pre_flight_node(state: dict) -> dict:
    """LangGraph Node A. Optional input: state['max_cost'] (USD cap for this query).
    Add 'estimate_detail' to the state class."""
    d = estimate(state["query"], state.get("budget_mode", "balanced"), state.get("max_cost"))
    return {"estimated_cost": d["estimated_cost"], "estimate_detail": d}


def route_after_preflight(state: dict) -> str:
    """Use as a LangGraph conditional edge: 'blocked' -> END, 'continue' -> triage."""
    return "blocked" if state.get("estimate_detail", {}).get("decision") == "BLOCKED" else "continue"