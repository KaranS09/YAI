"""Node F: Post-Run Ledger. True cost of completing the task, waste, savings, audit trail."""
from collections import Counter, defaultdict
from pricing import cost_for, BASELINE_MODEL, BASELINES, normalize_ledger
from calibration import update_history, mape
from governance import append_audit
from router_advisor import RETRY_NODES


def ledger_entry(node: str, model: str, usage) -> dict:
    """Helper for P2's wrapper: build an entry from LiteLLM response.usage."""
    return {"node": node, "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}


def running_cost(ledger: list) -> float:
    return sum(cost_for(e["model"], e["prompt_tokens"], e["completion_tokens"]) for e in ledger)


def check_budget(state: dict, limit_usd: float) -> bool:
    """True if still within budget. Use in a conditional edge to stop replan loops."""
    return running_cost(state.get("token_ledger", [])) <= limit_usd


def audit(ledger: list, estimated_cost: float = 0.0, estimate_detail: dict | None = None,
          passed: bool = True) -> dict:
    ledger = normalize_ledger(ledger)
    last = {}
    for i, e in enumerate(ledger):
        if e["node"] in RETRY_NODES:
            last[e["node"]] = i
    seen, rows = Counter(), []
    by_node, by_model = defaultdict(float), defaultdict(float)
    total = waste = 0.0
    tin = tout = wasted_calls = 0
    for i, e in enumerate(ledger):
        n = e["node"]; seen[n] += 1
        pt, ct = e.get("prompt_tokens", 0), e.get("completion_tokens", 0)
        c = cost_for(e["model"], pt, ct)
        is_waste = n in RETRY_NODES and (not passed or i != last[n])
        rows.append({"node": n, "model": e["model"], "attempt": seen[n], "prompt_tokens": pt,
                     "completion_tokens": ct, "cost": round(c, 6), "is_waste": is_waste})
        total += c; tin += pt; tout += ct
        by_node[n] += c; by_model[e["model"]] += c
        if is_waste:
            waste += c; wasted_calls += 1

    baselines = {m: round(cost_for(m, tin, tout), 6) for m in BASELINES}
    base = baselines[BASELINE_MODEL]
    savings = (base - total) / base * 100 if base else 0.0
    diff = total - estimated_cost
    tok = tin + tout
    return {
        "actual_cost": round(total, 6), "baseline_cost": base, "savings_pct": round(savings, 1),
        "cost_breakdown": rows,
        "cost_report": {
            "run_id": (estimate_detail or {}).get("run_id"),
            "passed": passed,
            "attempts": max(1, sum(1 for e in ledger if e["node"] == "executor")),
            "tokens": {"prompt": tin, "completion": tout, "total": tok},
            "baseline_model": BASELINE_MODEL, "baselines": baselines,
            "savings_pct_by_baseline": {m: round((b - total) / b * 100, 1) if b else 0.0
                                        for m, b in baselines.items()},
            "waste": {"cost": round(waste, 6), "pct_of_total": round(waste / total * 100, 1) if total else 0.0,
                      "wasted_calls": wasted_calls},
            "cost_per_successful_task": round(total, 6) if passed else None,
            "cost_of_failed_task": None if passed else round(total, 6),
            "by_node": {k: round(v, 6) for k, v in by_node.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "blended_usd_per_1m_tokens": {"actual": round(total / tok * 1e6, 3) if tok else 0.0,
                                          "baseline": round(base / tok * 1e6, 3) if tok else 0.0},
            "variance": {"estimated": estimated_cost, "actual": round(total, 6), "diff": round(diff, 6),
                         "diff_pct": round(diff / estimated_cost * 100, 1) if estimated_cost else None},
            "decision": (estimate_detail or {}).get("decision"),
            "within_cap": total <= (estimate_detail or {}).get("policy", {}).get("per_query_cap", float("inf")),
            "estimator_recent_error_pct": mape(),
        },
    }


def post_run_node(state: dict) -> dict:
    """LangGraph Node F. Add 'cost_report' to the state class.
    Optional input: state['qa_passed'] (bool, default True)."""
    ledger = normalize_ledger(state.get("token_ledger", []))
    est = state.get("estimated_cost", 0.0)
    detail = state.get("estimate_detail")
    passed = state.get("qa_passed", True)
    res = audit(ledger, est, detail, passed)
    try:   # persistence must never crash the graph
        d = detail or {}
        append_audit({"run_id": d.get("run_id"), "mode": d.get("mode"), "decision": d.get("decision"),
                      "route_plan": d.get("route_plan"), "estimated_cost": est,
                      "actual_cost": res["actual_cost"], "baseline_cost": res["baseline_cost"],
                      "savings_pct": res["savings_pct"], "passed": passed,
                      "waste_cost": res["cost_report"]["waste"]["cost"], "ledger": ledger})
        update_history(ledger, detail, est, res["actual_cost"], passed)
    except Exception as ex:
        print(f"[cost_engine] persistence skipped: {ex}")
    return res