from __future__ import annotations
"""Cost-aware route advisor. Enumerates every model combination, prices it, scores quality,
keeps the cost-vs-quality Pareto frontier, and picks a route for the budget mode.
P2 can call recommend_route() directly as their router."""
from itertools import product
from pricing import cost_for, models_for_role, quality_prior
from calibration import model_stats
from tokens import count_tokens, layer_tokens

SIMPLE_NODES = ["triage", "executor", "qa"]
COMPLEX_NODES = ["triage", "planner", "executor", "qa"]
RETRY_NODES = {"planner", "executor", "qa"}          # re-run on a replan
WEIGHTS = {"triage": 0.1, "planner": 0.2, "executor": 0.5, "qa": 0.2}
MIN_Q = {"simple":  {"cost-first": 4.5, "balanced": 5.5, "quality-first": 6.5},
         "complex": {"cost-first": 6.0, "balanced": 7.0, "quality-first": 8.0}}
_COMPLEX_WORDS = ("analyze", "analyse", "compare", "calculate", "cross-reference", "evaluate",
                  "assess", "summarize", "explain why", "step by step", "risk", "compliance",
                  "design", "plan", "optimize", "architect")


def guess_complexity(query: str) -> str:
    q = query.lower().replace("-", " ")  # so 'step-by-step' matches 'step by step'
    hits = sum(w in q for w in _COMPLEX_WORDS)
    return "complex" if (len(q) > 300 or (hits >= 1 and len(q) > 120) or hits >= 2) else "simple"


def effective_quality(model: str) -> float:
    """Prior quality blended with observed QA win-rate (3 pseudo-observations of prior)."""
    s = model_stats(model)
    n = s["wins"] + s["losses"]
    return (quality_prior(model) * 3 + s["wins"] * 10) / (3 + n)


def route_quality(plan: dict) -> float:
    w = sum(WEIGHTS[n] for n in plan)
    return sum(effective_quality(m) * WEIGHTS[n] for n, m in plan.items()) / w


def route_cost(plan: dict, q_tokens: int) -> dict:
    layers, total, ti, to = [], 0.0, 0, 0
    for node, model in plan.items():
        p_in, p_out, base_in = layer_tokens(node, q_tokens)
        c = cost_for(model, p_in, p_out)
        layers.append({"node": node, "model": model, "est_prompt_tokens": p_in,
                       "base_prompt_tokens": base_in, "est_completion_tokens": p_out,
                       "est_cost": round(c, 6)})
        total += c; ti += p_in; to += p_out
    retry = sum(l["est_cost"] for l in layers if l["node"] in RETRY_NODES)
    return {"layers": layers, "cost": total, "worst_cost": total + retry,
            "prompt_tokens": ti, "completion_tokens": to}


def all_routes(complexity: str, q_tokens: int) -> list:
    nodes = COMPLEX_NODES if complexity == "complex" else SIMPLE_NODES
    pools = [models_for_role(n) for n in nodes]
    out = []
    for combo in product(*pools):
        plan = dict(zip(nodes, combo))
        rc = route_cost(plan, q_tokens)
        out.append({"plan": plan, "quality": route_quality(plan), **rc})
    return out


def pareto(routes: list) -> list:
    """Non-dominated routes: no other route is both cheaper and better."""
    front, best_q = [], -1.0
    for r in sorted(routes, key=lambda r: (r["cost"], -r["quality"])):
        if r["quality"] > best_q + 1e-9:
            front.append(r); best_q = r["quality"]
    return front


def choose(routes: list, complexity: str, mode: str, max_cost: float | None = None) -> dict:
    pool = routes
    if max_cost is not None:
        within = [r for r in routes if r["worst_cost"] <= max_cost]
        if not within:
            return min(routes, key=lambda r: r["worst_cost"])   # cheapest possible
        pool = within
    if mode == "quality-first":
        return max(pool, key=lambda r: (r["quality"], -r["cost"]))
    ok = [r for r in pool if r["quality"] >= MIN_Q[complexity][mode]]
    if ok:
        return min(ok, key=lambda r: r["cost"])
    return max(pool, key=lambda r: r["quality"])


def recommend_route(query: str, mode: str = "balanced", max_cost: float | None = None) -> dict:
    """Public API for P2: returns {'route': {node: model}, 'cost', 'worst_cost', 'quality', ...}."""
    if mode not in MIN_Q["simple"]:
        mode = "balanced"
    q = count_tokens(query)
    cx = guess_complexity(query)
    routes = all_routes(cx, q)
    best = choose(routes, cx, mode, max_cost)
    front = [{"cost": round(r["cost"], 6), "quality": round(r["quality"], 2), "route": r["plan"]}
             for r in pareto(routes)]
    return {"route": best["plan"], "cost": best["cost"], "worst_cost": best["worst_cost"],
            "quality": round(best["quality"], 2), "complexity": cx, "query_tokens": q,
            "frontier": front, "n_candidates": len(routes)}