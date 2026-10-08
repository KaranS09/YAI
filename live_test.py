"""
Live end-to-end test harness — detailed per-layer reporting.

Runs several varied prompts through the integrated graph with REAL API calls
(Gemini / Groq / OpenRouter) and prints, for each prompt:
  * difficulty score + detected intent (decision layer)
  * pre-flight estimate vs post-run actual
  * every layer's model / provider / tokens / cost / LIVE-or-MOCK / attempt
  * QA verdict + replan count
  * savings vs the monolith baseline
"""

from __future__ import annotations

import logging

# Importing portfolio_manager wires sys.path, loads keys, builds everything.
from portfolio_manager import build_graph
from router import ROUTER
from pricing import cost_for, BASELINE_MODEL  # cost engine (on path via import above)

# Show the decision layer's per-pick reasoning (intent + difficulty).
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("router").setLevel(logging.INFO)

PROMPTS = [
    "Convert 100 US dollars to euros at an exchange rate of 0.92.",
    "Write a SQL query to find the top 5 customers by total purchase amount.",
    "Compare REST and GraphQL for a mobile-app backend and recommend one, "
    "evaluating the trade-offs with clear reasoning.",
]


def run_one(app, query: str) -> dict:
    diff = ROUTER._difficulty(query)
    intent, _ = ROUTER._intent(query)
    print("\n" + "=" * 78)
    print(f"USER QUERY: {query}")
    print(f"decision-layer read: difficulty={diff:.2f}  intent={intent}")
    print("=" * 78)

    final = app.invoke({
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "replan_count": 0,
        "max_retries": 2,
        "budget_mode": "balanced",
    })

    detail = final.get("estimate_detail", {})
    ledger = final.get("token_ledger", [])
    est = final.get("estimated_cost", 0.0)
    actual = final.get("actual_cost", 0.0)
    verdict = final.get("qa_verdict")
    replans = int(final.get("replan_count", 0))

    # Per-layer table
    print(f"\n  complexity={detail.get('complexity')}  route_source={detail.get('route_source')}"
          f"  policy={detail.get('decision')}")
    print(f"  {'#':<3}{'layer':<10}{'provider':<10}{'model':<46}{'in/out':>11}{'cost$':>10}  mode")
    seen: dict[str, int] = {}
    total_in = total_out = 0
    for i, e in enumerate(ledger, 1):
        tier = e["tier"]; seen[tier] = seen.get(tier, 0) + 1
        c = cost_for(e["model"], e["prompt_tokens"], e["completion_tokens"])
        total_in += e["prompt_tokens"]; total_out += e["completion_tokens"]
        mode = "MOCK" if e.get("mock") else "LIVE"
        toks = f"{e['prompt_tokens']}/{e['completion_tokens']}"
        print(f"  {i:<3}{tier:<10}{e['provider']:<10}{e['model'][:44]:<46}{toks:>11}{c:>10.5f}  {mode}"
              + (f"  (try {seen[tier]})" if seen[tier] > 1 else ""))

    base_total = cost_for(BASELINE_MODEL, total_in, total_out)
    saved = base_total - actual
    est_err = (actual - est) / est * 100 if est else 0.0
    print(f"\n  tokens total: in={total_in} out={total_out}   QA={verdict}  replans={replans}")
    print("  " + "-" * 54)
    print(f"  {'PREDICTED cost (pre-flight)':<34} ${est:.5f}")
    print(f"  {'ACTUAL cost (post-run)':<34} ${actual:.5f}   ({est_err:+.0f}% vs predicted)")
    print(f"  {'MONOLITH baseline (same tokens)':<34} ${base_total:.5f}  ({BASELINE_MODEL})")
    if base_total:
        print(f"  {'>>> SAVINGS vs monolith':<34} ${saved:.5f}  ({saved / base_total * 100:.1f}% cheaper)")
    print("  " + "-" * 54)

    return {"query": query, "est": est, "actual": actual, "base": base_total,
            "verdict": verdict, "replans": replans,
            "models": [e["model"] for e in ledger],
            "live": sum(0 if e.get("mock") else 1 for e in ledger),
            "mock": sum(1 if e.get("mock") else 0 for e in ledger)}


if __name__ == "__main__":
    app = build_graph()
    print(f"Decision-layer backend: {ROUTER.backend}")
    results = [run_one(app, q) for q in PROMPTS]

    # Cross-prompt roll-up
    print("\n" + "#" * 78)
    print("# OVERALL TEST SUMMARY")
    print("#" * 78)
    tot_est = sum(r["est"] for r in results)
    tot_actual = sum(r["actual"] for r in results)
    tot_base = sum(r["base"] for r in results)
    tot_live = sum(r["live"] for r in results)
    tot_mock = sum(r["mock"] for r in results)
    print(f"  {'prompt':<46}{'predicted':>11}{'actual':>10}{'saved':>8}  QA")
    for r in results:
        pct = (r["base"] - r["actual"]) / r["base"] * 100 if r["base"] else 0
        print(f"  {r['query'][:44]:<46}${r['est']:>9.5f}${r['actual']:>9.5f}{pct:>7.0f}%  {r['verdict']}")
    print(f"\n  LLM calls: {tot_live} LIVE / {tot_mock} mock")
    print(f"  TOTAL predicted: ${tot_est:.5f}   TOTAL actual: ${tot_actual:.5f}   "
          f"TOTAL monolith: ${tot_base:.5f}")
    if tot_base:
        print(f"  TOTAL SAVINGS: ${tot_base - tot_actual:.5f}  "
              f"({(tot_base - tot_actual) / tot_base * 100:.1f}% cheaper than monolith)")
    print("#" * 78)
