from __future__ import annotations
"""Run: python test_cost.py  (no API keys, no teammates needed). Uses a temp data dir."""
import os, tempfile, json
os.environ["COST_ENGINE_DATA_DIR"] = tempfile.mkdtemp()   # keep demo data out of the real history

from pricing import cost_for, get_price
from router_advisor import recommend_route
from estimator import pre_flight_node, estimate, route_after_preflight
from ledger import post_run_node, check_budget
from governance import verify_audit
from calibration import _load
from report import scorecard

SIMPLE = "What is our refund policy?"
COMPLEX = ("Analyze our 50-page compliance policy and cross-reference the refund rules "
           "to calculate Q3 liabilities, then assess the regulatory risk step by step.")

# 1. basics
assert abs(cost_for("gpt-4o", 1_000_000, 1_000_000) - 12.50) < 1e-9
assert get_price("some/unknown-model")["input_per_1m"] > 0
assert get_price("apodex/apodex-1.1-mini")["input_per_1m"] == get_price("openrouter/apodex/apodex-1.1-mini:free")["input_per_1m"]

# 2. route advisor: cost-aware, modes differ, Pareto frontier is monotonic
rs, rc = recommend_route(SIMPLE, "balanced"), recommend_route(COMPLEX, "balanced")
assert rs["complexity"] == "simple" and rc["complexity"] == "complex" and "planner" in rc["route"]
cheap, best = recommend_route(COMPLEX, "cost-first"), recommend_route(COMPLEX, "quality-first")
assert cheap["cost"] <= rc["cost"] <= best["cost"] and cheap["quality"] <= best["quality"]
f = rc["frontier"]
assert all(f[i]["cost"] <= f[i+1]["cost"] and f[i]["quality"] < f[i+1]["quality"] for i in range(len(f)-1))
print(f"advisor: {rc['n_candidates']} candidate routes, {len(f)} on Pareto frontier")

# 3. policy gate
d = estimate(COMPLEX, "balanced");               assert d["decision"] == "APPROVED"
cap = recommend_route(COMPLEX, "cost-first")["worst_cost"] * 1.01
d = estimate(COMPLEX, "quality-first", max_cost=cap)
assert d["decision"] == "DOWNGRADED" and d["estimated_cost_worst_case"] <= cap
d = estimate(COMPLEX, "balanced", max_cost=1e-9); assert d["decision"] == "BLOCKED"
assert route_after_preflight({"estimate_detail": d}) == "blocked"
print("policy: APPROVED / DOWNGRADED / BLOCKED all work")

# 4. full run with a replan (executor + qa + planner twice) -> waste attribution
state = {"query": COMPLEX, "budget_mode": "balanced"}
state.update(pre_flight_node(state))
plan = state["estimate_detail"]["route_plan"]
mk = lambda n, pi, co: {"node": n, "model": plan[n], "prompt_tokens": pi, "completion_tokens": co}
state["token_ledger"] = [mk("triage", 180, 25), mk("planner", 400, 350), mk("executor", 600, 500),
                         mk("qa", 900, 30), mk("planner", 450, 300), mk("executor", 650, 520), mk("qa", 950, 30)]
res = post_run_node(state)
r = res["cost_report"]
assert r["attempts"] == 2 and r["waste"]["wasted_calls"] == 3 and r["waste"]["cost"] > 0
assert res["actual_cost"] < res["baseline_cost"] and r["cost_per_successful_task"] == res["actual_cost"]
print("\n" + scorecard(state["estimate_detail"], res))

# 5. failed task: all retry-node spend is waste, no 'successful task' cost
state["qa_passed"] = False
r2 = post_run_node(state)["cost_report"]
assert r2["cost_per_successful_task"] is None and r2["cost_of_failed_task"] > 0

# 6. self-calibration: estimate error should shrink as the engine sees real runs
def one_run():
    s = {"query": COMPLEX, "budget_mode": "balanced"}; s.update(pre_flight_node(s))
    p = s["estimate_detail"]["layers"]
    s["token_ledger"] = [{"node": l["node"], "model": l["model"],
                          "prompt_tokens": int(l["base_prompt_tokens"] * 1.4),
                          "completion_tokens": int(l["est_completion_tokens"] * 0.6) or 5} for l in p]
    post_run_node(s)
for _ in range(8): one_run()
acc = _load()["accuracy"]
errs = [a["err_pct"] for a in acc]
print(f"\ncalibration: estimate error per run -> {errs}")
assert errs[-1] < errs[0] * 0.3, "estimator should improve with history"

# 7. tamper-evident audit log
ok, n, bad = verify_audit(); assert ok and n >= 10
p = os.path.join(os.environ["COST_ENGINE_DATA_DIR"], "audit_log.jsonl")
lines = open(p).read().splitlines()
rec = json.loads(lines[3]); rec["actual_cost"] = 0.0           # someone edits a past record
lines[3] = json.dumps(rec); open(p, "w").write("\n".join(lines) + "\n")
ok, n, bad = verify_audit(); assert not ok and bad == 3
print(f"audit log: {n} records, chain verified, tampering detected at record #{bad}")

# 8. P2's real wrapper format: 'tier' instead of 'node', 'execution' instead of 'executor'
p2 = [{"tier": "triage", "model": "groq/openai/gpt-oss-20b", "provider": "groq", "prompt_tokens": 120,
       "completion_tokens": 30, "total_tokens": 150, "estimated_prompt_tokens": 118},
      {"tier": "execution", "model": "groq/qwen/qwen3.8-27b", "provider": "groq", "prompt_tokens": 600,
       "completion_tokens": 500, "total_tokens": 1100, "estimated_prompt_tokens": 590},
      {"tier": "qa", "model": "openrouter/nvidia/nemotron-3.5-lightning:free", "provider": "openrouter",
       "prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930, "estimated_prompt_tokens": 880}]
r3 = post_run_node({"token_ledger": p2, "estimated_cost": 0.0005})
assert [row["node"] for row in r3["cost_breakdown"]] == ["triage", "executor", "qa"]
assert r3["actual_cost"] > 0 and "[cost_engine] WARNING" not in ""   # all 3 models are priced
from pricing import resolve
assert all(resolve(e["model"]) for e in p2), "a P2 model string has no price entry"
print("P2 ledger format accepted (tier/execution aliases work)")

# 9. budget guard
assert check_budget(state, 1.0) and not check_budget(state, 1e-7)
print("\nAll tests passed.")