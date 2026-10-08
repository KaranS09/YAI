from __future__ import annotations
"""Plain-text scorecard for terminal demos (the UI can reuse the same dicts later)."""


def scorecard(estimate_detail: dict, result: dict) -> str:
    d, r = estimate_detail, result["cost_report"]
    L = []
    L.append("=" * 64)
    L.append(f" AI MODEL PORTFOLIO MANAGER  | run {d['run_id']} | mode: {d['mode']} | {d['complexity']}")
    L.append("=" * 64)
    L.append(f" PRE-FLIGHT  est ${d['estimated_cost']:.5f} (worst ${d['estimated_cost_worst_case']:.5f})"
             f"  vs {d['baseline_model']} ${d['baseline_estimate']:.5f}")
    L.append(f"   policy: {d['decision']} - {d['decision_note']}")
    L.append(f"   route source: {d['route_source']} | confidence: {d['confidence']}"
             f" ({d['calibration_runs']} calibration runs)")
    L.append("-" * 64)
    L.append(f" {'step':<9}{'try':<5}{'model':<44}{'cost':>8}")
    for row in result["cost_breakdown"]:
        tag = " *WASTE" if row["is_waste"] else ""
        L.append(f" {row['node']:<9}{row['attempt']:<5}{row['model'][:42]:<44}${row['cost']:.5f}{tag}")
    L.append("-" * 64)
    L.append(f" ACTUAL ${result['actual_cost']:.5f} | estimate was ${r['variance']['estimated']:.5f}"
             f" ({r['variance']['diff_pct']}% off)")
    for m, pct in r["savings_pct_by_baseline"].items():
        L.append(f"   saved {pct:>5}% vs monolith {m} (${r['baselines'][m]:.5f})")
    L.append(f" WASTE  ${r['waste']['cost']:.5f} ({r['waste']['pct_of_total']}% of spend) in "
             f"{r['waste']['wasted_calls']} rejected/retried calls | attempts: {r['attempts']}")
    cps = r["cost_per_successful_task"]
    L.append(f" COST PER SUCCESSFUL TASK: {'$%.5f' % cps if cps is not None else 'TASK FAILED (cost of failure $%.5f)' % r['cost_of_failed_task']}")
    L.append("=" * 64)
    return "\n".join(L)