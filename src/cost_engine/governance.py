"""Governance: budget policies + tamper-evident audit log (hash chain)."""
import json, os, hashlib
from datetime import datetime, timezone
from calibration import data_path

POLICIES = {  # USD per query, worst case incl. one replan. TUNE for your demo.
    "cost-first":    {"max_cost_per_query": 0.0025},
    "balanced":      {"max_cost_per_query": 0.0060},
    "quality-first": {"max_cost_per_query": 0.0200},
}
DAILY_BUDGET_USD = float(os.environ.get("COST_ENGINE_DAILY_BUDGET", "1.0"))


def check_policy(worst_cost, mode, cap=None, spent_today=0.0) -> dict:
    cap = cap if cap is not None else POLICIES[mode]["max_cost_per_query"]
    info = {"per_query_cap": cap, "daily_budget": DAILY_BUDGET_USD, "spent_today": round(spent_today, 6)}
    if spent_today + worst_cost > DAILY_BUDGET_USD:
        return {"ok": False, "kind": "daily", "reason": "daily budget would be exceeded", **info}
    if worst_cost > cap:
        return {"ok": False, "kind": "per_query",
                "reason": f"worst-case ${worst_cost:.5f} exceeds per-query cap ${cap:.5f}", **info}
    return {"ok": True, "kind": None, "reason": "within policy", **info}


# ---- audit log -------------------------------------------------------------
def _hash(prev: str, rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()


def _read() -> list:
    p = data_path("audit_log.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_audit(record: dict) -> dict:
    recs = _read()
    prev = recs[-1]["hash"] if recs else "GENESIS"
    rec = dict(record)
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    rec["prev_hash"] = prev
    rec["hash"] = _hash(prev, rec)
    with open(data_path("audit_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def verify_audit() -> tuple:
    """(ok, n_records, first_bad_index_or_None). Editing any past record breaks the chain."""
    prev = "GENESIS"
    recs = _read()
    for i, r in enumerate(recs):
        if r.get("prev_hash") != prev or r.get("hash") != _hash(prev, r):
            return False, len(recs), i
        prev = r["hash"]
    return True, len(recs), None


def spent_today() -> float:
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(r.get("actual_cost", 0.0) for r in _read() if r.get("ts", "").startswith(today))