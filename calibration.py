"""Self-calibration: the engine learns from every run (token sizes, model win-rates,
estimate accuracy). Stored in history.json next to the code."""
import json, os

ALPHA = 0.3  # weight of the newest observation


def data_dir() -> str:
    d = os.environ.get("COST_ENGINE_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(d, exist_ok=True)
    return d


def data_path(name: str) -> str:
    return os.path.join(data_dir(), name)


def _load() -> dict:
    try:
        with open(data_path("history.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"nodes": {}, "models": {}, "accuracy": []}


def _save(h: dict) -> None:
    with open(data_path("history.json"), "w", encoding="utf-8") as f:
        json.dump(h, f, indent=1)


def node_stats(node: str):
    return _load()["nodes"].get(node)


def model_stats(model: str) -> dict:
    return _load()["models"].get(model, {"wins": 0, "losses": 0})


def total_runs() -> int:
    return len(_load()["accuracy"])


def mape() -> float | None:
    """Mean absolute % error of the last 10 estimates (None if no history)."""
    acc = _load()["accuracy"][-10:]
    return round(sum(a["err_pct"] for a in acc) / len(acc), 1) if acc else None


def _ema(old, new):
    return new if old is None else (1 - ALPHA) * old + ALPHA * new


def update_history(ledger, estimate_detail, estimated_cost, actual_cost, passed) -> None:
    h = _load()
    est_layers = {l["node"]: l for l in (estimate_detail or {}).get("layers", [])}
    seen = set()
    for e in ledger:
        n = e["node"]
        s = h["nodes"].setdefault(n, {"runs": 0})
        s["avg_out"] = _ema(s.get("avg_out"), e["completion_tokens"])
        if n in est_layers and n not in seen:
            base = est_layers[n].get("base_prompt_tokens", 0)
            if base > 0:
                ratio = min(5.0, max(0.2, e["prompt_tokens"] / base))
                s["prompt_scale"] = _ema(s.get("prompt_scale"), ratio)
        seen.add(n)
        s["runs"] += 1

    # model win-rate: only the LAST executor attempt can "win", earlier ones failed QA
    ex = [e for e in ledger if e["node"] == "executor"]
    for i, e in enumerate(ex):
        m = h["models"].setdefault(e["model"], {"wins": 0, "losses": 0})
        if i == len(ex) - 1 and passed:
            m["wins"] += 1
        else:
            m["losses"] += 1

    if estimated_cost and estimated_cost > 0:
        err = abs(actual_cost - estimated_cost) / estimated_cost * 100
        h["accuracy"].append({"est": round(estimated_cost, 6), "actual": round(actual_cost, 6),
                              "err_pct": round(err, 1)})
        h["accuracy"] = h["accuracy"][-50:]
    _save(h)