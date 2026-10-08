from __future__ import annotations
"""Pricing registry: prices, quality priors, role eligibility."""
import json, os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pricing.json")
with open(_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)

MODELS = _DATA["models"]
META = _DATA["_meta"]
BASELINE_MODEL = META["baseline_model"]
BASELINES = [k for k, v in MODELS.items() if v.get("baseline")]
_warned = set()


def resolve(model: str):
    """Return the registry key for a model string (forgiving), or None."""
    if model in MODELS:
        return model
    cleaned = model.replace(":free", "").lower()
    for key in MODELS:
        k = key.replace(":free", "").lower()
        if cleaned == k or cleaned.endswith("/" + k) or k.endswith("/" + cleaned):
            return key
    return None


def get_price(model: str) -> dict:
    key = resolve(model)
    if key:
        return MODELS[key]
    if model not in _warned:
        print(f"[cost_engine] WARNING: no price for '{model}', using default")
        _warned.add(model)
    return META["default_price"]


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = get_price(model)
    return (prompt_tokens * p["input_per_1m"] + completion_tokens * p["output_per_1m"]) / 1_000_000


def quality_prior(model: str) -> float:
    key = resolve(model)
    return MODELS[key].get("quality", 5.0) if key else 5.0


def models_for_role(role: str, allow_paid: bool = False) -> list:
    return [k for k, v in MODELS.items()
            if role in v.get("roles", []) and (allow_paid or v.get("is_free"))]


# ---- compatibility with P2's wrapper: {"tier": "execution", ...} -----------------
NODE_ALIASES = {"execution": "executor", "exec": "executor", "planning": "planner",
                "classify": "triage", "verify": "qa", "evaluator": "qa"}


def norm_node(name: str) -> str:
    name = (name or "unknown").lower()
    return NODE_ALIASES.get(name, name)


def normalize_ledger(ledger: list) -> list:
    """Accept entries keyed by 'node' OR 'tier'; map names to ours; keep extra keys."""
    out = []
    for e in ledger or []:
        e = dict(e)
        e["node"] = norm_node(e.get("node") or e.get("tier"))
        e["prompt_tokens"] = e.get("prompt_tokens") or 0
        e["completion_tokens"] = e.get("completion_tokens") or 0
        out.append(e)
    return out


def normalize_route(plan: dict) -> dict:
    return {norm_node(k): v for k, v in (plan or {}).items()}


def to_p2_names(plan: dict) -> dict:
    return {("execution" if k == "executor" else k): v for k, v in plan.items()}