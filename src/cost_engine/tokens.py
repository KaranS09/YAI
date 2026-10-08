from __future__ import annotations
"""Token counting + per-layer token assumptions (auto-calibrated from history)."""
from calibration import node_stats

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:  # offline -> rough fallback
    _ENC = None


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text)) if _ENC is not None else max(1, len(text) // 4)


# Starting guesses. The engine overrides them with learned values once it has
# history. These are deliberately realistic/generous so the cold-start estimate
# is a sensible CEILING (actual should land at or under it): the QA step in
# particular reads the entire draft answer, so its input is large.
LAYER_ASSUMPTIONS = {
    "triage":   {"overhead": 160, "sees_query": 1.0, "out": 60},
    "planner":  {"overhead": 300, "sees_query": 1.0, "out": 500},
    "executor": {"overhead": 400, "sees_query": 1.0, "out": 750},
    "qa":       {"overhead": 700, "sees_query": 1.0, "out": 150},
}


def layer_tokens(node: str, q_tokens: int):
    """Return (expected_prompt_tokens, expected_completion_tokens, uncalibrated_prompt_tokens)."""
    a = LAYER_ASSUMPTIONS[node]
    base_in = a["overhead"] + int(q_tokens * a["sees_query"])
    p_in, p_out = base_in, a["out"]
    cal = node_stats(node)
    if cal and cal.get("runs", 0) > 0:
        p_in = int(base_in * cal.get("prompt_scale", 1.0))
        p_out = int(cal.get("avg_out", p_out))
    return p_in, p_out, base_in