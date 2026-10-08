"""
The Router  ("Jev")  —  provider-neutral, offline-capable model selection.

It combines TWO techniques, both running locally (no API keys, no network
after the one-time model download):

  1. semantic-router (real package)  -> INTENT / capability classification.
     "Is this a coding task? math? vision? long-context? small talk?"

  2. RouteLLM's routing *idea*, reimplemented on a local FastEmbed encoder
     -> DIFFICULTY score in [0,1]  ("how strong a model does this need?").
     We do NOT use the RouteLLM package: its hosted `mf` router embeds every
     prompt via a paid OpenAI call, which breaks offline use and bakes in
     vendor bias — the two things this project explicitly avoids. We keep the
     algorithm (embed prompt, score against reference anchors) and drop the
     vendor dependency.

Decision = features(THIS request) -> model, over the whole catalog:
     required_quality  (difficulty, data-derived)
   + required_tags     (intent + a soft per-layer floor)
   + context fit
   -> filter the catalog -> rank by TOKEN-WEIGHTED expected cost
      (divided by each model's measured success rate) -> cheapest wins.

The layer is just ONE feature (a soft floor), never a hard model mapping — so
the same layer can land on different providers depending on the request. No
provider is ever preferred by name.

Resilience: if the embedding stack can't load (missing package / offline with
no cached model), the Router transparently falls back to a heuristic so the
graph still runs. It logs which brain is active.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from model_catalog import MODEL_CATALOG, AVAILABLE_MODELS

log = logging.getLogger("router")


# ---------------------------------------------------------------------------
# Reference anchors (seed data, not trained weights).
# Difficulty = where the prompt sits on the easy<->hard axis, measured by
# embedding similarity to these examples. Teammates can extend these lists.
# ---------------------------------------------------------------------------
_EASY_ANCHORS = [
    "hi", "hello there", "what is 2 + 2",
    "say thanks", "what's the capital of France",
    "translate 'cat' to Spanish",
]
_HARD_ANCHORS = [
    "prove that sqrt(2) is irrational with a rigorous argument",
    "design a fault-tolerant distributed rate limiter for 1M requests per second",
    "derive the backpropagation equations for a transformer layer",
    "plan a multi-step migration of a monolith to microservices with rollback",
    "analyze this legal contract for conflicting indemnity clauses",
]

# Intent routes for semantic-router. name -> required capability tags.
# This adds CONSTRAINTS (what the model must be able to do); among models that
# qualify, cost still decides — so it is not model earmarking.
_INTENT_ROUTES = {
    "code":        {"utterances": ["write a function", "fix this bug", "refactor this code",
                                   "implement an algorithm", "debug a stack trace"],
                    "requires": {"code"}},
    "math_logic":  {"utterances": ["prove this theorem", "solve this equation",
                                   "reason step by step", "evaluate this argument"],
                    "requires": {"reasoning"}},
    "analysis":    {"utterances": ["analyze the trade-offs", "compare these options and recommend",
                                   "evaluate the pros and cons", "assess and recommend",
                                   "weigh the options and decide"],
                    "requires": {"reasoning"}},
    "vision":      {"utterances": ["what's in this image", "describe this screenshot",
                                   "read this chart", "analyze this photo"],
                    "requires": {"vision"}},
    "long_doc":    {"utterances": ["summarize this long document word by word",
                                   "read these 200 pages and extract", "condense this lengthy transcript",
                                   "review this entire 50-page report"],
                    "requires": {"long_context"}},
    "chitchat":    {"utterances": ["hello", "how are you", "thanks", "good morning"],
                    "requires": set()},
}

# Soft per-layer quality floors. A FLOOR, not a mapping — difficulty can push
# higher, and any qualifying provider can win. Kept gentle on purpose so the
# request (not the layer) drives the choice. Tunable; the learning loop can
# make these redundant over time.
_LAYER_QUALITY_FLOOR = {
    "triage": 0.0,      # pure difficulty-driven
    "execution": 0.0,   # pure difficulty-driven
    "planner": 0.45,    # planning benefits from some reasoning capability
    "qa": 0.55,         # auditing must be reasonably strong
}

# Cap on the quality bar. Difficulty saturates toward 1.0 on hard prompts, but
# demanding quality==1.0 excludes every real model and forces the expensive
# fallback. Capping here means a hard prompt first tries a capable mid-tier
# model (cheap); QA + the replan cycle escalate if that proves insufficient.
_MAX_REQUIRED_QUALITY = 0.80

# Rough per-layer output-token expectation (until Node A supplies real
# estimates). Input tokens are estimated from the query length. This is what
# makes cost ranking TOKEN-WEIGHTED rather than a flat input+output sum.
_LAYER_OUTPUT_TOKENS = {"triage": 50, "planner": 600, "execution": 1200, "qa": 300}


class Router:
    def __init__(self) -> None:
        self._embed = None          # FastEmbed model (difficulty)
        self._route_layer = None    # semantic-router (intent)
        self._easy_vecs = None
        self._hard_vecs = None
        self._diff_lo = 0.0         # calibration bounds, derived from anchors
        self._diff_hi = 1.0
        # Online-learning state: per-model (passes, attempts) from QA outcomes.
        self._stats: dict[str, dict[str, int]] = {
            m: {"passes": 0, "attempts": 0} for m in AVAILABLE_MODELS
        }
        self._backend = "heuristic"
        self._init_embedding_brains()

    # -- brain initialization (with graceful fallback) ----------------------
    def _init_embedding_brains(self) -> None:
        try:
            import numpy as np
            from fastembed import TextEmbedding

            self._np = np
            self._embed = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._easy_vecs = [self._vec(t) for t in _EASY_ANCHORS]
            self._hard_vecs = [self._vec(t) for t in _HARD_ANCHORS]
            # Calibrate the difficulty scale from the anchors themselves
            # (data-derived bounds — no hand-set thresholds).
            easy_scores = [self._raw_difficulty(v) for v in self._easy_vecs]
            hard_scores = [self._raw_difficulty(v) for v in self._hard_vecs]
            self._diff_lo = sum(easy_scores) / len(easy_scores)
            self._diff_hi = sum(hard_scores) / len(hard_scores)

            # semantic-router for intent (its actual job)
            from semantic_router import RouteLayer, Route
            from semantic_router.encoders import FastEmbedEncoder

            routes = [Route(name=n, utterances=cfg["utterances"])
                      for n, cfg in _INTENT_ROUTES.items()]
            self._route_layer = RouteLayer(encoder=FastEmbedEncoder(), routes=routes)
            self._backend = "semantic-router + local-embedding difficulty"
            log.info("[router] brains loaded: %s", self._backend)
        except Exception as e:  # offline w/o cache, or package missing
            self._backend = "heuristic fallback"
            log.warning("[router] embedding brains unavailable (%s) — using %s",
                        type(e).__name__, self._backend)

    # -- embedding helpers --------------------------------------------------
    def _vec(self, text: str):
        return self._np.array(list(self._embed.embed([text]))[0])

    def _cos(self, a, b) -> float:
        return float(a @ b / (self._np.linalg.norm(a) * self._np.linalg.norm(b)))

    def _raw_difficulty(self, v) -> float:
        h = max(self._cos(v, x) for x in self._hard_vecs)
        e = max(self._cos(v, x) for x in self._easy_vecs)
        return h / (h + e)

    # -- feature extraction -------------------------------------------------
    def _difficulty(self, query: str) -> float:
        """0..1 'how strong a model does this need', RouteLLM-style."""
        if self._embed is None:
            # Heuristic: longer / question-dense prompts skew harder.
            n = len(query)
            return max(0.0, min(1.0, (n - 20) / 400))
        raw = self._raw_difficulty(self._vec(query))
        # Normalize against anchor-derived bounds, then clamp.
        span = max(self._diff_hi - self._diff_lo, 1e-6)
        return max(0.0, min(1.0, (raw - self._diff_lo) / span))

    def _intent(self, query: str) -> tuple[str, set[str]]:
        """(intent_name, required_tags) via semantic-router, with fallback."""
        if self._route_layer is not None:
            name = self._route_layer(query).name
            if name and name in _INTENT_ROUTES:
                return name, set(_INTENT_ROUTES[name]["requires"])
        # Heuristic keyword fallback
        q = query.lower()
        if any(k in q for k in ("code", "bug", "function", "refactor", "python")):
            return "code", {"code"}
        if any(k in q for k in ("prove", "solve", "reason", "equation")):
            return "math_logic", {"reasoning"}
        if any(k in q for k in ("image", "screenshot", "photo", "chart")):
            return "vision", {"vision"}
        return "general", set()

    @staticmethod
    def _est_input_tokens(query: str) -> int:
        return max(1, len(query) // 4)  # ~4 chars/token; replaced by Node A later

    def _success_rate(self, model: str) -> float:
        s = self._stats[model]
        # Optimistic prior (1.0) so new/untried models aren't cold-start-penalized.
        return (s["passes"] + 1) / (s["attempts"] + 1) if s["attempts"] else 1.0

    # -- the decision -------------------------------------------------------
    def rank(self, layer: str, query: str, state: dict | None = None) -> list[str]:
        """Return qualifying models for this layer, CHEAPEST-QUALIFYING FIRST.

        The caller tries them in order and fails over to the next on any live
        error — so a provider outage escalates to the next-cheapest real model
        instead of ever falling back to mock.
        """
        difficulty = self._difficulty(query)
        intent, required_tags = self._intent(query)
        required_quality = max(difficulty, _LAYER_QUALITY_FLOOR.get(layer, 0.0))
        required_quality = min(required_quality, _MAX_REQUIRED_QUALITY)
        est_in = self._est_input_tokens(query)
        est_out = _LAYER_OUTPUT_TOKENS.get(layer, 500)

        # 1) FILTER: capability tags, quality floor, context fit.
        candidates = []
        for m in AVAILABLE_MODELS:
            spec = MODEL_CATALOG[m]
            if not required_tags.issubset(spec["tags"]):
                continue
            if spec["quality"] < required_quality:
                continue
            if spec["context"] < est_in:
                continue
            candidates.append(m)

        # Graceful degradation: nothing clears the bar -> allow all models
        # (still ranked by cost) rather than crash.
        relaxed = False
        if not candidates:
            relaxed = True
            candidates = list(AVAILABLE_MODELS)

        # 2) RANK: token-weighted expected cost, retry-adjusted by success rate.
        #    Lower = better. No provider term anywhere -> vendor-neutral.
        def expected_cost(m: str) -> float:
            spec = MODEL_CATALOG[m]
            raw = (spec["cost_in"] * est_in + spec["cost_out"] * est_out) / 1_000_000
            return raw / max(self._success_rate(m), 0.1)

        ranked = sorted(candidates, key=expected_cost)
        chosen = ranked[0]
        spec = MODEL_CATALOG[chosen]

        log.info(
            "   [route] layer=%-9s intent=%-9s diff=%.2f req_q=%.2f -> %-16s "
            "(%s, $%.2f/$%.2f)%s  [%d candidates]",
            layer, intent, difficulty, required_quality, chosen,
            spec["provider"], spec["cost_in"], spec["cost_out"],
            "  [relaxed]" if relaxed else "", len(ranked),
        )

        if state is not None:
            state.setdefault("_trace", []).append({
                "layer": layer, "intent": intent, "difficulty": round(difficulty, 3),
                "required_quality": round(required_quality, 3),
                "chosen": chosen, "provider": spec["provider"],
                "candidates": len(ranked),
            })
        return ranked

    def choose(self, layer: str, query: str, state: dict | None = None) -> str:
        """Single best model (used by the pre-flight estimator's expected_route)."""
        return self.rank(layer, query, state)[0]

    # -- online learning: fed by QA outcomes --------------------------------
    def record_outcome(self, model: str, qa_passed: bool, cost: float | None = None) -> None:
        """Called after QA. This is the self-improving loop: models that keep
        passing get cheaper 'expected cost' (via success rate), models that
        fail get quietly down-weighted. The reward signal is YOUR pipeline's
        QA verdict — no external labels needed."""
        if model not in self._stats:
            self._stats[model] = {"passes": 0, "attempts": 0}
        self._stats[model]["attempts"] += 1
        if qa_passed:
            self._stats[model]["passes"] += 1

    @property
    def backend(self) -> str:
        return self._backend


# Module-level singleton so the embedding model loads exactly once.
ROUTER = Router()


# ---------------------------------------------------------------------------
# Pre-flight hook for the cost engine.
# The cost engine's estimator does `from router import expected_route` so it can
# price the SAME plan our decision layer will actually run. This makes the
# Node A estimate and the Node F actual use one brain — the auditor then
# measures the (small) gap instead of comparing two unrelated routers.
# ---------------------------------------------------------------------------
def expected_route(query: str, mode: str = "balanced") -> dict:
    """Return {node_name: model_id} planned by the real decision layer.

    Node names use the cost engine's vocabulary ('execution' is normalized to
    'executor' on its side). Simple queries skip the planner.
    """
    import logging as _logging

    complex_markers = ("analyze", "compare", "evaluate", "plan", "design",
                       "prove", "step by step", "architect", "optimize")
    is_simple = (len(query) < 80
                 and not any(m in query.lower() for m in complex_markers))
    nodes = ["triage", "execution", "qa"] if is_simple else \
            ["triage", "planner", "execution", "qa"]

    # Silence the per-pick logs during planning so they don't duplicate the
    # runtime routing logs.
    prev = log.level
    log.setLevel(_logging.WARNING)
    try:
        return {n: ROUTER.choose(n, query, None) for n in nodes}
    finally:
        log.setLevel(prev)
