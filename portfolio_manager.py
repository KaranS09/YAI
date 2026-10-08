"""
AI MODEL PORTFOLIO MANAGER — end-to-end integrated pipeline.

Brings together three teammate components into one runnable LangGraph:

  * Cost Engine  (YAI/src/cost_engine)  -> Node A (pre-flight estimate) and
    Node F (post-run audit, baseline comparison, savings, scorecard).
  * Decision Layer (router.py + model_catalog.py) -> picks a model per request
    (semantic intent + offline difficulty + provider-neutral cost ranking),
    and feeds the SAME plan into the pre-flight estimate via expected_route().
  * LLM Caller (LiteLLM) -> Nodes B/C/D/E actually call the chosen model, with
    a try/except fallback to mock data so a rate-limit / missing key / offline
    run never crashes the demo.

Run (from the YAI folder):
    cd YAI && python portfolio_manager.py
"""

from __future__ import annotations

import json
import operator
import os
import sys
from typing import Annotated, Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# 0. Environment / provider keys  (requirement: config block up top)
# ---------------------------------------------------------------------------
# ONLY three providers are used, each via its own key — fill them in YAI/.env:
#   GEMINI_API_KEY      -> Google / Gemini   (model strings 'gemini/...')
#   OPENROUTER_API_KEY  -> OpenRouter        (model strings 'openrouter/.../...')
#   GROQ_API_KEY        -> Groq              (model strings 'groq/...')
# This file lives in YAI/, so _ROOT is the YAI folder itself.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_COST_ENGINE = os.path.join(_ROOT, "src", "cost_engine")
sys.path.insert(0, _COST_ENGINE)          # cost engine uses flat imports
# Keep the engine's learning/audit files tidy under YAI/.
os.environ.setdefault("COST_ENGINE_DATA_DIR", os.path.join(_ROOT, ".cost_engine_data"))

# Load keys from YAI/.env (sits next to this file). Real shell env still wins.
_ENV_PATH = os.path.join(_ROOT, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except Exception:
    pass
# Friendly alias: accept GOOGLE_API_KEY spelling for Gemini too.
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
for _key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
    os.environ.setdefault(_key, "")

from langgraph.graph import START, END, StateGraph

# Decision layer (loads the offline embedding brains once).
from router import ROUTER  # noqa: E402
from model_catalog import MODEL_CATALOG  # noqa: E402

# Cost engine (Node A / Node F logic).
from estimator import estimate            # noqa: E402  Node A
from ledger import audit                  # noqa: E402  Node F
from report import scorecard              # noqa: E402  terminal scorecard
from tokens import count_tokens           # noqa: E402  token counting

# LiteLLM is optional at runtime — absence just forces the mock path.
try:
    import logging as _logging
    import litellm
    from litellm import completion as _completion

    # Quiet LiteLLM's verbose error banners so the demo output stays clean.
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    _logging.getLogger("LiteLLM").setLevel(_logging.CRITICAL)
    os.environ.setdefault("LITELLM_LOG", "CRITICAL")
except Exception:  # pragma: no cover
    _completion = None


# ---------------------------------------------------------------------------
# 1. Unified state  (reducers make token_ledger / messages append-only)
# ---------------------------------------------------------------------------
class PortfolioState(TypedDict, total=False):
    query: str
    messages: Annotated[list, operator.add]        # append-only
    token_ledger: Annotated[list, operator.add]    # append-only (one entry/call)

    # routing / work products
    complexity: str
    current_plan: list
    draft_answer: str
    qa_verdict: str
    qa_reason: str

    # replan cycle guard
    replan_count: int
    max_retries: int

    # cost
    budget_mode: str
    max_cost: float
    estimated_cost: float
    estimate_detail: dict
    actual_cost: float
    cost_report: dict
    final_answer: str


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
_LAYER_OUT = {"triage": 30, "planner": 350, "execution": 500, "qa": 40}
# Per-layer output ceilings. Execution needs real headroom — truncating a
# complex answer at 1024 tokens makes the QA judge (correctly) fail it, which
# then triggers wasteful replans. Give each layer what it actually needs.
_LAYER_MAX_TOKENS = {"triage": 256, "planner": 2048, "execution": 4096, "qa": 1024}
_COMPLEX_MARKERS = ("analyze", "compare", "evaluate", "plan", "design",
                    "prove", "step by step", "architect", "optimize")

_SYS = {
    "triage": ('Classify the request as Simple or Complex. '
               'Return only JSON: {"complexity":"Simple"} or {"complexity":"Complex"}.'),
    "planner": ('Create a concise step-by-step plan. '
                'Return only JSON: {"steps":["step 1","step 2"]}.'),
    "execution": "Execute the plan and produce the best final answer.",
    "qa": ('Judge whether the draft fully solves the request. Return only JSON: '
           '{"verdict":"PASS","reason":"..."} or {"verdict":"FAIL","reason":"..."}'),
}


def _user_query(state: PortfolioState) -> str:
    """The original user request — skip assistant/tool messages so later
    layers never route on the executor's own output."""
    def is_user(m) -> bool:
        t = getattr(m, "type", None)
        if t is not None:
            return t == "human"
        return isinstance(m, dict) and m.get("role") == "user"

    for m in reversed(state.get("messages", [])):
        if is_user(m):
            return getattr(m, "content", None) or (
                m.get("content", "") if isinstance(m, dict) else str(m))
    return state.get("query", "")


def _looks_complex(query: str) -> bool:
    q = query.lower()
    return len(q) > 120 or any(m in q for m in _COMPLEX_MARKERS)


def _parse_json(raw: str, default: dict) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(raw[a:b + 1])
            except Exception:
                pass
    return default


def llm_call(tier: str, models: list[str], prompt: str, system_prompt: str):
    """Call the ranked models via LiteLLM, failing over LIVE through the list.

    `models` is the router's cheapest-qualifying-first list. We try each real
    provider in turn (each with its own retries); the first success wins. Mock
    is only ever used as an absolute last resort if EVERY provider fails
    (e.g. no keys / total outage) — so the graph can never crash a demo.

    Returns (content, ledger_entry, is_mock)."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    est_prompt = count_tokens(system_prompt + "\n" + prompt)

    if _completion is not None:
        for model_id in models:
            try:
                # num_retries handles transient blips (503/timeout/rate limit)
                # on THIS model before we move on to the next provider.
                resp = _completion(model=model_id, messages=messages,
                                   temperature=0.2,
                                   max_tokens=_LAYER_MAX_TOKENS.get(tier, 1024),
                                   timeout=30, num_retries=2)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or est_prompt)
                ct = int(getattr(usage, "completion_tokens", 0) or _LAYER_OUT.get(tier, 200))
                entry = {
                    "tier": tier, "model": model_id,
                    "provider": MODEL_CATALOG.get(model_id, {}).get("provider", "unknown"),
                    "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                    "estimated_prompt_tokens": est_prompt, "mock": False,
                }
                return content, entry, False
            except Exception as exc:
                nxt = "mock" if model_id == models[-1] else "next provider"
                print(f"      [llm] {model_id} failed ({type(exc).__name__}); failing over to {nxt}")
                continue

    # Absolute last resort — every provider failed (or no LiteLLM). Keeps the
    # graph alive; with working keys this is never reached.
    model_id = models[0]
    entry = {
        "tier": tier, "model": model_id,
        "provider": MODEL_CATALOG.get(model_id, {}).get("provider", "unknown"),
        "prompt_tokens": est_prompt, "completion_tokens": _LAYER_OUT.get(tier, 200),
        "total_tokens": est_prompt + _LAYER_OUT.get(tier, 200),
        "estimated_prompt_tokens": est_prompt, "mock": True,
    }
    return f'[MOCK:{model_id}] response for a {tier} step', entry, True


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def node_a_preflight(state: PortfolioState) -> PortfolioState:
    """Node A — Cost Engine pre-flight estimate (routes via our decision layer)."""
    query = state.get("query") or _user_query(state)
    try:
        detail = estimate(query, state.get("budget_mode", "balanced"), state.get("max_cost"))
    except Exception as exc:  # never crash the graph on the estimator
        print(f"[Node A] estimate failed ({type(exc).__name__}); using safe default")
        detail = {"run_id": "fallback", "mode": state.get("budget_mode", "balanced"),
                  "complexity": "simple", "estimated_cost": 0.0,
                  "estimated_cost_worst_case": 0.0, "baseline_model": "gpt-4o",
                  "baseline_estimate": 0.0, "decision": "APPROVED",
                  "decision_note": "estimator unavailable", "route_source": "fallback",
                  "confidence": "low", "calibration_runs": 0, "baselines": {}}

    print(f"[Node A] Pre-Flight: est ${detail['estimated_cost']:.5f} "
          f"(worst ${detail.get('estimated_cost_worst_case', 0):.5f}) | "
          f"baseline {detail['baseline_model']} ${detail.get('baseline_estimate', 0):.5f} | "
          f"policy {detail.get('decision')}")
    return {"estimated_cost": detail["estimated_cost"], "estimate_detail": detail}


def route_after_preflight(state: PortfolioState) -> Literal["blocked", "continue"]:
    return "blocked" if state.get("estimate_detail", {}).get("decision") == "BLOCKED" else "continue"


def node_b_triage(state: PortfolioState) -> PortfolioState:
    query = _user_query(state)
    models = ROUTER.rank("triage", query, state)
    content, entry, is_mock = llm_call("triage", models, query, _SYS["triage"])
    complexity = ("Complex" if _looks_complex(query) else "Simple") if is_mock \
        else _parse_json(content, {"complexity": "Complex"}).get("complexity", "Complex")
    print(f"[Node B] Triage -> {entry['model']} | complexity={complexity}")
    return {"complexity": complexity, "token_ledger": [entry]}


def route_after_triage(state: PortfolioState) -> Literal["execution", "planner"]:
    return "execution" if state.get("complexity") == "Simple" else "planner"


def node_c_planner(state: PortfolioState) -> PortfolioState:
    query = _user_query(state)
    models = ROUTER.rank("planner", query, state)
    content, entry, is_mock = llm_call("planner", models, query, _SYS["planner"])
    plan = (["Understand the request", "Do the work", "Verify the result"] if is_mock
            else _parse_json(content, {"steps": [query]}).get("steps", [query]))
    print(f"[Node C] Planner -> {entry['model']} | {len(plan)} steps")
    return {"current_plan": plan, "token_ledger": [entry]}


def node_d_execution(state: PortfolioState) -> PortfolioState:
    query = _user_query(state)
    plan = state.get("current_plan") or [query]
    models = ROUTER.rank("execution", query, state)
    content, entry, is_mock = llm_call(
        "execution", models, f"Request:\n{query}\n\nPlan:\n{json.dumps(plan)}", _SYS["execution"])
    draft = f"[draft answer for: {query[:48]}]" if is_mock else content
    print(f"[Node D] Execution -> {entry['model']}")
    return {"draft_answer": draft,
            "messages": [{"role": "assistant", "content": draft}],
            "token_ledger": [entry]}


def node_e_qa(state: PortfolioState) -> PortfolioState:
    query = _user_query(state)
    draft = state.get("draft_answer", "")
    models = ROUTER.rank("qa", query, state)
    content, entry, is_mock = llm_call(
        "qa", models, f"Request:\n{query}\n\nDraft:\n{draft}", _SYS["qa"])

    replan = int(state.get("replan_count", 0))
    if is_mock:
        # Fail once to demonstrate the replan cycle, then pass.
        verdict = "PASS" if replan >= 1 else "FAIL"
        reason = "mock audit (demo)"
    else:
        parsed = _parse_json(content, {"verdict": "FAIL", "reason": "unparseable"})
        verdict, reason = parsed.get("verdict", "FAIL"), parsed.get("reason", "")
    passed = verdict == "PASS"

    # Feed the verdict into the decision layer's learning loop (credit the
    # execution model that produced the judged draft).
    exec_model = next((e["model"] for e in reversed(state.get("token_ledger", []))
                       if e["tier"] == "execution"), entry["model"])
    ROUTER.record_outcome(exec_model, passed)

    print(f"[Node E] QA -> {entry['model']} | verdict={verdict} (replan={replan})")
    out: PortfolioState = {"qa_verdict": verdict, "qa_reason": reason,
                           "token_ledger": [entry]}
    if not passed:
        out["replan_count"] = replan + 1
    return out


def route_after_qa(state: PortfolioState) -> Literal["post_run", "planner"]:
    passed = state.get("qa_verdict") == "PASS"
    if passed or int(state.get("replan_count", 0)) >= int(state.get("max_retries", 2)):
        return "post_run"
    return "planner"


def node_f_post_run(state: PortfolioState) -> PortfolioState:
    """Node F — Cost Engine post-run audit + baseline comparison + scorecard."""
    ledger = state.get("token_ledger", [])
    est = state.get("estimated_cost", 0.0)
    detail = state.get("estimate_detail")
    passed = state.get("qa_verdict") == "PASS"
    try:
        result = audit(ledger, est, detail, passed)
    except Exception as exc:
        print(f"[Node F] audit failed ({type(exc).__name__}); minimal report")
        result = {"actual_cost": 0.0, "savings_pct": 0.0, "cost_breakdown": [],
                  "cost_report": {}}

    try:
        print("\n" + scorecard(detail or {}, result))
    except Exception:
        pass  # scorecard is cosmetic; never let it break the run

    return {"cost_report": result.get("cost_report", {}),
            "actual_cost": result.get("actual_cost", 0.0),
            "final_answer": state.get("draft_answer", "")}


# ---------------------------------------------------------------------------
# 4. Graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(PortfolioState)
    g.add_node("preflight", node_a_preflight)   # A
    g.add_node("triage", node_b_triage)          # B
    g.add_node("planner", node_c_planner)        # C
    g.add_node("execution", node_d_execution)    # D
    g.add_node("qa", node_e_qa)                  # E
    g.add_node("post_run", node_f_post_run)      # F

    g.add_edge(START, "preflight")
    g.add_conditional_edges("preflight", route_after_preflight,
                            {"blocked": END, "continue": "triage"})
    g.add_conditional_edges("triage", route_after_triage,
                            {"execution": "execution", "planner": "planner"})
    g.add_edge("planner", "execution")
    g.add_edge("execution", "qa")
    g.add_conditional_edges("qa", route_after_qa,
                            {"post_run": "post_run", "planner": "planner"})
    g.add_edge("post_run", END)
    return g.compile()


# ---------------------------------------------------------------------------
# 5. Runnable test harness
# ---------------------------------------------------------------------------
def _summary(query: str, final: PortfolioState) -> None:
    detail = final.get("estimate_detail", {})
    est = final.get("estimated_cost", 0.0)
    actual = final.get("actual_cost", 0.0)
    report = final.get("cost_report", {})
    baselines = report.get("baselines", {}) or detail.get("baselines", {})
    base_model = detail.get("baseline_model", "gpt-4o")
    base_cost = baselines.get(base_model, 0.0)

    print("\n" + "#" * 64)
    print(f"# HEADLINE SUMMARY")
    print("#" * 64)
    print(f"Query            : {query[:70]}")
    print(f"Pre-flight est.  : ${est:.5f}")
    print(f"Post-run actual  : ${actual:.5f}")
    print(f"Monolith baseline: ${base_cost:.5f}  ({base_model}, same tokens)")
    if base_cost:
        saved = base_cost - actual
        print(f"SAVINGS          : ${saved:.5f}  ({saved / base_cost * 100:.1f}% cheaper than monolith)")

    print("\nLayer-by-layer (from token_ledger):")
    print(f"  {'tier':<10}{'provider':<10}{'model':<40}{'tok(in/out)':>14}")
    for e in final.get("token_ledger", []):
        toks = f"{e['prompt_tokens']}/{e['completion_tokens']}"
        print(f"  {e['tier']:<10}{e['provider']:<10}{e['model'][:38]:<40}{toks:>14}")
    print("#" * 64 + "\n")


if __name__ == "__main__":
    app = build_graph()
    print(f"Decision-layer backend: {ROUTER.backend}\n")

    demos = [
        "What is the capital of France?",                       # simple -> skip planner
        "Analyze our Q3 cloud spend and design a step-by-step plan to cut it 20%.",  # complex
    ]
    for q in demos:
        print("=" * 64)
        print(f"USER QUERY: {q}")
        print("=" * 64)
        final = app.invoke({
            "query": q,
            "messages": [{"role": "user", "content": q}],
            "replan_count": 0,
            "max_retries": 2,
            "budget_mode": "balanced",
        })
        _summary(q, final)
