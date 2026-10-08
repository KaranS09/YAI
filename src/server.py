"""
FastAPI server for the AI Model Portfolio Manager UI.

  * GET  /                -> serves the single-page UI (ui/index.html)
  * GET  /api/run?prompt= -> Server-Sent Events stream of the live pipeline:
       preflight (predicted cost)  ->  per-node events (model TIER + tokens
       + cost, as control flows A->B->C->D->E->F)  ->  done (actual cost,
       savings, final answer).

Run (from the YAI folder):
    cd YAI && python src/server.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import queue
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

# Build the pipeline once (loads embedding brains, wires cost engine + keys).
import portfolio_manager as pm
from model_catalog import MODEL_CATALOG

# This file lives in YAI/src/; _ROOT points at the YAI project folder (for ui/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_GRAPH = pm.build_graph()

app = FastAPI(title="AI Model Portfolio Manager")


# ---------------------------------------------------------------------------
# Cost tier: present the chosen model as Cheap / Medium / Costly (not a name).
# Tiered by blended $/1M (input+output) so the UI shows meaningful variety:
# our portfolio models land cheap/medium; frontier models would be costly.
# ---------------------------------------------------------------------------
def cost_tier(model_id: str) -> str:
    spec = MODEL_CATALOG.get(model_id, {})
    blended = spec.get("cost_in", 0) + spec.get("cost_out", 0)
    if blended <= 0.55:
        return "cheap"
    if blended <= 1.0:
        return "medium"
    return "costly"


# Human-friendly model names for the UI (never show raw ids).
_PRETTY = {
    "gpt-4o": "GPT-4o",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini/gemini-3.5-flash": "Gemini 3.5 Flash",
    "groq/openai/gpt-oss-20b": "GPT-OSS 20B",
    "groq/openai/gpt-oss-120b": "GPT-OSS 120B",
    "openrouter/openai/gpt-4o-mini": "GPT-4o mini",
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "openrouter/meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "openrouter/anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
}


def pretty_model(model_id: str) -> str:
    if not model_id:
        return ""
    return _PRETTY.get(model_id, model_id.split("/")[-1])


_NODE_LABELS = {
    "triage": "Triage", "planner": "Planner",
    "execution": "Execution", "qa": "QA / Verifier",
}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def run_stream(prompt: str):
    """Yield SSE events as the LangGraph pipeline streams node updates."""
    yield _sse({"type": "start", "prompt": prompt})

    initial = {
        "query": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "replan_count": 0,
        "max_retries": 2,
        "budget_mode": "balanced",
    }

    final_state: dict = {}
    try:
        for update in APP_GRAPH.stream(initial, stream_mode="updates"):
            for node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                final_state.update({k: v for k, v in delta.items()
                                    if k not in ("messages", "token_ledger")})

                if node == "preflight":
                    d = delta.get("estimate_detail", {})
                    est_plan = d.get("route_plan", {}) or {}
                    est_model = est_plan.get("executor") or est_plan.get("execution")
                    # Show the WORST-CASE estimate (includes one replan) as the
                    # predicted cost, so it's a sensible ceiling actual lands under.
                    predicted = round(d.get("estimated_cost_worst_case")
                                      or delta.get("estimated_cost", 0.0), 6)
                    yield _sse({
                        "type": "preflight",
                        "predicted_cost": predicted,
                        "baseline_cost": round(d.get("baseline_estimate", 0.0), 6),
                        "baseline_model": pretty_model(d.get("baseline_model", "")) or "GPT-4o",
                        "est_model": pretty_model(est_model),
                        "complexity": d.get("complexity"),
                        "policy": d.get("decision"),
                    })

                elif node in ("triage", "planner", "execution", "qa"):
                    ledger = delta.get("token_ledger", [])
                    e = ledger[-1] if ledger else {}
                    model = e.get("model", "")
                    yield _sse({
                        "type": "node",
                        "node": node,
                        "label": _NODE_LABELS.get(node, node),
                        "tier": cost_tier(model),
                        "model": pretty_model(model),
                        "provider": e.get("provider", ""),
                        "prompt_tokens": e.get("prompt_tokens", 0),
                        "completion_tokens": e.get("completion_tokens", 0),
                        "cost": round(_entry_cost(e), 6),
                        "complexity": delta.get("complexity"),
                        "verdict": delta.get("qa_verdict"),
                        "replan": int(delta.get("replan_count", 0)),
                    })

                elif node == "post_run":
                    report = delta.get("cost_report", {}) or {}
                    _ed = final_state.get("estimate_detail", {}) or {}
                    predicted = round(_ed.get("estimated_cost_worst_case")
                                      or final_state.get("estimated_cost", 0.0), 6)
                    actual = round(delta.get("actual_cost", 0.0), 6)
                    base = (report.get("baselines", {}) or {}).get(
                        report.get("baseline_model", ""), 0.0)
                    # savings_pct is a top-level audit field (not in cost_report);
                    # compute it from the baseline + actual so it's always right.
                    saved_pct = round((base - actual) / base * 100, 1) if base else 0.0
                    yield _sse({
                        "type": "done",
                        "predicted_cost": predicted,
                        "actual_cost": actual,
                        "baseline_cost": round(base, 6),
                        "baseline_model": pretty_model(report.get("baseline_model", "")) or "GPT-4o",
                        "savings_pct": saved_pct,
                        "verdict": final_state.get("qa_verdict"),
                        "replans": int(final_state.get("replan_count", 0)),
                        "answer": (delta.get("final_answer") or "")[:4000],
                    })
    except Exception as exc:  # never hang the UI
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    yield _sse({"type": "end"})


def _entry_cost(e: dict) -> float:
    spec = MODEL_CATALOG.get(e.get("model", ""), {})
    return (e.get("prompt_tokens", 0) * spec.get("cost_in", 0)
            + e.get("completion_tokens", 0) * spec.get("cost_out", 0)) / 1_000_000


@app.get("/api/run")
def api_run(prompt: str):
    return StreamingResponse(
        run_stream(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(os.path.join(_ROOT, "ui", "index.html"))


if __name__ == "__main__":
    print("\n  AI Model Portfolio Manager UI  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
