# 🧠 AI Model Portfolio Manager

**Route every AI request to the *cheapest capable model* — automatically — and see the cost before and after.**

Instead of paying for one expensive frontier model (e.g. GPT‑4o) for *everything*, this project runs a
multi‑step pipeline that, for each request, picks the cheapest model that can still do the job — across
**7 models from 6 makers** (Google, OpenAI, Anthropic, Meta, Mistral, and open models on Groq). It shows the
**predicted cost up front**, the **actual cost** after running, and the **savings vs a single big model**
(typically **~95% cheaper**). It ships with a live, animated web UI.

> One sentence: *a smart router that reads how hard each request is, hires the cheapest AI that can handle it,
> learns from its mistakes, streams the whole thing to a glassy dashboard, and never locks you to one vendor.*

---

## Table of contents
1. [How it works (the 60‑second version)](#how-it-works-the-60-second-version)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup (step by step)](#setup-step-by-step)
6. [API keys — where to get them](#api-keys--where-to-get-them)
7. [Running the project](#running-the-project)
8. [The pipeline, node by node](#the-pipeline-node-by-node)
9. [The decision layer (how a model is chosen)](#the-decision-layer-how-a-model-is-chosen)
10. [The cost engine (Node A & Node F)](#the-cost-engine-node-a--node-f)
11. [The model portfolio](#the-model-portfolio)
12. [The web UI](#the-web-ui)
13. [Design decisions & honest notes](#design-decisions--honest-notes)
14. [Troubleshooting / FAQ](#troubleshooting--faq)

---

## How it works (the 60‑second version)

For every prompt, the request flows through a small **LangGraph** pipeline of 6 steps (A→F). Before any
model runs, a **cost estimate** is produced (Node A). Then a **decision layer** picks a model *per step*
based on the request's **intent** (what kind of task) and **difficulty** (how hard), always choosing the
**cheapest model that qualifies**. A **QA / review** step checks the answer; if it's not good enough, the
pipeline **replans with a different model** (up to 2 retries). Finally, a **cost auditor** (Node F) reports
the **actual cost** and the **savings** vs a single frontier model.

Key properties:
- **Provider‑neutral** — the router picks on capability + cost only, never by brand name.
- **Self‑improving** — every QA pass/fail updates the router's success stats, so it gets better over time.
- **Offline brains** — intent + difficulty run on a local embedding model (no API calls to decide routing).
- **Never crashes a demo** — if a provider errors, it **fails over live** to the next cheapest model.

---

## Architecture

```
                        ┌───────────────────────── Web UI (ui/index.html) ──────────────────────────┐
                        │  dark-navy glassy SPA · circular node graph · cost cards · chatbox          │
                        └───────────────▲───────────────────────────────┬────────────────────────────┘
                                        │ Server-Sent Events (live)      │ GET /api/run?prompt=...
                        ┌───────────────┴───────────────────────────────▼────────────────────────────┐
                        │                         FastAPI server (src/server.py)                      │
                        └───────────────────────────────────┬───────────────────────────────────────┘
                                                            │ streams node-by-node events
     ┌──────────────────────────────────────────────────────▼──────────────────────────────────────────┐
     │                        LangGraph pipeline (src/portfolio_manager.py)                              │
     │                                                                                                   │
     │   START ─▶ A: Pre-Flight ─▶ B: Triage ─┬─(simple)──────────────▶ D: Execution ─▶ E: QA ─┬─(pass)─▶ F: Ledger ─▶ END
     │           (estimate cost)  (understand) └─(complex)▶ C: Planner ─▶ D: Execution ─▶ E: QA ─┘ (fail, <2 retries)─▶ back to C
     │                                                                                                   │
     │   Nodes A & F  ──▶  Cost Engine (src/cost_engine)      estimate() / audit() / scorecard()         │
     │   Nodes B–E    ──▶  Decision Layer (src/router.py)     rank(layer, query) → cheapest capable      │
     │                 └─▶ LLM Caller (LiteLLM)               llm_call() with live provider failover     │
     └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Three components, integrated:**

| Component | Folder | Role |
|---|---|---|
| **Decision Layer** | `src/router.py`, `src/model_catalog.py` | Picks the model per step (semantic intent + difficulty + cost + learning). |
| **Cost Engine** | `src/cost_engine/` | Node A pre‑flight estimate (+ budget policy) and Node F post‑run audit (actual cost, waste, savings, self‑calibration). |
| **Model Portfolio** | `src/model_portfolio/` | LiteLLM wrapper + token ledger (the original per‑tier LLM caller). |
| **Orchestration** | `src/portfolio_manager.py` | Wires all three into one LangGraph pipeline (this is the main entry point). |

---

## Project structure

```
YAI/
├── src/                              # ALL application code
│   ├── portfolio_manager.py          # ⭐ the integrated pipeline (START→A→B→C→D→E→F→END)
│   ├── router.py                     # decision layer: rank()/choose() + learning loop + expected_route()
│   ├── model_catalog.py              # the 7-model registry (ids, cost, quality, tags)
│   ├── server.py                     # FastAPI + SSE backend for the UI
│   ├── cost_optimization_graph.py    # earlier standalone graph (reference; superseded by portfolio_manager)
│   ├── cost_engine/                  # Node A / Node F: pricing, estimator, ledger, report, governance, calibration
│   │   └── pricing.json              # model prices + roles (keys MUST match model_catalog.py)
│   └── model_portfolio/              # LiteLLM wrapper + LangGraph (original teammate module)
├── tests/
│   ├── live_test.py                  # detailed live end-to-end test (real API calls, 3 prompts)
│   └── test_litellm_client.py        # unit tests for the LiteLLM wrapper (no network)
├── integration_tests/                # smoke scripts for the LiteLLM wrapper
├── ui/
│   └── index.html                    # the single-page dashboard (React via CDN, zero build step)
├── .env                              # YOUR API keys (gitignored — never committed)
├── .env.example                      # template for .env
├── requirements.txt                  # pinned dependencies (use this)
├── pyproject.toml                    # package manifest for the model_portfolio module (teammate's)
└── README.md                         # you are here
```

---

## Prerequisites

- **Python 3.9+** (developed and verified on **3.9.6**).
  - ⚠️ If you are on **Python 3.9**, `semantic-router` **must be `0.0.72`** (pinned in `requirements.txt`).
    Newer `0.1.x` requires Python 3.10+ and will fail to import on 3.9. On Python 3.10+ you may use a newer version.
- **Internet access** at runtime — the LLM calls go out to the providers, and the UI loads React from a CDN.
- **~150 MB disk** for the local embedding model (downloaded once on first run, then cached offline).
- Three API keys (free tiers are fine): **Google/Gemini**, **OpenRouter**, **Groq** — see below.

---

## Setup (step by step)

```bash
# 1. Go into the project
cd YAI

# 2. (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Install all dependencies (pinned, verified)
pip install -r requirements.txt

# 4. Create your .env from the template and add your keys
cp .env.example .env               # then edit .env  (see next section)
```

That's it. The first run downloads the embedding model (~150 MB) once; subsequent runs are offline for routing.

---

## API keys — where to get them

Only **three** keys are used. Put them in `YAI/.env` (next to this README). The file is **gitignored**.

```dotenv
# YAI/.env
GEMINI_API_KEY=AIza...            # Google / Gemini   — used by model strings "gemini/..."
OPENROUTER_API_KEY=sk-or-v1-...   # OpenRouter        — used by model strings "openrouter/.../..."
GROQ_API_KEY=gsk_...              # Groq              — used by model strings "groq/..."
```

| Key | Where to get it | Expected prefix | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | `AIza…` | Free tier. **Not** the same as a Google Cloud OAuth token (`AQ.…` won't work). |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | `sk-or-v1-…` | One key serves OpenAI, Anthropic, Meta, Mistral models. Pay‑as‑you‑go. |
| `GROQ_API_KEY` | https://console.groq.com/keys | `gsk_…` | Free, very fast. **Groq ≠ Grok** — this is the inference company, not xAI's model. |

> **Common mix‑up:** `gsk_…` = **Groq** (this project). `xai-…` = **Grok/xAI** (not used here). They're different companies.

You don't strictly need all three — any missing/invalid key simply causes that provider's models to be
skipped and the router **fails over live** to another provider. But all three give the best variety.

---

## Running the project

All commands are run from the **`YAI/`** folder.

### 1. The web UI (recommended — this is the demo)
```bash
python src/server.py
# then open http://127.0.0.1:8000
```
Type a prompt (or click an example), and watch the node graph light up step‑by‑step with the model/tier used,
plus predicted vs actual cost and savings.

### 2. The CLI demo
```bash
python src/portfolio_manager.py
```
Runs two sample prompts through the full pipeline and prints the cost‑engine scorecard + a headline summary.

### 3. The detailed live test
```bash
python tests/live_test.py
```
Runs 3 varied prompts with **real** API calls and prints, per prompt and per layer: difficulty, intent, model,
provider, tokens, cost, LIVE‑vs‑mock, QA verdict, replans, and predicted‑vs‑actual‑vs‑savings.

### 4. Unit tests (no network)
```bash
python tests/test_litellm_client.py
```

---

## The pipeline, node by node

The graph lives in `src/portfolio_manager.py`. State is a `TypedDict`; `messages` and `token_ledger` are
append‑only (`Annotated[list, operator.add]`).

| Node | Name in code | UI label | What it does |
|---|---|---|---|
| **A** | `preflight` | **Estimate** | Pre‑flight cost estimate (cost engine), using the same decision layer that will actually run. Applies a budget policy (APPROVED / DOWNGRADED / BLOCKED). |
| **B** | `triage` | **Understand** | Classifies the request as **Simple** or **Complex** (an LLM call via the routed model). |
| **C** | `planner` | **Plan** | Produces a step‑by‑step plan (only for Complex requests). |
| **D** | `execution` | **Answer** | Does the actual work and produces the draft answer. |
| **E** | `qa` | **Review** | Judges whether the draft fully solves the request (PASS / FAIL). |
| **F** | `post_run` | **Final Bill** | Post‑run audit: actual cost, waste on retries, savings vs baselines, self‑calibration. |

**Routing / edges:**
- `START → A → B`
- `B →` **Simple**: straight to `D` (skip the planner)  ·  **Complex**: `C → D`
- `D → E`
- `E →` **PASS** (or retries exhausted): `F → END`  ·  **FAIL**: back to `C` (replan). A `max_retries` counter
  in state (default **2**) prevents infinite loops.

---

## The decision layer (how a model is chosen)

`src/router.py` exposes `ROUTER.rank(layer, query, state)` → a list of candidate models, **cheapest‑qualifying
first**. Each node tries them in order (live), so a provider outage fails over to the next real model.

For each request it computes:
1. **Intent** — via **`semantic-router`** (embedding classification) → required capability tags
   (code / reasoning / vision / long‑context). Runs locally.
2. **Difficulty** — a 0..1 score using **`fastembed`** local embeddings (the RouteLLM *technique*, run offline,
   with no vendor API call) → a required quality bar (capped so hard prompts still try a capable‑but‑cheap model).
3. **Filter** the catalog by capability tags, quality bar, and context size.
4. **Rank** survivors by **token‑weighted expected cost**, divided by each model's measured success rate
   (so models that keep failing QA get quietly down‑weighted). **No provider term anywhere** → vendor‑neutral.
5. **Learn** — `ROUTER.record_outcome(model, qa_passed)` is called after QA, closing the self‑improving loop.

The same function also powers the pre‑flight estimate: the cost engine calls `router.expected_route(query)`
so Node A estimates on the *same* plan the pipeline will actually run.

---

## The cost engine (Node A & Node F)

`src/cost_engine/` (teammate component) provides the real cost math. Prices live in `src/cost_engine/pricing.json`
— **its keys must match `src/model_catalog.py` exactly** so the token ledger prices cleanly.

- **Node A — `estimate()`**: predicts per‑layer tokens & cost, compares to frontier baselines (GPT‑4o, Claude,
  Gemini Pro), applies a per‑query budget policy, and reports estimated savings %.
- **Node F — `audit()` + `scorecard()`**: prices the real token ledger, flags **waste** (tokens spent on
  rejected/retried attempts), computes **actual savings** vs each baseline, and records estimate‑vs‑actual
  **variance**. The engine also **self‑calibrates** its token assumptions from each run's history.

> Baselines (GPT‑4o etc.) are **comparison‑only monoliths — never actually called.** They're the "what if you
> used one big model for everything" yardstick.

---

## The model portfolio

All runtime models are callable via the **three keys** (Gemini direct, Groq direct, everything else through
OpenRouter). Defined in `src/model_catalog.py` and priced in `src/cost_engine/pricing.json`.

| Model (UI name) | LiteLLM id | Provider | Reached via |
|---|---|---|---|
| Gemini 3.5 Flash | `gemini/gemini-3.5-flash` | Google | `GEMINI_API_KEY` |
| GPT‑OSS 20B | `groq/openai/gpt-oss-20b` | OpenAI (open) | `GROQ_API_KEY` |
| GPT‑OSS 120B | `groq/openai/gpt-oss-120b` | OpenAI (open) | `GROQ_API_KEY` |
| GPT‑4o mini | `openrouter/openai/gpt-4o-mini` | OpenAI | `OPENROUTER_API_KEY` |
| Mistral Small 3.2 | `openrouter/mistralai/mistral-small-3.2-24b-instruct` | Mistral | `OPENROUTER_API_KEY` |
| Llama 3.3 70B | `openrouter/meta-llama/llama-3.3-70b-instruct` | Meta | `OPENROUTER_API_KEY` |
| Claude Sonnet 4.5 | `openrouter/anthropic/claude-sonnet-4.5` | Anthropic | `OPENROUTER_API_KEY` |

> ⚠️ **Prices in `pricing.json` and `model_catalog.py` are seed estimates — verify against each provider's
> pricing page before relying on absolute dollar numbers.** The *relative* savings story is robust.

To add a model: add one entry to `model_catalog.py` **and** a matching entry (same key) in `pricing.json`.
No other code changes needed — the router adapts automatically.

---

## The web UI

- **Backend:** `src/server.py` (FastAPI) streams the live pipeline over **Server‑Sent Events** and serves the SPA.
- **Frontend:** `ui/index.html` — a single file, **React via CDN (no build step)**, dark‑navy glassy theme
  (Outfit font, azure `#0082FB`/`#0064E0` + mint `#2AFFCC` accents).
- **Features:** animated circular node graph (control "flows" step‑to‑step), each step shows its **tier**
  (Cheap/Medium/Costly) **and the actual model name**; Predicted / Actual / Savings cost cards with count‑up;
  the model the estimate assumed vs the model that actually answered; a chatbox; and the final answer below.

Because the UI is served by the backend, there's **nothing to build or deploy separately** — one command
(`python src/server.py`) runs the whole thing.

---

## Design decisions & honest notes

- **Why not the RouteLLM package?** We evaluated it and found its hosted router makes a **paid OpenAI embedding
  call for every routing decision** — which breaks offline use and bakes in vendor bias. So we kept RouteLLM's
  *technique* (embed the prompt, score difficulty) but run it on a **local** embedding model. `semantic-router`
  *is* a real dependency, used for intent.
- **Never mock when keys work.** `llm_call()` tries every ranked model live (with retries) before ever using
  mock data. Mock is an absolute last resort so the graph can't crash during a demo.
- **Estimate vs actual variance is expected.** The estimator uses conservative per‑layer token assumptions;
  it tightens over runs via self‑calibration (starts at `0 calibration runs`).
- **Python 3.9 patches.** The cost‑engine modules include `from __future__ import annotations` so their
  `X | None` type hints work on Python 3.9.

---

## Troubleshooting / FAQ

**`ModuleNotFoundError: No module named 'semantic_router'` / import errors on 3.9**
→ You likely installed a newer `semantic-router`. Run `pip install -r requirements.txt` (it pins `0.0.72`,
required for Python 3.9).

**Everything routes to one provider / I see "failing over to next provider"**
→ One of your keys is missing/invalid (or that provider is rate‑limited). The router fails over live to the
next cheapest model. Check `.env` with:
```bash
python3 -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print({k: len(os.getenv(k) or '') for k in ('GEMINI_API_KEY','OPENROUTER_API_KEY','GROQ_API_KEY')})"
```
You want non‑zero lengths. Prefixes should be `AIza…`, `sk-or…`, `gsk_…`.

**Gemini returns `NotFoundError: model ... no longer available`**
→ The model name is outdated for your account. `gemini/gemini-3.5-flash` is current; new Google accounts can't
use some older Gemini model names.

**The UI loads but is blank**
→ The SPA pulls React from a CDN, so it needs internet on first load (you already need it for the LLM calls).

**Port 8000 already in use**
→ Stop the old server: `lsof -ti:8000 | xargs kill -9` (macOS/Linux), then rerun `python src/server.py`.

**Where do costs/learning history get stored?**
→ In `YAI/.cost_engine_data/` (gitignored). Delete it to reset the self‑calibration history.

---

*Built for a hackathon: a provider‑neutral, cost‑optimizing, self‑improving multi‑agent router with a live dashboard.*
