"""
Unified, provider-neutral model registry — the SINGLE source of truth.

Every model is keyed by its LiteLLM call string, so the SAME key flows through:
  * the decision layer (router.py reads quality / tags / cost to pick),
  * the LLM caller (the key IS the LiteLLM model string),
  * the cost engine (YAI/src/cost_engine/pricing.json uses identical keys).

PROVIDER SCOPE: every model here is callable with only THREE API keys —
  * Google / Gemini   -> 'gemini/...'
  * Groq               -> 'groq/...'
  * OpenRouter         -> 'openrouter/.../...'  (serves OpenAI, Anthropic, Meta, Mistral, ...)
All model strings below were confirmed live against these keys. Selection is
vendor-neutral — the router picks on capability + cost + measured success only.

Pricing/quality are SEED PRIORS (VERIFY before demo). cost_in/out are USD/1M.
Keep keys & prices in sync with pricing.json.
"""

from __future__ import annotations

# Capability tags a request may require:
#   code / reasoning / vision / long_context / tool / cheap

MODEL_CATALOG: dict[str, dict] = {
    # ---- Google / Gemini (direct, GEMINI_API_KEY) ----
    "gemini/gemini-3.5-flash": {
        "provider": "google", "cost_in": 0.10, "cost_out": 0.40, "context": 1_000_000,
        "quality": 0.82, "speed": 0.95,
        "tags": {"code", "reasoning", "vision", "tool", "long_context", "cheap"},
    },
    # ---- Groq (direct, GROQ_API_KEY) — fastest inference ----
    "groq/openai/gpt-oss-20b": {
        "provider": "groq", "cost_in": 0.10, "cost_out": 0.50, "context": 131_072,
        "quality": 0.68, "speed": 0.98, "tags": {"code", "tool", "cheap"},
    },
    "groq/openai/gpt-oss-120b": {
        "provider": "groq", "cost_in": 0.15, "cost_out": 0.60, "context": 131_072,
        "quality": 0.82, "speed": 0.92, "tags": {"code", "reasoning", "tool"},
    },
    # ---- Via OpenRouter (OPENROUTER_API_KEY) — many makers, one key ----
    "openrouter/openai/gpt-4o-mini": {
        "provider": "openai", "cost_in": 0.15, "cost_out": 0.60, "context": 128_000,
        "quality": 0.72, "speed": 0.90, "tags": {"code", "tool", "cheap"},
    },
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": {
        "provider": "mistral", "cost_in": 0.10, "cost_out": 0.30, "context": 128_000,
        "quality": 0.66, "speed": 0.88, "tags": {"code", "tool", "cheap"},
    },
    "openrouter/meta-llama/llama-3.3-70b-instruct": {
        "provider": "meta", "cost_in": 0.20, "cost_out": 0.40, "context": 131_072,
        "quality": 0.80, "speed": 0.75, "tags": {"code", "reasoning", "tool"},
    },
    "openrouter/anthropic/claude-sonnet-4.5": {
        "provider": "anthropic", "cost_in": 3.0, "cost_out": 15.0, "context": 200_000,
        "quality": 0.95, "speed": 0.55,
        "tags": {"code", "reasoning", "vision", "tool", "long_context"},
    },
}

AVAILABLE_MODELS: list[str] = list(MODEL_CATALOG.keys())
