from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, MutableMapping, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps local tests usable before setup.
    def load_dotenv() -> bool:
        return False

try:
    from litellm import completion
except ImportError:  # pragma: no cover - call_model raises a clearer runtime error.
    completion = None

try:
    import tiktoken
except ImportError:  # pragma: no cover - pyproject installs this, but keep fallback humane.
    tiktoken = None


ModelTier = Literal["triage", "planner", "execution", "qa"]


class TokenTotals(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenLedgerEntry(TypedDict):
    tier: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_prompt_tokens: int


class GraphState(TypedDict, total=False):
    messages: list[dict[str, str]]
    current_plan: list[str]
    token_ledger: list[TokenLedgerEntry]
    token_totals: TokenTotals
    estimated_cost: float
    actual_cost: float
    last_model_response: str


@dataclass(frozen=True)
class ModelConfig:
    model: str
    provider: str
    temperature: float = 0.2
    max_tokens: int = 1024


MODEL_TIERS: dict[ModelTier, ModelConfig] = {
    "triage": ModelConfig(
        model=os.getenv("TRIAGE_MODEL", "groq/openai/gpt-oss-20b"),
        provider="groq",
        temperature=0.0,
        max_tokens=256,
    ),
    "planner": ModelConfig(
        model="openrouter/apodex/apodex-1.1-mini:free",
        provider="openrouter",
        temperature=0.2,
        max_tokens=1024,
    ),
    "execution": ModelConfig(
        model=os.getenv("EXECUTION_MODEL", "groq/qwen/qwen3.8-27b"),
        provider="groq",
        temperature=0.2,
        max_tokens=2048,
    ),
    "qa": ModelConfig(
        model="openrouter/nvidia/nemotron-3.5-lightning:free",
        provider="openrouter",
        temperature=0.0,
        max_tokens=512,
    ),
}


def call_model(
    prompt: str,
    model_tier: ModelTier,
    state: MutableMapping[str, Any],
    *,
    system_prompt: str | None = None,
    extra_messages: list[dict[str, str]] | None = None,
    **completion_kwargs: Any,
) -> tuple[str, GraphState]:
    """Call the configured LiteLLM model tier and append token usage to graph state."""
    load_dotenv()
    _validate_provider_key(model_tier)
    if completion is None:
        raise RuntimeError('Missing dependency: install LiteLLM with `pip install -e ".[dev]"`.')

    config = MODEL_TIERS[model_tier]
    messages = _build_messages(prompt, system_prompt, extra_messages)
    estimated_prompt_tokens = estimate_messages_tokens(messages, config.model)

    response = completion(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=completion_kwargs.pop("timeout", 30),
        **_openrouter_headers(config.provider),
        **completion_kwargs,
    )

    content = response.choices[0].message.content or ""
    usage = _extract_usage(response)
    updated_state = _append_usage(
        state=state,
        tier=model_tier,
        model=config.model,
        provider=config.provider,
        usage=usage,
        estimated_prompt_tokens=estimated_prompt_tokens,
        content=content,
    )
    return content, updated_state


def estimate_messages_tokens(messages: list[dict[str, str]], model: str) -> int:
    """Estimate prompt size before a request; provider usage remains authoritative after."""
    if tiktoken is None:
        return sum(max(1, len(message.get("content", "")) // 4) for message in messages)

    try:
        encoding = tiktoken.encoding_for_model(model)
        return sum(len(encoding.encode(message.get("content", ""))) + 4 for message in messages)
    except Exception:
        return _estimate_by_chars(messages)


def _estimate_by_chars(messages: list[dict[str, str]]) -> int:
    return sum(max(1, len(message.get("content", "")) // 4) + 4 for message in messages)


def _build_messages(
    prompt: str,
    system_prompt: str | None,
    extra_messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_usage(response: Any) -> TokenTotals:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _append_usage(
    *,
    state: MutableMapping[str, Any],
    tier: str,
    model: str,
    provider: str,
    usage: TokenTotals,
    estimated_prompt_tokens: int,
    content: str,
) -> GraphState:
    updated: GraphState = deepcopy(dict(state))
    ledger = list(updated.get("token_ledger", []))
    ledger.append(
        {
            "tier": tier,
            "model": model,
            "provider": provider,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_prompt_tokens": estimated_prompt_tokens,
        }
    )

    totals = updated.get(
        "token_totals",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    updated["token_ledger"] = ledger
    updated["token_totals"] = {
        "prompt_tokens": int(totals.get("prompt_tokens", 0)) + usage["prompt_tokens"],
        "completion_tokens": int(totals.get("completion_tokens", 0)) + usage["completion_tokens"],
        "total_tokens": int(totals.get("total_tokens", 0)) + usage["total_tokens"],
    }
    updated["last_model_response"] = content
    return updated


def _validate_provider_key(model_tier: ModelTier) -> None:
    provider = MODEL_TIERS[model_tier].provider
    env_var = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[provider]
    if not os.getenv(env_var):
        raise RuntimeError(f"Missing {env_var}. Add it to your environment or a local .env file.")


def _openrouter_headers(provider: str) -> dict[str, Any]:
    if provider != "openrouter":
        return {}

    headers: dict[str, str] = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    if os.getenv("OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]
    return {"extra_headers": headers} if headers else {}
