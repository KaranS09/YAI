# Free Model Portfolio

LiteLLM integration for the hackathon graph. It exposes one wrapper:

```python
from model_portfolio import call_model

answer, state = call_model(
    prompt="Classify this request as Simple or Complex.",
    model_tier="triage",
    state={},
)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill in `.env`:

```dotenv
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

LiteLLM reads both keys from environment variables. Groq models use the `groq/...`
prefix and OpenRouter models use the `openrouter/...` prefix.

## Model tiers

| Tier | Provider | Model |
| --- | --- | --- |
| `triage` | Groq | `groq/openai/gpt-oss-20b` |
| `planner` | OpenRouter | `openrouter/apodex/apodex-1.1-mini:free` |
| `execution` | Groq | `groq/qwen/qwen3.8-27b` |
| `qa` | OpenRouter | `openrouter/nvidia/nemotron-3.5-lightning:free` |

Override the execution model without code changes:

```powershell
$env:EXECUTION_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
```

## State updates

Every call appends a row to `state["token_ledger"]` and updates
`state["token_totals"]`:

```python
{
    "token_ledger": [
        {
            "tier": "triage",
            "model": "groq/llama-3.1-8b-instant",
            "provider": "groq",
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "estimated_prompt_tokens": 10,
        }
    ],
    "token_totals": {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    },
    "last_model_response": "Simple",
}
```

## LangGraph integration

```python
from model_portfolio.graph import run_portfolio_graph

state = run_portfolio_graph("Summarize the tradeoffs of Groq vs OpenRouter.")
print(state["final_answer"])
print(state["token_totals"])
```

Or compile the graph yourself:

```python
from model_portfolio.graph import build_graph

app = build_graph()
state = app.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```
