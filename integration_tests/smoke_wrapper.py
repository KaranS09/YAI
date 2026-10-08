from __future__ import annotations

import os

os.environ.setdefault("EXECUTION_MODEL", "groq/openai/gpt-oss-20b")

from model_portfolio import call_model


PROMPT = "Give me a concise 3 step plan to prepare for a hackathon demo."


def main() -> None:
    answer, state = call_model(
        prompt=PROMPT,
        model_tier="execution",
        state={},
        system_prompt="Answer concisely with exactly three numbered steps.",
        timeout=20,
    )

    print("PROMPT")
    print(PROMPT)
    print()

    print("ANSWER")
    print(answer)
    print()

    print("TOKEN TOTALS")
    print(state.get("token_totals", {}))
    print()

    print("LEDGER")
    for row in state.get("token_ledger", []):
        print(
            {
                "tier": row["tier"],
                "provider": row["provider"],
                "model": row["model"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
            }
        )


if __name__ == "__main__":
    main()
