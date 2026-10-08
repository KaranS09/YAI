import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_portfolio import litellm_client


def fake_response(prompt_tokens=12, completion_tokens=7, content="done"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class LiteLLMClientTests(unittest.TestCase):
    def setUp(self):
        self.original_completion = litellm_client.completion
        self.original_load_dotenv = litellm_client.load_dotenv
        self.original_groq_key = os.environ.get("GROQ_API_KEY")
        self.original_openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        litellm_client.load_dotenv = lambda: False

    def tearDown(self):
        litellm_client.completion = self.original_completion
        litellm_client.load_dotenv = self.original_load_dotenv
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        if self.original_groq_key is not None:
            os.environ["GROQ_API_KEY"] = self.original_groq_key
        if self.original_openrouter_key is not None:
            os.environ["OPENROUTER_API_KEY"] = self.original_openrouter_key

    def test_call_model_updates_token_ledger(self):
        os.environ["GROQ_API_KEY"] = "test-key"
        litellm_client.completion = lambda **kwargs: fake_response()

        content, state = litellm_client.call_model("Classify this.", "triage", {})

        self.assertEqual(content, "done")
        self.assertEqual(
            state["token_totals"],
            {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        )
        self.assertEqual(state["token_ledger"][0]["tier"], "triage")
        self.assertEqual(state["token_ledger"][0]["provider"], "groq")

    def test_call_model_accumulates_existing_totals(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        litellm_client.completion = lambda **kwargs: fake_response(3, 5, "ok")
        existing_state = {
            "token_totals": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "token_ledger": [],
        }

        _, state = litellm_client.call_model("Check answer.", "qa", existing_state)

        self.assertEqual(
            state["token_totals"],
            {"prompt_tokens": 13, "completion_tokens": 25, "total_tokens": 38},
        )

    def test_missing_provider_key_has_clear_error(self):
        os.environ.pop("GROQ_API_KEY", None)

        with self.assertRaisesRegex(RuntimeError, "Missing GROQ_API_KEY"):
            litellm_client.call_model("hello", "triage", {})


if __name__ == "__main__":
    unittest.main()
