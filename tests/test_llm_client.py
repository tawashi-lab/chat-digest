import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from chat_digest.llm.client import (
    LLMCallError,
    _build_request_kwargs,
    _build_responses_request_kwargs,
    count_gemini_tokens,
    resolve_model,
)


MESSAGES = [{"role": "user", "content": "hello"}]


def build_request_kwargs(
    model: str,
    *,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
) -> dict:
    return _build_request_kwargs(
        resolved_model=resolve_model(model),
        messages=MESSAGES,
        temperature=0.0,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        reasoning_effort=None,
        verbosity=None,
        service_tier=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
    )


class LiteLLMRequestKwargsTest(unittest.TestCase):
    def test_gpt_55_alias_preserves_reasoning_and_verbosity(self):
        resolved = resolve_model("gpt-5.5-hl")

        self.assertEqual(resolved.provider, "openai")
        self.assertEqual(resolved.model_name, "gpt-5.5")
        self.assertEqual(resolved.request_model, "openai/gpt-5.5")
        self.assertEqual(resolved.reasoning_effort, "high")
        self.assertEqual(resolved.verbosity, "low")

    def test_gpt_55_responses_kwargs_keep_existing_alias_parameters(self):
        kwargs = _build_responses_request_kwargs(
            resolved_model=resolve_model("gpt-5.5-hm"),
            messages=MESSAGES,
            reasoning_effort=None,
            verbosity=None,
            service_tier=None,
            prompt_cache_key=None,
            prompt_cache_retention=None,
        )

        self.assertEqual(kwargs["model"], "openai/gpt-5.5")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["text"], {"verbosity": "medium"})

    def test_gemini_3_converts_thinking_level_to_thinking_config(self):
        kwargs = build_request_kwargs(
            "gemini-3-flash-preview",
            thinking_budget=0,
            thinking_level="low",
        )

        self.assertEqual(kwargs["model"], "gemini/gemini-3-flash-preview")
        self.assertEqual(kwargs["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertNotIn("thinking_level", kwargs)
        self.assertNotIn("thinking_budget", kwargs)

    def test_gemini_3_converts_minimal_thinking_level_to_thinking_config(self):
        kwargs = build_request_kwargs(
            "gemini-3-flash-preview",
            thinking_budget=0,
            thinking_level="minimal",
        )

        self.assertEqual(kwargs["model"], "gemini/gemini-3-flash-preview")
        self.assertEqual(kwargs["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertNotIn("thinking_level", kwargs)
        self.assertNotIn("thinking_budget", kwargs)

    def test_gemini_25_converts_thinking_budget_to_thinking_config(self):
        kwargs = build_request_kwargs(
            "gemini-2.5-flash",
            thinking_budget=0,
            thinking_level="low",
        )

        self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(kwargs["thinkingConfig"], {"thinkingBudget": 0})
        self.assertNotIn("thinking_budget", kwargs)
        self.assertNotIn("thinking_level", kwargs)

    def test_count_gemini_tokens_uses_google_genai_client(self):
        clients = []

        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.models = Mock()
                self.models.count_tokens.return_value = SimpleNamespace(total_tokens=1057)
                clients.append(self)

        fake_genai = SimpleNamespace(Client=FakeClient)

        with patch("chat_digest.llm.client._load_google_genai", return_value=fake_genai), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key"},
        ):
            tokens = count_gemini_tokens("gemini-3-flash-preview", "cacheable text")

        self.assertEqual(tokens, 1057)
        self.assertEqual(clients[0].api_key, "test-key")
        clients[0].models.count_tokens.assert_called_once_with(
            model="gemini-3-flash-preview",
            contents="cacheable text",
        )

    def test_count_gemini_tokens_wraps_sdk_errors(self):
        class FakeClient:
            def __init__(self, api_key):
                self.models = Mock()
                self.models.count_tokens.side_effect = RuntimeError("boom")

        fake_genai = SimpleNamespace(Client=FakeClient)

        with patch("chat_digest.llm.client._load_google_genai", return_value=fake_genai), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key"},
        ):
            with self.assertRaises(LLMCallError):
                count_gemini_tokens("gemini-3-flash-preview", "cacheable text")

    def test_count_gemini_tokens_requires_total_tokens(self):
        class FakeClient:
            def __init__(self, api_key):
                self.models = Mock()
                self.models.count_tokens.return_value = SimpleNamespace()

        fake_genai = SimpleNamespace(Client=FakeClient)

        with patch("chat_digest.llm.client._load_google_genai", return_value=fake_genai), patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key"},
        ):
            with self.assertRaises(LLMCallError):
                count_gemini_tokens("gemini-3-flash-preview", "cacheable text")


if __name__ == "__main__":
    unittest.main()
