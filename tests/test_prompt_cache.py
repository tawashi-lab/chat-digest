import unittest

from chat_digest.llm.prompt_cache import (
    build_cached_topic_judge_messages,
    build_topic_judge_cacheable_text,
    build_topic_judge_prompt_payload,
    call_topic_judge_llm,
    model_supports_cache_control,
    split_topic_judge_user_prompt,
)


TOPIC_JUDGE_TEMPLATE = """
主に{topic_num_plusone}つのトピックについて話しているチャットのコメントについて、どのトピックについて話しているのかを判定してください。
主なトピックとその説明
"{topic_list}"

そのコメントの前のコメント:{former_chats}

以下のコメントについて、そのトピックを一つ判定してください。
トピックを判定するコメント:{target_chat}

また、判定した理由などは答えずに、判定した一つのtopicのindexのみを答えてください
コメントのtopic判定:
"""


class TopicPromptCacheTest(unittest.TestCase):
    def test_model_supports_cache_control_for_supported_gemini_versions(self):
        self.assertTrue(model_supports_cache_control("gemini-3-flash-preview"))
        self.assertTrue(model_supports_cache_control("gemini/gemini-3-flash-preview"))
        self.assertTrue(model_supports_cache_control("gemini-2.5-flash"))
        self.assertFalse(model_supports_cache_control("gemini-2.0-flash"))

    def test_split_reconstructs_existing_prompt_exactly(self):
        cached_prefix, suffix, full_prompt = split_topic_judge_user_prompt(
            TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\ntopic1の説明: Bitcoin\n",
            topic_num_plusone=2,
            former_chats="alice: 前の発言\n",
            target_chat="bob: 今の発言",
        )

        expected = TOPIC_JUDGE_TEMPLATE.format(
            topic_list="topic1: BTC\ntopic1の説明: Bitcoin\n",
            topic_num_plusone=2,
            former_chats="alice: 前の発言\n",
            target_chat="bob: 今の発言",
        )
        self.assertEqual(cached_prefix + suffix, expected)
        self.assertEqual(full_prompt, expected)
        self.assertIn("主なトピックとその説明", cached_prefix)
        self.assertTrue(suffix.startswith("alice: 前の発言\n"))

    def test_gemini_cache_enabled_moves_system_prompt_into_cached_prefix(self):
        full_prompt, messages = build_topic_judge_prompt_payload(
            model="gemini-3-flash-preview",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=True,
        )

        self.assertIsNotNone(messages)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        user_blocks = messages[0]["content"]
        self.assertEqual(user_blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertTrue(user_blocks[0]["text"].startswith("system\n\n"))
        self.assertEqual(
            user_blocks[0]["text"].removeprefix("system\n\n") + user_blocks[1]["text"],
            full_prompt,
        )

    def test_gemini_cacheable_text_matches_cached_message_prefix(self):
        cacheable_text = build_topic_judge_cacheable_text(
            model="gemini-3-flash-preview",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=True,
        )
        _, messages = build_topic_judge_prompt_payload(
            model="gemini-3-flash-preview",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=True,
        )

        self.assertIsNotNone(messages)
        self.assertEqual(cacheable_text, messages[0]["content"][0]["text"])
        self.assertTrue(cacheable_text.startswith("system\n\n"))

    def test_cacheable_text_is_none_when_cache_disabled(self):
        cacheable_text = build_topic_judge_cacheable_text(
            model="gemini-3-flash-preview",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=False,
        )

        self.assertIsNone(cacheable_text)

    def test_anthropic_cache_enabled_keeps_system_message(self):
        full_prompt, messages = build_topic_judge_prompt_payload(
            model="claude-sonnet-4-20250514",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=True,
        )

        self.assertIsNotNone(messages)
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        user_blocks = messages[1]["content"]
        self.assertEqual(user_blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(user_blocks[0]["text"] + user_blocks[1]["text"], full_prompt)

    def test_cache_disabled_uses_plain_prompt(self):
        full_prompt, messages = build_topic_judge_prompt_payload(
            model="gemini-3-flash-preview",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=False,
        )

        self.assertIsNone(messages)
        self.assertIn("トピックを判定するコメント:target", full_prompt)

    def test_openai_model_does_not_get_cache_control(self):
        _, messages = build_topic_judge_prompt_payload(
            model="gpt-5.4-hl",
            system_prompt="system",
            user_template=TOPIC_JUDGE_TEMPLATE,
            topic_list="topic1: BTC\n",
            topic_num_plusone=2,
            former_chats="prev\n",
            target_chat="target",
            use_prompt_cache=True,
        )

        self.assertIsNone(messages)

    def test_cache_call_failure_falls_back_to_plain_prompt(self):
        calls = []

        def call_llm_messages_fn(**kwargs):
            calls.append(("messages", kwargs))
            raise RuntimeError("cache failed")

        def call_llm_fn(**kwargs):
            calls.append(("plain", kwargs))
            return "1"

        result = call_topic_judge_llm(
            model="gemini-3-flash-preview",
            system_prompt="system",
            full_prompt="full prompt",
            cache_messages=build_cached_topic_judge_messages(
                model="gemini-3-flash-preview",
                system_prompt="system",
                cached_prefix="prefix",
                suffix="suffix",
            ),
            call_llm_fn=call_llm_fn,
            call_llm_messages_fn=call_llm_messages_fn,
        )

        self.assertEqual(result, "1")
        self.assertEqual([name for name, _ in calls], ["messages", "plain"])
        self.assertEqual(calls[1][1]["user_prompt"], "full prompt")

    def test_without_cache_messages_calls_plain_prompt_only(self):
        calls = []

        def call_llm_messages_fn(**kwargs):
            calls.append(("messages", kwargs))
            return "2"

        def call_llm_fn(**kwargs):
            calls.append(("plain", kwargs))
            return "1"

        result = call_topic_judge_llm(
            model="gemini-3-flash-preview",
            system_prompt="system",
            full_prompt="full prompt",
            cache_messages=None,
            call_llm_fn=call_llm_fn,
            call_llm_messages_fn=call_llm_messages_fn,
        )

        self.assertEqual(result, "1")
        self.assertEqual([name for name, _ in calls], ["plain"])

    def test_thinking_level_passed_to_cache_call(self):
        calls = []

        def call_llm_messages_fn(**kwargs):
            calls.append(("messages", kwargs))
            return "2"

        def call_llm_fn(**kwargs):
            calls.append(("plain", kwargs))
            return "1"

        result = call_topic_judge_llm(
            model="gemini-3-flash-preview",
            system_prompt="system",
            full_prompt="full prompt",
            cache_messages=build_cached_topic_judge_messages(
                model="gemini-3-flash-preview",
                system_prompt="system",
                cached_prefix="prefix",
                suffix="suffix",
            ),
            call_llm_fn=call_llm_fn,
            call_llm_messages_fn=call_llm_messages_fn,
            thinking_level="low",
        )

        self.assertEqual(result, "2")
        self.assertEqual([name for name, _ in calls], ["messages"])
        self.assertEqual(calls[0][1]["thinking_level"], "low")

    def test_thinking_level_passed_to_cache_fallback_call(self):
        calls = []

        def call_llm_messages_fn(**kwargs):
            calls.append(("messages", kwargs))
            raise RuntimeError("cache failed")

        def call_llm_fn(**kwargs):
            calls.append(("plain", kwargs))
            return "1"

        result = call_topic_judge_llm(
            model="gemini-3-flash-preview",
            system_prompt="system",
            full_prompt="full prompt",
            cache_messages=build_cached_topic_judge_messages(
                model="gemini-3-flash-preview",
                system_prompt="system",
                cached_prefix="prefix",
                suffix="suffix",
            ),
            call_llm_fn=call_llm_fn,
            call_llm_messages_fn=call_llm_messages_fn,
            thinking_level="low",
        )

        self.assertEqual(result, "1")
        self.assertEqual([name for name, _ in calls], ["messages", "plain"])
        self.assertEqual(calls[0][1]["thinking_level"], "low")
        self.assertEqual(calls[1][1]["thinking_level"], "low")


if __name__ == "__main__":
    unittest.main()
