import unittest
from unittest.mock import patch

import pandas as pd

from chat_digest.analysis.topics import get_chats_topic


PROMPT_DICT = {
    "TOPIC_JUDGE_SYS_FORMAT": "system",
    "TOPIC_JUDGE_USER_FORMAT": (
        "topics:{topic_list}\n"
        "former:{former_chats}\n"
        "target:{target_chat}\n"
        "count:{topic_num_plusone}"
    ),
}
TOPIC_LIST = [["BTC", "Bitcoin topic"]]


class GetChatsTopicCacheTest(unittest.TestCase):
    def _run_get_chats_topic(
        self,
        model: str,
        *,
        token_count: int = 1024,
        token_count_side_effect=None,
        chat_rows: list[str] | None = None,
    ) -> tuple[list[dict], object]:
        calls = []

        def fake_fetch_chat_topic(**kwargs):
            calls.append(kwargs)
            return 1

        if chat_rows is None:
            chat_rows = ["alice: BTC の話"]
        chat_df = pd.DataFrame(
            {
                "timestamp": [
                    f"2026-05-17T00:00:0{i}"
                    for i, _ in enumerate(chat_rows)
                ],
                "integrated_content": chat_rows,
            }
        )

        count_patch = patch(
            "chat_digest.analysis.topics.count_gemini_tokens",
            return_value=token_count,
            side_effect=token_count_side_effect,
        )
        with (
            patch("chat_digest.analysis.topics.fetch_chat_topic", side_effect=fake_fetch_chat_topic),
            count_patch as count_mock,
        ):
            tagged_df, topic_id_df = get_chats_topic(
                chat_df=chat_df,
                topic_list=TOPIC_LIST,
                topic_num=1,
                prompt_dict=PROMPT_DICT,
                model=model,
                former_chat_num=1,
                max_workers=1,
                use_prompt_cache=True,
            )

        self.assertEqual(tagged_df.loc[0, "topic_id"], 1)
        self.assertEqual(topic_id_df.iloc[0]["topic"], "BTC")
        return calls, count_mock

    def test_gemini_3_prompt_cache_is_disabled(self):
        # Gemini は LiteLLM の cachedContents リクエスト爆発を避けるため明示キャッシュを
        # 使わない。トークンカウント(キャッシュ可否判定)も呼ばれない
        calls, count_mock = self._run_get_chats_topic("gemini-3-flash-preview")

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["use_prompt_cache"])
        count_mock.assert_not_called()

    def test_gemini_3_multiple_chats_all_plain(self):
        calls, count_mock = self._run_get_chats_topic(
            "gemini-3-flash-preview",
            chat_rows=["alice: BTC の話", "bob: ETH の話", "carol: 相場の話"],
        )

        self.assertEqual(len(calls), 3)
        self.assertFalse(any(call["use_prompt_cache"] for call in calls))
        count_mock.assert_not_called()

    def test_gemini_20_prompt_cache_is_not_enabled(self):
        calls, count_mock = self._run_get_chats_topic("gemini-2.0-flash")

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["use_prompt_cache"])
        count_mock.assert_not_called()

    def test_anthropic_prompt_cache_stays_enabled(self):
        calls, count_mock = self._run_get_chats_topic("claude-sonnet-4-20250514")

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["use_prompt_cache"])
        count_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
