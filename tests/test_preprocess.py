import os
import sys
import unittest
from pathlib import Path

import pandas as pd

from chat_digest.preprocess import (
    chat_df_to_str,
    get_topic_num,
    preprocess_discord_messages,
)
from chat_digest.sources.base import UrlContext

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_raw_df(rows):
    columns = [
        "id",
        "unixtime",
        "timestamp",
        "author",
        "content",
        "total_reaction_count",
        "attachments",
        "reference_id",
        "embed_title",
        "embed_description",
    ]
    return pd.DataFrame(rows, columns=columns)


def _row(msg_id, ts, author, content, **kwargs):
    return {
        "id": msg_id,
        "unixtime": ts,
        "timestamp": ts,
        "author": author,
        "content": content,
        "total_reaction_count": kwargs.get("total_reaction_count", 0),
        "attachments": kwargs.get("attachments"),
        "reference_id": kwargs.get("reference_id"),
        "embed_title": kwargs.get("embed_title"),
        "embed_description": kwargs.get("embed_description"),
    }


class PreprocessTest(unittest.TestCase):
    def test_consecutive_messages_grouped(self):
        df = _make_raw_df(
            [
                _row(1, "2026/07/01 10:00:00", "alice", "hello"),
                _row(2, "2026/07/01 10:00:30", "alice", "world"),
                _row(3, "2026/07/01 10:01:00", "bob", "hi"),
            ]
        )
        result = preprocess_discord_messages(df)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.iloc[0]["integrated_content"], "hello\nworld")
        self.assertEqual(result.iloc[1]["integrated_content"], "hi")

    def test_group_window_breaks_grouping(self):
        df = _make_raw_df(
            [
                _row(1, "2026/07/01 10:00:00", "alice", "hello"),
                _row(2, "2026/07/01 10:05:00", "alice", "later"),
            ]
        )
        result = preprocess_discord_messages(df, group_window_seconds=120)
        self.assertEqual(result.shape[0], 2)

    def test_url_replaced_and_html_stripped(self):
        df = _make_raw_df(
            [_row(1, "2026/07/01 10:00:00", "alice", "see https://example.com/x <b>bold</b>")]
        )
        result = preprocess_discord_messages(df)
        self.assertEqual(result.iloc[0]["integrated_content"], "see [URL] bold")

    def test_reference_becomes_quote(self):
        df = _make_raw_df(
            [
                _row(1, "2026/07/01 10:00:00", "alice", "original message"),
                _row(2, "2026/07/01 10:10:00", "bob", "reply text", reference_id=1),
            ]
        )
        result = preprocess_discord_messages(df)
        self.assertTrue(result.iloc[1]["integrated_content"].startswith("引用: original message"))

    def test_news_bot_requires_reaction(self):
        df = _make_raw_df(
            [
                _row(1, "2026/07/01 10:00:00", "newsbot", "", embed_description="ignored news"),
                _row(
                    2,
                    "2026/07/01 10:10:00",
                    "newsbot",
                    "",
                    embed_description="popular news",
                    total_reaction_count=3,
                ),
            ]
        )
        result = preprocess_discord_messages(df, news_bot_authors=["newsbot"])
        contents = list(result["integrated_content"])
        self.assertEqual(contents, ["popular news"])

    def test_get_topic_num_bounds(self):
        small = pd.DataFrame({"integrated_content": ["a"] * 10})
        large = pd.DataFrame({"integrated_content": ["a"] * 100000})
        self.assertEqual(get_topic_num(small, base_num=2, max_num=4, denominator=200), 2)
        self.assertEqual(get_topic_num(large, base_num=2, max_num=4, denominator=200), 4)

    def test_chat_df_to_str(self):
        df = pd.DataFrame({"integrated_content": ["a", "b"]})
        self.assertEqual(chat_df_to_str(df), "a\n\nb\n\n")


class UrlContextTest(unittest.TestCase):
    def test_discord_style(self):
        ctx = UrlContext("https://discord.com/channels", "1", "2")
        self.assertEqual(ctx.message_url(3), "https://discord.com/channels/1/2/3")

    def test_telegram_style_skips_empty_parts(self):
        ctx = UrlContext("https://t.me", "SomeChat", "")
        self.assertEqual(ctx.message_url(3), "https://t.me/SomeChat/3")


class PreprocessParityTest(unittest.TestCase):
    """旧 preprocess_tantore と新実装の出力一致を実ログで検証する。

    実ログの場所や bot 名は環境変数で渡す(未設定ならスキップ):
      CHAT_DIGEST_PARITY_LOG_GLOB   例: logs/MYSERVER/*/*/*/messages_*.pkl
      CHAT_DIGEST_PARITY_NEWS_BOTS  例: newsbot1,newsbot2
    """

    def test_matches_legacy_output_on_real_log(self):
        log_glob = os.getenv("CHAT_DIGEST_PARITY_LOG_GLOB", "")
        legacy_module = REPO_ROOT / "preprocess.py"
        log_files = sorted(REPO_ROOT.glob(log_glob)) if log_glob else []
        if not legacy_module.is_file() or not log_files:
            self.skipTest("legacy module or raw logs not present")

        sys.path.insert(0, str(REPO_ROOT))
        try:
            from preprocess import preprocess_tantore
        finally:
            sys.path.pop(0)

        news_bots = [
            s for s in os.getenv("CHAT_DIGEST_PARITY_NEWS_BOTS", "").split(",") if s
        ]
        raw_df = pd.read_pickle(log_files[-1])
        expected = preprocess_tantore(raw_df.copy())
        actual = preprocess_discord_messages(
            raw_df.copy(),
            news_bot_authors=news_bots,
            group_window_seconds=120,
        )
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True), actual.reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
