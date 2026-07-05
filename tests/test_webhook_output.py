import json
import unittest

import pandas as pd

from chat_digest.output.webhook import (
    FIELD_CHAR_LIMIT,
    make_embed_payload,
    make_summary_dicts,
    split_summary_text,
)
from chat_digest.pipeline import _build_simple_fields


class SplitSummaryTextTest(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(split_summary_text("短い要約。"), ["短い要約。"])

    def test_long_text_split_within_limit(self):
        text = "。".join(["あ" * 100] * 20) + "。"
        parts = split_summary_text(text)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), FIELD_CHAR_LIMIT)

    def test_markdown_link_not_broken(self):
        link = "[フレーズ](https://discord.com/channels/1/2/3)"
        text = "a" * (FIELD_CHAR_LIMIT - 10) + link + "b" * 50
        parts = split_summary_text(text)
        joined = "".join(parts)
        self.assertIn(link, joined)
        for part in parts:
            # リンクが分断されていれば ]( が孤立する
            self.assertEqual(part.count("["), part.count("]"))


class MakeSummaryDictsTest(unittest.TestCase):
    def _chat_df(self):
        return pd.DataFrame({"topic_id": [1, 1, 1, 2, 3]})

    def test_percentage_and_titles(self):
        dicts = make_summary_dicts(
            topic_summary_list=["s1", "s2", "s-other"],
            topic_list=[["Topic A", "desc"], ["Topic B", "desc"]],
            chat_df=self._chat_df(),
            topic_num=2,
            urls_list=[[], []],
            cut_others=False,
        )
        self.assertEqual(len(dicts), 3)
        # 「その他」(topic_id=3) は割合の分母に含まれない
        self.assertEqual(dicts[0]["title"], "Topic A: 75%")
        self.assertEqual(dicts[0]["percentage"], 75)
        self.assertEqual(dicts[2]["title"], "その他")
        self.assertEqual(dicts[2]["percentage"], 0)

    def test_cut_others_drops_last_summary(self):
        dicts = make_summary_dicts(
            topic_summary_list=["s1", "s2", "s-other"],
            topic_list=[["Topic A", "desc"], ["Topic B", "desc"]],
            chat_df=self._chat_df(),
            topic_num=2,
            urls_list=[[], []],
            cut_others=True,
        )
        self.assertEqual(len(dicts), 2)
        self.assertEqual([d["title"] for d in dicts], ["Topic A: 75%", "Topic B: 25%"])


class EmbedPayloadTest(unittest.TestCase):
    def test_branding_and_fields(self):
        summaries = [
            {
                "title": "Topic A: 60%",
                "description": "本文",
                "description_parts": ["本文"],
                "urls": [],
                "percentage": 60,
            }
        ]
        payload = make_embed_payload(
            summaries,
            "テストまとめ",
            "",
            description_off=False,
            url_off=False,
            highlight_all="ハイライト",
            username="AIまとめ",
            avatar_url="https://example.com/a.png",
            footer="フッター",
        )
        data = json.loads(payload["payload_json"])
        self.assertEqual(data["username"], "AIまとめ")
        self.assertEqual(data["avatar_url"], "https://example.com/a.png")
        embed = data["embeds"][0]
        self.assertEqual(embed["title"], "テストまとめ")
        self.assertEqual(embed["footer"]["text"], "フッター")
        self.assertEqual(embed["fields"][0]["name"], "ワンポイント")
        self.assertEqual(embed["fields"][1]["name"], "Topic A: 60%")
        self.assertTrue(embed["fields"][1]["value"].startswith("> 本文"))

    def test_no_avatar_or_footer_when_empty(self):
        payload = make_embed_payload(
            [], "t", "", description_off=False, url_off=False
        )
        data = json.loads(payload["payload_json"])
        self.assertNotIn("avatar_url", data)
        self.assertNotIn("footer", data["embeds"][0])


class SimpleFieldsTest(unittest.TestCase):
    def test_headlines_sorted_by_percentage(self):
        summary_dicts = [
            {"title": "A: 10%", "percentage": 10},
            {"title": "B: 60%", "percentage": 60},
        ]
        fields = _build_simple_fields(
            summary_dicts,
            [["A", "d"], ["B", "d"]],
            "全体ハイライト",
            "https://discord.com/channels/1/2/3",
        )
        self.assertEqual(fields[0]["name"], "ワンポイント")
        self.assertEqual(fields[1]["name"], "トピック")
        self.assertEqual(fields[1]["value"], "- B (60%)\n- A (10%)")
        self.assertEqual(fields[2]["name"], "本編リンク")


if __name__ == "__main__":
    unittest.main()
