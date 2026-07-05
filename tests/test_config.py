import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from chat_digest.config import AppConfig, load_config
from chat_digest.prompts import (
    PromptProfileNotFoundError,
    list_bundled_profiles,
    load_prompt_profile,
)

MINIMAL_CONFIG = """
sources:
  myserver:
    type: discord_bot
    token_env: DISCORD_TOKEN_MYSERVER
    server_id: 1
    channels:
      general: 100
outputs:
  out1:
    url_env: WEBHOOK_OUT1
jobs:
  general-24h:
    source: myserver
    channel: general
    title: まとめ
    hours: 24
    prompt_profile: casual
    post_to: [out1]
"""


class ConfigTest(unittest.TestCase):
    def _load(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(textwrap.dedent(text), encoding="utf-8")
            return load_config(path)

    def test_minimal_config_loads_with_defaults(self):
        config = self._load(MINIMAL_CONFIG)
        job = config.jobs["general-24h"]
        self.assertEqual(job.hours, 24)
        self.assertEqual(job.message_limit, 7000)
        self.assertEqual(job.topics.base_num, 1)
        self.assertFalse(job.features.fact_check.enabled)
        self.assertTrue(job.features.references)
        self.assertEqual(config.llm.models.topic_judge, "gemini-3-flash-preview")

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            self._load(MINIMAL_CONFIG.replace("source: myserver", "source: nonexistent"))

    def test_unknown_channel_rejected(self):
        with self.assertRaises(ValueError):
            self._load(MINIMAL_CONFIG.replace("channel: general", "channel: nonexistent"))

    def test_unknown_output_rejected(self):
        with self.assertRaises(ValueError):
            self._load(MINIMAL_CONFIG.replace("post_to: [out1]", "post_to: [nonexistent]"))

    def test_discord_source_requires_token_env(self):
        with self.assertRaises(ValueError):
            self._load(MINIMAL_CONFIG.replace("\n    token_env: DISCORD_TOKEN_MYSERVER", ""))

    def test_job_model_override(self):
        config = self._load(
            MINIMAL_CONFIG.replace(
                "post_to: [out1]",
                "post_to: [out1]\n    models:\n      topic_summary: claude-opus-4-8",
            )
        )
        models = config.job_models("general-24h")
        self.assertEqual(models.topic_summary, "claude-opus-4-8")
        self.assertEqual(models.topic_judge, "gemini-3-flash-preview")

    def test_unknown_model_key_rejected(self):
        with self.assertRaises(ValueError):
            self._load(
                MINIMAL_CONFIG.replace(
                    "post_to: [out1]",
                    "post_to: [out1]\n    models:\n      not_a_step: foo",
                )
            )

    def test_output_resolve_url_from_env(self):
        config = self._load(MINIMAL_CONFIG)
        os.environ["WEBHOOK_OUT1"] = "https://discord.com/api/webhooks/x/y"
        try:
            self.assertEqual(
                config.outputs["out1"].resolve_url(),
                "https://discord.com/api/webhooks/x/y",
            )
        finally:
            del os.environ["WEBHOOK_OUT1"]
        with self.assertRaises(ValueError):
            config.outputs["out1"].resolve_url()


class PromptProfileTest(unittest.TestCase):
    def test_bundled_profiles_available(self):
        profiles = list_bundled_profiles()
        for name in ["general", "crypto", "stocks", "casual", "ai", "crypto_airdrop"]:
            self.assertIn(name, profiles)

    def test_load_bundled_profile_merges_general(self):
        prompts = load_prompt_profile("crypto")
        # プロファイル本体のキー
        self.assertIn("TOPIC_DETECT_USER_PROMPT", prompts)
        self.assertIn("{chats_str}", prompts["TOPIC_DETECT_USER_PROMPT"])
        # general 由来のキー
        self.assertIn("TITLE_USER", prompts)
        self.assertIn("EXTRACT_FEATURES", prompts)

    def test_local_dir_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "crypto.yaml").write_text(
                'TOPIC_DETECT_USER_PROMPT: "local override {chats_str} {topic_num}"\n',
                encoding="utf-8",
            )
            prompts = load_prompt_profile("crypto", extra_dirs=[tmp])
        self.assertTrue(prompts["TOPIC_DETECT_USER_PROMPT"].startswith("local override"))
        # general は同梱のものが引き続き読まれる
        self.assertIn("TITLE_USER", prompts)

    def test_overrides_win(self):
        prompts = load_prompt_profile("crypto", overrides={"TITLE_USER": "custom"})
        self.assertEqual(prompts["TITLE_USER"], "custom")

    def test_missing_profile_raises(self):
        with self.assertRaises(PromptProfileNotFoundError):
            load_prompt_profile("does-not-exist")


class ProductionConfigTest(unittest.TestCase):
    """本番 config.yaml が存在する環境でのみ検証(公開リポジトリには含まれない)。"""

    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    def test_production_config_valid(self):
        if not self.CONFIG_PATH.is_file():
            self.skipTest("production config not present")
        config = load_config(self.CONFIG_PATH)
        self.assertIn("tantore-24h", config.jobs)
        for job_name in config.jobs:
            job = config.jobs[job_name]
            prompts = load_prompt_profile(job.prompt_profile, extra_dirs=config.prompt_dirs)
            self.assertIn("TOPIC_JUDGE_USER_FORMAT", prompts, job_name)
            self.assertIn("TOPIC_SUMMARY_CHAT_USER_PROMPT", prompts, job_name)


if __name__ == "__main__":
    unittest.main()
