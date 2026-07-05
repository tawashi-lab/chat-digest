"""YAML 設定のロードとバリデーション。

設定は「sources(取得元)」「outputs(投稿先 Webhook)」「jobs(1回の要約実行の
プリセット)」の3層。シークレットは YAML に直接書かず、環境変数名(*_env)で参照する。
環境変数は CLI 起動時に .env からロードされる。
"""

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelSettings(BaseModel):
    """処理ステップごとの使用モデル。"""

    topic_detect: str = "gpt-5.5-hl"
    topic_vote: str = "gpt-5.5-hl"
    topic_judge: str = "gemini-3-flash-preview"
    topic_summary: str = "gpt-5.5-hl"
    condense: str = "gpt-5.5-hl"
    topic_element: str = "gpt-5.5-hl"
    highlight: str = "gpt-5.5-hl"
    title: str = "gpt-5.5-hl"
    fact_check_item: str = "gpt-5.5-hm"
    fact_check_rewrite: str = "gpt-5.5-hm"
    grok: str = "grok-4-fast-reasoning"
    embedding: str = "gemini-embedding-2"
    extract_phrases: str = "gpt-5.4-hl"
    judge_reference: str = "gpt-5.4-hl"


class LLMSettings(BaseModel):
    models: ModelSettings = Field(default_factory=ModelSettings)
    max_retries: int = 3
    timeout_seconds: int = 120
    topic_judge_thinking_level: Optional[str] = "minimal"
    use_prompt_cache: bool = True


class SourceConfig(BaseModel):
    type: Literal["discord_bot", "discord_selfbot", "telegram"]
    # Discord 用
    token_env: str = ""
    server_id: Optional[int] = None
    # channels: 論理名 -> チャンネルID(Discord)またはチャット名(Telegram)
    channels: dict[str, int | str] = Field(default_factory=dict)
    # Telegram 用
    api_id_env: str = "TELEGRAM_API_ID"
    api_hash_env: str = "TELEGRAM_API_HASH"
    session_name: str = "chat_digest_telegram"
    # 前処理オプション
    # ニュース速報 bot 等: リアクションが付いたメッセージのみ採用(Discord)
    news_bot_authors: list[str] = Field(default_factory=list)
    # 無視する発言者 ID(Telegram の welcome bot 等)
    skip_author_ids: list[int | str] = Field(default_factory=list)
    # 同一発言者の連投を1メッセージに束ねる時間窓(秒、Discord)
    group_window_seconds: int = 120

    @model_validator(mode="after")
    def _check_required(self) -> "SourceConfig":
        if self.type in ("discord_bot", "discord_selfbot") and not self.token_env:
            raise ValueError(f"source type '{self.type}' requires 'token_env'")
        return self


class BrandingConfig(BaseModel):
    """Webhook 投稿時の表示設定。"""

    username: str = "chat-digest"
    avatar_url: str = ""
    footer: str = ""


class OutputConfig(BaseModel):
    type: Literal["discord_webhook"] = "discord_webhook"
    url_env: str

    def resolve_url(self) -> str:
        url = os.getenv(self.url_env, "")
        if not url:
            raise ValueError(f"Webhook URL env var '{self.url_env}' is not set")
        return url


class TopicSettings(BaseModel):
    base_num: int = 1
    max_num: int = 4
    denominator: int = 150
    candidate_num: int = 2
    vote_attempts: int = 1
    judge_former_chat_num: int = 2
    judge_max_workers: int = 50
    cut_others: bool = False


class CondenseSettings(BaseModel):
    enabled: bool = False
    max_chars: int = 200


class FactCheckSettings(BaseModel):
    enabled: bool = False
    max_items: int = 5


class FeatureSettings(BaseModel):
    condense: CondenseSettings = Field(default_factory=CondenseSettings)
    fact_check: FactCheckSettings = Field(default_factory=FactCheckSettings)
    # 全体の50字ヘッドラインを生成するか
    highlight: bool = False
    # トピック推移グラフ画像を添付するか
    graph: bool = False
    # 要約中の特徴フレーズに元チャットへの参照リンクを埋め込むか
    references: bool = True
    # Grok 検索でトピックの補足情報(銘柄の値動き等)を取得するか
    topic_context: bool = False


class JobConfig(BaseModel):
    source: str
    channel: str
    title: str
    hours: int
    start_delta_hours: int = 0
    message_limit: int = 7000
    # 前処理後のメッセージ数がこの値以下なら要約をスキップ
    cutoff_num: int = 0
    embed_color: int = 36625
    description_off: bool = False
    url_off: bool = False
    topics: TopicSettings = Field(default_factory=TopicSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    prompt_profile: str
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    post_to: list[str] = Field(min_length=1)
    simple_post_to: list[str] = Field(default_factory=list)
    # ステップ別モデルの job 単位上書き(キーは ModelSettings のフィールド名)
    models: dict[str, str] = Field(default_factory=dict)


class AppConfig(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    branding: BrandingConfig = Field(default_factory=BrandingConfig)
    # ローカルプロンプトプロファイルの検索ディレクトリ(config ファイルからの相対も可)
    prompt_dirs: list[str] = Field(default_factory=list)
    log_dir: str = "logs"
    sources: dict[str, SourceConfig]
    outputs: dict[str, OutputConfig]
    jobs: dict[str, JobConfig]

    @model_validator(mode="after")
    def _check_references(self) -> "AppConfig":
        for job_name, job in self.jobs.items():
            if job.source not in self.sources:
                raise ValueError(f"job '{job_name}': unknown source '{job.source}'")
            source = self.sources[job.source]
            if job.channel not in source.channels:
                raise ValueError(
                    f"job '{job_name}': channel '{job.channel}' not defined in source '{job.source}'"
                )
            for out in [*job.post_to, *job.simple_post_to]:
                if out not in self.outputs:
                    raise ValueError(f"job '{job_name}': unknown output '{out}'")
            unknown_models = set(job.models) - set(ModelSettings.model_fields)
            if unknown_models:
                raise ValueError(f"job '{job_name}': unknown model keys {sorted(unknown_models)}")
        return self

    def job_models(self, job_name: str) -> ModelSettings:
        """グローバル設定に job 単位の上書きを適用したモデル設定を返す。"""
        job = self.jobs[job_name]
        return self.llm.models.model_copy(update=job.models)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    config = AppConfig.model_validate(data)
    # prompt_dirs は config ファイルの場所からの相対パスとして解決
    config.prompt_dirs = [
        str((path.parent / d).resolve()) if not Path(d).is_absolute() else d
        for d in config.prompt_dirs
    ]
    return config
