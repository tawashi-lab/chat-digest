from chat_digest.config import SourceConfig
from chat_digest.sources.base import FetchResult, MessageSource, UrlContext


def create_source(name: str, config: SourceConfig) -> MessageSource:
    """設定から source 実装を生成する。実装モジュールは遅延 import(extras 依存のため)。"""
    if config.type == "discord_bot":
        from chat_digest.sources.discord_bot import DiscordBotSource

        return DiscordBotSource(name, config)
    if config.type == "discord_selfbot":
        from chat_digest.sources.discord_selfbot import DiscordSelfbotSource

        return DiscordSelfbotSource(name, config)
    if config.type == "telegram":
        from chat_digest.sources.telegram import TelegramSource

        return TelegramSource(name, config)
    raise ValueError(f"Unknown source type: {config.type}")


__all__ = ["FetchResult", "MessageSource", "UrlContext", "SourceConfig", "create_source"]
