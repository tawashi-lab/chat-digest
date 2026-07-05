"""公式 Bot API(discord.py)による履歴取得。

Bot を対象サーバーに招待し、Developer Portal で MESSAGE CONTENT INTENT を
有効にしておく必要がある。
"""

import os

from chat_digest.preprocess import preprocess_discord_messages
from chat_digest.sources.base import FetchResult, MessageSource, UrlContext
from chat_digest.sources.discord_common import collect_history

DISCORD_URL_BASE = "https://discord.com/channels"


class DiscordBotSource(MessageSource):
    def fetch(self, channel, *, hours, start_delta_hours=0, limit=7000):
        try:
            import discord
        except ImportError as exc:
            raise RuntimeError(
                "discord.py is not installed. Install with: pip install 'chat-digest[discord]'"
            ) from exc

        token = os.getenv(self.config.token_env, "")
        if not token:
            raise RuntimeError(f"Discord bot token env var '{self.config.token_env}' is not set")

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        raw = collect_history(
            client,
            token,
            int(self.resolve_channel(channel)),
            hours=hours,
            start_delta_hours=start_delta_hours,
            limit=limit,
        )
        effective = preprocess_discord_messages(
            raw,
            news_bot_authors=self.config.news_bot_authors,
            group_window_seconds=self.config.group_window_seconds,
        )
        return FetchResult(raw=raw, effective=effective)

    def url_context(self, channel):
        return UrlContext(
            url_base=DISCORD_URL_BASE,
            server_part=str(self.config.server_id or ""),
            channel_part=str(self.resolve_channel(channel)),
        )
