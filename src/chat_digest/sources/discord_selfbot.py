"""セルフボット(discord.py-self)による履歴取得。

⚠️ ユーザーアカウントの自動化は Discord の利用規約に違反し、アカウント停止の
リスクがあります。Bot を導入できないサーバー向けの非推奨手段であり、自己責任で
使用してください。discord.py-self は discord.py と同じ `discord` 名前空間を使うため、
両方を同時にインストールすることはできません。
"""

import os
import warnings

from chat_digest.preprocess import preprocess_discord_messages
from chat_digest.sources.base import FetchResult, MessageSource, UrlContext
from chat_digest.sources.discord_common import collect_history

DISCORD_URL_BASE = "https://discord.com/channels"


class DiscordSelfbotSource(MessageSource):
    def fetch(self, channel, *, hours, start_delta_hours=0, limit=7000):
        try:
            from discord.ext.commands import Bot
        except ImportError as exc:
            raise RuntimeError(
                "discord.py-self is not installed. Install with: pip install 'chat-digest[selfbot]'"
            ) from exc

        token = os.getenv(self.config.token_env, "")
        if not token:
            raise RuntimeError(f"Discord token env var '{self.config.token_env}' is not set")

        warnings.warn(
            "discord_selfbot source violates Discord's Terms of Service. Use at your own risk.",
            stacklevel=2,
        )
        client = Bot(command_prefix="$##", case_insensitive=True, self_bot=True)

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
