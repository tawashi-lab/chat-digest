"""Telethon による Telegram チャンネルの履歴取得。

初回実行時は Telethon が電話番号認証を対話的に求める(セッションファイルに保存され、
以降は不要)。cron 等での運用前に一度手動実行しておくこと。
"""

import asyncio
import datetime
import os

import pandas as pd

from chat_digest.sources.base import FetchResult, MessageSource, UrlContext

TELEGRAM_URL_BASE = "https://t.me"
_FETCH_CHUNK = 500


async def _collect_messages(
    session_name: str,
    api_id: int,
    api_hash: str,
    chat_name: str,
    *,
    cutoff_time: datetime.datetime,
    upper_time: datetime.datetime,
    limit: int,
):
    from telethon import TelegramClient

    messages = []
    async with TelegramClient(session_name, api_id, api_hash) as client:
        chat_info = await client.get_entity(chat_name)
        max_id = None

        while True:
            kwargs = {"entity": chat_info, "limit": _FETCH_CHUNK, "wait_time": 1}
            if max_id is not None:
                kwargs["max_id"] = max_id
            fetched_messages = await client.get_messages(**kwargs)
            if not fetched_messages:
                break

            for message in fetched_messages:
                if message.date > upper_time:
                    continue
                if (message.date < cutoff_time) or (len(messages) >= limit):
                    return messages
                messages.append(message)
            max_id = fetched_messages[-1].id

    return messages


def _clean_messages(message_dicts: list[dict], skip_author_ids: set) -> pd.DataFrame:
    """Telegram メッセージ辞書から integrated_content 付き DataFrame を作る。"""
    ids = []
    timestamps = []
    authors = []
    total_reaction_counts = []
    cleaned_messages = []

    cleaned_message_dicts: dict = {}

    for message_dict in reversed(message_dicts):
        # システムメッセージ等は本文なし
        if "message" not in message_dict.keys():
            continue
        # チャンネル投稿など user_id を持たない発言者
        if not message_dict.get("from_id") or "user_id" not in message_dict["from_id"]:
            continue
        if message_dict["from_id"]["user_id"] in skip_author_ids:
            continue

        message_str = message_dict["message"]
        if message_dict.get("media"):
            media = message_dict["media"]
            if media["_"] == "MessageMediaWebPage":
                webpage_dict = media["webpage"]
                if webpage_dict["_"] == "WebPage" and webpage_dict["description"] is not None:
                    message_str += webpage_dict["description"]
            elif media["_"] in ("MessageMediaDocument", "MessageMediaPhoto"):
                continue

        cleaned_message_dicts[message_dict["id"]] = message_str

        if message_dict.get("reply_to"):
            try:
                reply_to = cleaned_message_dicts[message_dict["reply_to"]["reply_to_msg_id"]]
                message_str = f"reply to '{reply_to[:50]}'\n" + message_str
            except KeyError:
                message_str = "reply to deleted message\n" + message_str

        reaction_count = 0
        if message_dict.get("reactions") is not None:
            for reaction_result in message_dict["reactions"]["results"]:
                reaction_count += reaction_result["count"]

        ids.append(message_dict["id"])
        timestamps.append(pd.to_datetime(message_dict["date"]).tz_convert("Asia/Tokyo"))
        authors.append(message_dict["from_id"]["user_id"])
        total_reaction_counts.append(reaction_count)
        cleaned_messages.append(message_str)

    return pd.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps,
            "author": authors,
            "total_reaction_count": total_reaction_counts,
            "integrated_content": cleaned_messages,
        }
    )


class TelegramSource(MessageSource):
    def fetch(self, channel, *, hours, start_delta_hours=0, limit=7000):
        try:
            import telethon  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Telethon is not installed. Install with: pip install 'chat-digest[telegram]'"
            ) from exc

        api_id = os.getenv(self.config.api_id_env, "")
        api_hash = os.getenv(self.config.api_hash_env, "")
        if not api_id or not api_hash:
            raise RuntimeError(
                f"Telegram credentials missing: set {self.config.api_id_env} and {self.config.api_hash_env}"
            )

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        upper_time = now - datetime.timedelta(hours=start_delta_hours)
        cutoff_time = upper_time - datetime.timedelta(hours=hours)

        messages = asyncio.run(
            _collect_messages(
                self.config.session_name,
                int(api_id),
                api_hash,
                str(self.resolve_channel(channel)),
                cutoff_time=cutoff_time,
                upper_time=upper_time,
                limit=limit,
            )
        )
        message_dicts = [message.to_dict() for message in messages]
        effective = _clean_messages(message_dicts, set(self.config.skip_author_ids))
        raw = pd.DataFrame(message_dicts)
        return FetchResult(raw=raw, effective=effective)

    def url_context(self, channel):
        return UrlContext(
            url_base=TELEGRAM_URL_BASE,
            server_part=str(self.resolve_channel(channel)),
            channel_part="",
        )
