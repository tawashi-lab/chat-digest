"""discord_bot / discord_selfbot 共通のメッセージ収集処理。

discord.py と discord.py-self は同じ `discord` 名前空間・ほぼ同じ API を持つため、
クライアント生成だけを各 source に任せ、履歴収集はここで共通化する。
"""

import datetime
from typing import Any

import pandas as pd

RAW_COLUMNS = [
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

JST = datetime.timezone(datetime.timedelta(hours=9))


def process_message(message: Any) -> dict:
    """discord.Message から必要な情報を抜き出して辞書にまとめる。"""
    total_reaction_count = 0
    if len(message.reactions) > 0:
        total_reaction_count = sum(r.count for r in message.reactions)

    attachments = None
    if len(message.attachments) > 0:
        attachments = [attachment.url for attachment in message.attachments]

    reference_id = message.reference.message_id if message.reference else None
    embed_title = message.embeds[0].title if message.embeds else None
    embed_description = message.embeds[0].description if message.embeds else None

    return {
        "id": message.id,
        "unixtime": message.created_at,
        "timestamp": message.created_at.astimezone(JST).strftime("%Y/%m/%d %H:%M:%S"),
        "author": message.author.name,
        "content": message.content,
        "total_reaction_count": total_reaction_count,
        "attachments": attachments,
        "reference_id": reference_id,
        "embed_title": embed_title,
        "embed_description": embed_description,
    }


def collect_history(
    client: Any,
    token: str,
    channel_id: int,
    *,
    hours: int,
    start_delta_hours: int = 0,
    limit: int = 7000,
) -> pd.DataFrame:
    """クライアントでログインし、指定チャンネルの履歴を DataFrame にして返す。

    on_ready で収集して即座に close するワンショット実行。
    """
    results: dict[str, Any] = {}

    @client.event
    async def on_ready():
        try:
            before = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                hours=start_delta_hours
            )
            after = before - datetime.timedelta(hours=hours)

            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)

            msg_list = []
            async for message in channel.history(
                after=after,
                before=before,
                oldest_first=True,
                limit=limit,
            ):
                msg_list.append(process_message(message))
            results["df"] = pd.DataFrame(msg_list, columns=RAW_COLUMNS)
        except Exception as exc:
            results["error"] = exc
        finally:
            await client.close()

    client.run(token)

    if "error" in results:
        raise RuntimeError(
            f"Failed to fetch Discord history for channel {channel_id}"
        ) from results["error"]
    if "df" not in results:
        raise RuntimeError("Discord client closed before history was collected.")
    return results["df"]
