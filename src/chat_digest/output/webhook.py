"""Discord Webhook への埋め込み投稿。

要約テキストは Discord Embed field の 1024 字制限に収まるよう、markdown リンクを
壊さない位置で分割される。
"""

import copy
import datetime
import json
import re

import pandas as pd
import requests

FIELD_CHAR_LIMIT = 900  # Discord Embed field is max 1024 chars, keep headroom for prefixes/URLs
LINK_PATTERN = re.compile(r"\[[^\]]+\]\([^)]+\)")
SAFE_SPLIT_CHARS = ['\n', '。', '！', '？', '.', '!', '?', '、', ',', ';', ' ']

JST = datetime.timezone(datetime.timedelta(hours=9))


def _find_safe_split(text: str, start: int, end: int) -> int:
    """自然な区切り(文末など)に揃えた分割位置を返す。"""
    for delimiter in SAFE_SPLIT_CHARS:
        idx = text.rfind(delimiter, start, end)
        if idx != -1 and idx + 1 > start:
            return idx + 1
    return end


def _split_plain_text_segment(text: str, limit: int) -> list[str]:
    segments = []
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        window_end = min(cursor + limit, text_length)
        if window_end == text_length:
            segments.append(text[cursor:window_end])
            break

        split_pos = _find_safe_split(text, cursor, window_end)
        if split_pos <= cursor:
            split_pos = window_end

        segments.append(text[cursor:split_pos])
        cursor = split_pos

    return segments


def split_summary_text(summary: str, limit: int = FIELD_CHAR_LIMIT) -> list[str]:
    """[text](url) 形式のリンクを分断しないように要約を limit 字以下に分割する。"""
    if not summary:
        return []

    tokens = []
    last_end = 0
    for match in LINK_PATTERN.finditer(summary):
        if match.start() > last_end:
            tokens.append(summary[last_end:match.start()])
        tokens.append(match.group(0))
        last_end = match.end()
    if last_end < len(summary):
        tokens.append(summary[last_end:])
    if not tokens:
        tokens = [summary]

    parts = []
    current_chunk = ""

    for token in tokens:
        is_link = bool(LINK_PATTERN.fullmatch(token))
        token_segments = [token]

        if not is_link and len(token) > limit:
            token_segments = _split_plain_text_segment(token, limit)

        for segment in token_segments:
            if not segment:
                continue
            segment_parts = [segment]
            if len(segment) > limit and is_link:
                # 極端に長いリンクは稀。最終手段として markdown が壊れても強制分割する
                segment_parts = [segment[i:i + limit] for i in range(0, len(segment), limit)]

            for seg in segment_parts:
                if len(current_chunk) + len(seg) <= limit:
                    current_chunk += seg
                else:
                    if current_chunk.strip():
                        parts.append(current_chunk.strip())
                    current_chunk = seg

    if current_chunk.strip():
        parts.append(current_chunk.strip())

    return [part for part in parts if part]


def make_summary_dicts(topic_summary_list: list,
                       topic_list: list,
                       chat_df: pd.DataFrame,
                       topic_num: int,
                       urls_list: list,
                       cut_others: bool = False):
    """トピックごとの投稿用 dict(タイトル・本文・リンク・件数比)を組み立てる。"""
    if not cut_others:
        topic_list_sonotaplus = copy.copy(topic_list)
        topic_list_sonotaplus.append("その他")
        total_messages = int(chat_df[chat_df['topic_id'] != len(topic_list_sonotaplus)].shape[0])
    else:
        topic_list_sonotaplus = topic_list
        topic_summary_list = topic_summary_list[:-1]
        total_messages = int(chat_df[chat_df['topic_id'] != len(topic_list_sonotaplus) + 1].shape[0])

    final_summary_dicts = []

    for topic_idx, topic_summary in enumerate(topic_summary_list):
        final_summary_dict = {}
        topic_id = topic_idx + 1
        topic_message_num = int(chat_df[chat_df['topic_id'] == topic_id].shape[0])

        if topic_id != topic_num + 1:
            percentage = str(int((topic_message_num / total_messages) * 100))
            topic_summary_title = topic_list_sonotaplus[topic_idx][0]
        else:
            percentage = None
            topic_summary_title = topic_list_sonotaplus[topic_idx]

        final_summary_dict['title'] = (
            topic_summary_title + ": " + percentage + "%"
            if isinstance(percentage, str)
            else topic_summary_title
        )
        final_summary_dict['description'] = topic_summary
        final_summary_dict['description_parts'] = split_summary_text(topic_summary, FIELD_CHAR_LIMIT)
        final_summary_dict['urls'] = []
        final_summary_dict['percentage'] = int(percentage) if isinstance(percentage, str) else 0
        if topic_id != topic_num + 1:
            final_summary_dict['urls'] = urls_list[topic_idx]
        final_summary_dicts.append(final_summary_dict)
    return final_summary_dicts


def make_summaryfield_dict(summaries, description_off, url_off, highlight_all=""):
    summaries_sorted = sorted(summaries, key=lambda x: x['percentage'], reverse=True)
    fields = []
    if highlight_all:
        fields.append({
            'name': "ワンポイント",
            'value': highlight_all,
            'inline': False
        })
    for summary in summaries_sorted:
        if summary.get('highlight'):
            fields.append({
                'name': "ワンポイント",
                'value': summary['highlight'],
                'inline': False
            })

        description_parts = summary.get('description_parts') or []
        if not description_parts and summary.get('description'):
            description_parts = [summary['description']]
        description_parts = [part for part in description_parts if part]

        desc_chunks = description_parts if not description_off else []
        if desc_chunks:
            chunks = desc_chunks
        elif summary['urls'] and not url_off:
            chunks = [None]
        else:
            chunks = []

        if not chunks:
            continue

        for idx, chunk in enumerate(chunks):
            field = {}
            field['name'] = summary['title'] if idx == 0 else f"{summary['title']} (続き{idx})"
            value_lines = []

            if chunk and not description_off:
                chunk_text = chunk.strip()
                chunk_text = chunk_text.replace("\n", "\n> ")
                value_lines.append("> " + chunk_text)

            is_last_chunk = idx == len(chunks) - 1
            if summary['urls'] and is_last_chunk and not url_off:
                urls_str = "リンク: "
                url_entries = []
                for url in summary['urls']:
                    url_entries.append(f"[{url['timestamp'].strftime('%H:%M')}]({url['url']})")
                urls_str += "　".join(url_entries)
                value_lines.append("> " + urls_str)

            if not value_lines:
                continue

            field['value'] = "\n".join(value_lines) + "\n"
            field['inline'] = False
            fields.append(field)
    return fields


def make_embed_payload(summaries, title, graph_png, description_off, url_off,
                       color=0x008f11, highlight_all="",
                       username="chat-digest", avatar_url="", footer=""):
    embed = {
        "title": title,
        "timestamp": str(datetime.datetime.now(JST)),
        "color": color,
        "image": {"url": None},
        "fields": [],
    }
    if footer:
        embed["footer"] = {"text": footer}
    if graph_png != "":
        embed["image"]["url"] = "attachment://peakgraph"

    embed["fields"] = make_summaryfield_dict(summaries, description_off, url_off, highlight_all)

    payload_json = {"username": username, "embeds": [embed]}
    if avatar_url:
        payload_json["avatar_url"] = avatar_url

    return {"payload_json": json.dumps(payload_json, ensure_ascii=False)}


def make_file_payload(graph_png):
    with open(graph_png, 'rb') as f:
        file_peakgraph = f.read()
    return {"peakgraph": ("graph.png", file_peakgraph)}


def send_embed_by_webhook(webhook_url, summaries, title, graph_png="",
                          description_off=False, url_off=False, color=0x008f11,
                          highlight_all="", wait=False,
                          username="chat-digest", avatar_url="", footer="",
                          server_id=None, channel_id=None):
    """埋め込みを Webhook へ送信する。wait=True で投稿メッセージの URL も返す。"""
    if wait:
        webhook_url = webhook_url + ("&wait=true" if "?" in webhook_url else "?wait=true")

    payload = make_embed_payload(
        summaries, title, graph_png,
        description_off=description_off, url_off=url_off, color=color,
        highlight_all=highlight_all,
        username=username, avatar_url=avatar_url, footer=footer,
    )

    if graph_png != "":
        files = make_file_payload(graph_png)
        res = requests.post(webhook_url, data=payload, files=files)
    else:
        res = requests.post(webhook_url, data=payload)

    message_url = None
    if wait and res.ok:
        try:
            data = res.json()
            message_url = data.get("url") or data.get("jump_url")
            if not message_url:
                msg_id = data.get("id")
                resp_channel = data.get("channel_id", channel_id)
                if msg_id and server_id and resp_channel:
                    message_url = f"https://discord.com/channels/{server_id}/{resp_channel}/{msg_id}"
        except Exception as e:
            print("failed to parse webhook response:", e)
    return res, message_url


def send_simple_webhook(webhook_url, title, fields, color=0x008f11,
                        username="chat-digest", avatar_url=""):
    payload_json = {
        "username": username,
        "embeds": [{
            "title": title,
            "color": color,
            "fields": fields
        }]
    }
    if avatar_url:
        payload_json["avatar_url"] = avatar_url
    payload = {"payload_json": json.dumps(payload_json, ensure_ascii=False)}
    return requests.post(webhook_url, data=payload)
