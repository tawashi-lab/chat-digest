"""Discord メッセージの前処理。

同一発言者の連投を1メッセージにグルーピングし、embed・引用を本文へ統合して
LLM に渡せる `integrated_content` 列を作る。
"""

import re

import bleach
import pandas as pd

_URL_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s]+")


def get_topic_num(df, base_num=2, max_num=4, denominator=200):
    """メッセージ数からトピック数を決める(denominator が大きいほど少なめ)。"""
    chats_num = df.shape[0]
    threshold = 0
    for i in range(1, max_num + 1):
        threshold += denominator * ((i ** 1.7) / i)
        if chats_num < threshold:
            return max(base_num, min(i, max_num))
    return max_num


def merge_values_with_newline(series):
    """グループ内の有効な値を改行区切りで結合。すべて None/空文字なら None。"""
    filtered = [str(x) for x in series if x not in [None, ""]]
    if len(filtered) == 0:
        return None
    return "\n".join(filtered)


def preprocess_discord_messages(
    df: pd.DataFrame,
    *,
    news_bot_authors: tuple[str, ...] | list[str] = (),
    group_window_seconds: int = 120,
) -> pd.DataFrame:
    """生のメッセージ DataFrame から要約対象の DataFrame を作る。

    - 同一発言者による group_window_seconds 以内の連投を1行に集約
    - content + embed(先頭100字)を integrated_content に統合
    - 返信は引用元の先頭40字を「引用:」として付与
    - news_bot_authors のメッセージはリアクションが付いた場合のみ
      embed の説明文(先頭100字)を採用(ニュース速報 bot の洪水対策)
    - URL は [URL] に置換し、HTML タグを除去
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
    df["same_author"] = df["author"] == df["author"].shift(1)
    df["new_group"] = (
        ~df["same_author"]
        | (df["time_diff"] > group_window_seconds)
        | (df["time_diff"].isna())
    )
    df["group_id"] = df["new_group"].cumsum()

    agg_dict = {
        "id": "first",
        "unixtime": "first",
        "timestamp": "first",
        "author": "first",
        "reference_id": "first",
        "content": merge_values_with_newline,
        "total_reaction_count": "sum",
        "attachments": merge_values_with_newline,
        "embed_title": merge_values_with_newline,
        "embed_description": merge_values_with_newline,
    }
    aggregated = df.groupby("group_id").agg(agg_dict).reset_index(drop=True)
    aggregated = aggregated.sort_values("timestamp").reset_index(drop=True)

    base_texts = []
    for chat in aggregated.itertuples():
        integrated_content = ""
        if chat.author in news_bot_authors:
            if chat.total_reaction_count > 0 and chat.embed_description is not None:
                integrated_content = chat.embed_description[:100]
            base_texts.append(integrated_content)
            continue

        if chat.content is not None:
            integrated_content += chat.content
        if chat.embed_description is not None:
            title = chat.embed_title if chat.embed_title is not None else ""
            description = chat.embed_description if chat.embed_description is not None else ""
            integrated_content += title + "\n" + description[:100]
        base_texts.append(integrated_content)

    id_to_text = {}
    for chat, text in zip(aggregated.itertuples(), base_texts):
        if chat.id is not None:
            id_to_text[chat.id] = text

    chats_list = []
    for chat, text in zip(aggregated.itertuples(), base_texts):
        reference_note = ""
        if getattr(chat, "reference_id", None):
            ref_text = id_to_text.get(chat.reference_id)
            if ref_text:
                reference_note = f"引用: {ref_text[:40]}\n"
        chats_list.append(reference_note + (text or ""))

    aggregated["integrated_content"] = chats_list

    effective_chat_df = aggregated[aggregated["integrated_content"] != ""]
    effective_chat_df.loc[:, "integrated_content"] = effective_chat_df["integrated_content"].str.replace(
        _URL_PATTERN, "[URL]", regex=True
    )
    effective_chat_df.loc[:, "integrated_content"] = effective_chat_df["integrated_content"].apply(
        lambda x: bleach.clean(x, tags=[], strip=True)
    )
    return effective_chat_df.sort_values("timestamp").reset_index(drop=True)


def chat_df_to_str(df: pd.DataFrame) -> str:
    chats_str = ""
    for chat in df.itertuples():
        chats_str += str(chat.integrated_content)
        chats_str += "\n\n"
    return chats_str
