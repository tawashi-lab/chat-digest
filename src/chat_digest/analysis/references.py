"""要約への参照リンク付与。

要約から特徴フレーズを抽出し、embedding 類似度で対応する元チャットを探して
フレーズを [フレーズ](メッセージURL) 形式のリンクに置き換える。
"""

import concurrent.futures
import copy
import re
import time

import numpy as np
import pandas as pd

from chat_digest.analysis.topics import call_llm_with_parse_retry, validate_string_list
from chat_digest.llm.client import call_llm, create_embeddings
from chat_digest.sources.base import UrlContext


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_summary(summary_text, full_split=False):
    """要約を句読点・改行で区切り、(本文, 区切り文字) のリストにする。"""
    if full_split:
        pattern = r'(?P<segment>.*?)(?P<delimiter>[。、！？\n])'
    else:
        pattern = r'(?P<segment>.*?)(?P<delimiter>[。\n])'

    summary_segments = []
    last_end = 0
    for match in re.finditer(pattern, summary_text):
        segment = match.group('segment')
        delimiter = match.group('delimiter')
        summary_segments.append((segment, delimiter))
        last_end = match.end()

    if last_end < len(summary_text):
        summary_segments.append((summary_text[last_end:], ""))
    summary_segments = [
        (segment[0].strip(), segment[1]) for segment in summary_segments if segment[0].strip()
    ]
    return summary_segments


def add_embedding_to_df(chat_df, model):
    chat_df = chat_df.copy()
    chat_df = chat_df.dropna(subset=['integrated_content'])
    content_list = chat_df['integrated_content'].tolist()
    content_list = [content if content else "空っぽ" for content in content_list]

    chat_df['embedding'] = create_embeddings(model, content_list)
    return chat_df


def replace_once_if_not_bracketed(text: str, phrase: str, url: str) -> str:
    pattern = rf"(?<!\[){re.escape(phrase)}(?!\])"
    replacement = f"[{phrase}]({url})"
    return re.sub(pattern, replacement, text, count=1)


def _process_topic(
    topic_idx: int,
    topic_summary: str,
    topic_list: list,
    chat_df: pd.DataFrame,
    url_ctx: UrlContext,
    prompt_dict: dict,
    extract_phrases_model: str,
    judge_reference_model: str,
    emb_model: str,
) -> str:
    """1トピック分の要約に参照リンクを付与して返す(並列実行される)。"""
    topic_id = topic_idx + 1
    topic = topic_list[topic_idx]

    # 同時に大量の embedding リクエストが飛ばないよう起動をずらす
    time.sleep((topic_idx % 10) * 2)

    featured_phrases = call_llm_with_parse_retry(
        model=extract_phrases_model,
        system_prompt="",
        user_prompt=prompt_dict["EXTRACT_FEATURES"].format(
            summary_title=topic,
            summary=topic_summary,
        ),
        context="featured phrases",
        validator=lambda value: validate_string_list(
            value,
            context="featured phrases",
        ),
    )

    id_df = chat_df[chat_df['topic_id'] == topic_id]
    id_df = add_embedding_to_df(id_df, emb_model)

    split_summaries = [res[0] + res[1] for res in split_summary(topic_summary, full_split=True)]

    featured_split = []
    for phrase in featured_phrases:
        for summary in split_summaries:
            if phrase in summary:
                featured_split.append(summary)

    similar_chats = []
    for featured_split_summary in featured_split:
        query_embedding = create_embeddings(emb_model, [featured_split_summary])
        cos_sim = id_df["embedding"].apply(lambda x: cosine_similarity(query_embedding[0], x))
        top_index = cos_sim.sort_values(ascending=False).index[:6]
        similar_chats.append(id_df.loc[top_index, "integrated_content"])

    referenced_summary = copy.deepcopy(topic_summary)
    for similar_chat, featured_summary, featured_phrase in zip(
        similar_chats, featured_split, featured_phrases
    ):
        similar_chat_str = ""
        for chat_idx, chat in enumerate(similar_chat, start=1):
            similar_chat_str += f"{chat_idx}: '{chat}'\n"

        reference_idx = call_llm(
            model=judge_reference_model,
            system_prompt="",
            user_prompt=prompt_dict["JUDGE_REFERENCE"].format(
                summary_title=topic,
                featured_summary=featured_summary,
                featured_phrase=featured_phrase,
                chat_str=similar_chat_str
            ),
        )
        reference_idx = int(reference_idx)

        row_label = similar_chat.index[reference_idx - 1]
        reference_chat_id = chat_df.loc[row_label, "id"]
        reference_url = url_ctx.message_url(reference_chat_id)

        referenced_summary = replace_once_if_not_bracketed(
            referenced_summary, featured_phrase, reference_url
        )

    return referenced_summary


def add_summary_reference(
    summary_list: list,
    topic_list: list,
    chat_df: pd.DataFrame,
    url_ctx: UrlContext,
    prompt_dict: dict,
    extract_phrases_model: str,
    judge_reference_model: str,
    emb_model: str = "gemini-embedding-2",
) -> list:
    """各トピック要約に参照リンクを並列で付与する。失敗したトピックは元の要約のまま。"""
    topic_list_addsonota = copy.copy(topic_list)
    topic_list_addsonota.append("その他")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_idx = {}
        for topic_idx, topic_summary in enumerate(summary_list):
            future = executor.submit(
                _process_topic,
                topic_idx,
                topic_summary,
                topic_list_addsonota,
                chat_df,
                url_ctx,
                prompt_dict,
                extract_phrases_model,
                judge_reference_model,
                emb_model,
            )
            future_to_idx[future] = topic_idx

        results = [None] * len(summary_list)
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error at topic_idx={idx}: {e}")
                results[idx] = summary_list[idx]

    return results
