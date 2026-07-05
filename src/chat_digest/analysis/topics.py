"""トピック分析: 抽出 → 投票 → チャット分類 → 要約 → 短縮 → タイトル/ハイライト生成。

LLM の list 出力は ast.literal_eval でパースし、形式不正時はリトライ指示を
付けて再試行する。
"""

import ast
import concurrent.futures
import copy

import pandas as pd

from chat_digest.llm.client import call_llm, call_llm_messages, count_gemini_tokens
from chat_digest.llm.prompt_cache import (
    GEMINI_PROMPT_CACHE_MIN_TOKENS,
    build_topic_judge_cacheable_text,
    build_topic_judge_prompt_payload,
    call_topic_judge_llm,
    is_gemini_model,
    model_supports_cache_control,
)

# トピック判定プロンプトは、プロンプトインジェクション検出時に 99 を返すよう指示している。
# これを超える値は「分類不能」として扱う。
INJECTION_SENTINEL_THRESHOLD = 90

DEFAULT_CONDENSE_SYSTEM_PROMPT = (
    "あなたは熟練の編集者です。重要な固有名詞や具体的な数値を残したまま、"
    "与えられた要約を簡潔に書き直してください。"
)
DEFAULT_CONDENSE_USER_PROMPT = """以下の要約を{max_chars}文字以内で簡潔にまとめ直してください。
- トレーダーにとって核となる情報を保持する。
- 重要な固有名詞や具体的な数値は残す。
- 箇条書きではなく、わかりやすい形式の文章で述べる。
- 話題が変わる場合は改行を入れて段落を分ける。ただし、改行する際は１回の改行で済ませる。
- 文章の先頭と改行後の先頭に、必ず全角のスペースを入れる。
- ネット民らしいフランクな言葉遣いでありながら自然な日本語にする。
- まとめ直した文章のみを出力する。
- 繰り返しますが、{max_chars}文字以内に収めてください。
元の要約:
{summary}
"""


class LLMParseError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected list."""


def _extract_list_literal(raw_text: str, context: str) -> str:
    if not isinstance(raw_text, str):
        raise LLMParseError(f"{context}: response is not text.")

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise LLMParseError(f"{context}: list brackets not found.")

    return raw_text[start:end + 1]


def parse_literal_list(raw_text: str, context: str = "LLM response") -> list:
    list_literal = _extract_list_literal(raw_text, context)
    try:
        parsed = ast.literal_eval(list_literal)
    except (SyntaxError, ValueError) as exc:
        raise LLMParseError(f"{context}: invalid list literal: {exc}") from exc

    if not isinstance(parsed, list):
        raise LLMParseError(f"{context}: parsed value is not a list.")
    return parsed


def validate_topic_list(value: list, topic_num: int | None = None,
                        context: str = "topic list") -> list:
    if not isinstance(value, list):
        raise LLMParseError(f"{context}: parsed value is not a list.")
    if topic_num is not None and len(value) != topic_num:
        raise LLMParseError(
            f"{context}: expected {topic_num} topics, got {len(value)}."
        )

    topics = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise LLMParseError(f"{context}: item {idx} is not [title, description].")
        title = str(item[0]).strip()
        description = str(item[1]).strip()
        if not title or not description:
            raise LLMParseError(f"{context}: item {idx} has empty title or description.")
        topics.append([title, description])
    return topics


def validate_string_list(value: list, context: str = "string list") -> list[str]:
    if not isinstance(value, list):
        raise LLMParseError(f"{context}: parsed value is not a list.")

    strings = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, str):
            raise LLMParseError(f"{context}: item {idx} is not a string.")
        text = item.strip()
        if text:
            strings.append(text)
    if not strings:
        raise LLMParseError(f"{context}: list is empty.")
    return strings


def _add_parse_retry_instruction(user_prompt: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "前回出力は構文エラーまたは形式不正でした。"
        "完全なlistのみを出力してください。"
        "前置き、Markdown、説明文は出力しないでください。"
    )


def call_llm_with_parse_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature=0.0,
    max_retries: int = 3,
    context: str = "LLM response",
    validator=None,
) -> list:
    last_error = None
    current_user_prompt = user_prompt
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                temperature=temperature,
            )
            parsed = parse_literal_list(raw, context=context)
            if validator is not None:
                return validator(parsed)
            return parsed
        except Exception as exc:
            last_error = exc
            print(f"{context} parse retry {attempt}/{max_retries} failed:", exc)
            if attempt < max_retries:
                current_user_prompt = _add_parse_retry_instruction(user_prompt)

    raise LLMParseError(f"{context}: failed after {max_retries} attempts.") from last_error


def detect_topics(chats_str: str,
                  prompt_dict: dict,
                  model: str,
                  topic_num: int,
                  attempts: int,
                  temperature=0.2) -> list:
    """attempts 回並列で LLM にトピック抽出させ、候補リストのリストを返す。"""
    topic_detect_userprompt = prompt_dict["TOPIC_DETECT_USER_PROMPT"].format(
        chats_str=chats_str,
        topic_num=topic_num
    )

    topic_candidates = []

    def _detect_once() -> list:
        return call_llm_with_parse_retry(
            model=model,
            system_prompt=prompt_dict["TOPIC_DETECT_SYS_PROMPT"],
            user_prompt=topic_detect_userprompt,
            temperature=temperature,
            context="topic detect",
            validator=lambda value: validate_topic_list(
                value,
                topic_num=topic_num,
                context="topic detect",
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_detect_once) for _ in range(attempts)]
        for future in concurrent.futures.as_completed(futures):
            try:
                topic_candidates.append(future.result())
            except Exception as e:
                print("Error during topic detection:", e)
                continue

    return topic_candidates


def vote_topics(prompt_dict: dict,
                model: str,
                topic_candidate_list_str: str,
                topic_num: int,
                attempts: int) -> list:
    topic_vote_userprompt = (
        prompt_dict["TOPIC_VOTE_USER_PROMPT"].
        format(TOPIC_DETECT_ATTEMPTS=attempts,
               cancidate_list=topic_candidate_list_str,
               topic_num=topic_num))

    return call_llm_with_parse_retry(
        model=model,
        system_prompt=prompt_dict["TOPIC_VOTE_SYSTEM_PROMPT"],
        user_prompt=topic_vote_userprompt,
        temperature=1,
        context="topic vote",
        validator=lambda value: validate_topic_list(
            value,
            topic_num=topic_num,
            context="topic vote",
        ),
    )


def fetch_chat_topic(target_index: int,
                     chat_df: pd.DataFrame,
                     topic_list_str: str,
                     topic_num: int,
                     prompt_dict: dict,
                     model: str,
                     former_chat_num: int,
                     use_prompt_cache: bool = True,
                     thinking_level: str | None = None) -> int:
    """target_index 行のチャットをトピック分類し、topic_id(1始まり)を返す。"""
    former_index = max(0, target_index - former_chat_num)

    former_chats = ""
    for i in range(target_index - former_index):
        row_idx = former_index + i
        former_chats += chat_df.iloc[row_idx]["integrated_content"]
        former_chats += "\n"

    target_chat = chat_df.iloc[target_index]["integrated_content"]

    system_prompt = prompt_dict["TOPIC_JUDGE_SYS_FORMAT"]
    topic_judge_userpro, cache_messages = build_topic_judge_prompt_payload(
        model=model,
        system_prompt=system_prompt,
        user_template=prompt_dict["TOPIC_JUDGE_USER_FORMAT"],
        topic_list=topic_list_str,
        topic_num_plusone=topic_num + 1,
        former_chats=former_chats,
        target_chat=target_chat,
        use_prompt_cache=use_prompt_cache,
    )

    res = call_topic_judge_llm(
        model=model,
        system_prompt=system_prompt,
        full_prompt=topic_judge_userpro,
        cache_messages=cache_messages,
        call_llm_fn=call_llm,
        call_llm_messages_fn=call_llm_messages,
        thinking_level=thinking_level,
        on_cache_error=lambda exc: print(
            "prompt cache request failed; retrying without cache:",
            exc,
        ),
    )

    try:
        res = int(res)
    except ValueError:
        print("value error", res)
        res = int(topic_num) + 1
    return res


def get_chats_topic(chat_df: pd.DataFrame,
                    topic_list: list,
                    topic_num: int,
                    prompt_dict: dict,
                    model: str,
                    former_chat_num: int,
                    max_workers=10,
                    topic_contexts: list | None = None,
                    use_prompt_cache: bool = True,
                    thinking_level: str | None = None):
    """全チャットを並列でトピック分類し、(topic_id 付き df, 集計用 df) を返す。"""
    target_df = copy.deepcopy(chat_df)

    topic_list_str = ""
    for i, topic in enumerate(topic_list):
        topic_list_str += f"topic{i + 1}: " + str(topic[0]) + "\n"
        context_text = topic[1]
        if topic_contexts and len(topic_contexts) > i and topic_contexts[i]:
            context_text = topic_contexts[i].get("judge_text", context_text)
        topic_list_str += f"topic{i + 1}の説明: " + str(context_text) + "\n"
    topic_list_str += f'topic{str(len(topic_list) + 1)}:"その他"'

    chat_topic_list = [None] * len(target_df.index)
    target_indices = list(target_df.index)
    effective_use_prompt_cache = (
        use_prompt_cache and model_supports_cache_control(model)
    )
    # Gemini の暗黙キャッシュには最低トークン数の下限があるため、事前に判定する
    if effective_use_prompt_cache and is_gemini_model(model) and target_indices:
        system_prompt = prompt_dict["TOPIC_JUDGE_SYS_FORMAT"]
        cacheable_text = build_topic_judge_cacheable_text(
            model=model,
            system_prompt=system_prompt,
            user_template=prompt_dict["TOPIC_JUDGE_USER_FORMAT"],
            topic_list=topic_list_str,
            topic_num_plusone=topic_num + 1,
            former_chats="",
            target_chat="",
            use_prompt_cache=effective_use_prompt_cache,
        )
        if cacheable_text is None:
            effective_use_prompt_cache = False
        else:
            try:
                cacheable_tokens = count_gemini_tokens(model, cacheable_text)
            except Exception as exc:
                print(f"prompt cache disabled: token count failed: {exc}")
                effective_use_prompt_cache = False
            else:
                if cacheable_tokens < GEMINI_PROMPT_CACHE_MIN_TOKENS:
                    print(
                        "prompt cache disabled: "
                        f"cacheable_tokens={cacheable_tokens}, "
                        f"min={GEMINI_PROMPT_CACHE_MIN_TOKENS}"
                    )
                    effective_use_prompt_cache = False

    # キャッシュ有効時は先頭の1件を先に流してキャッシュを作らせる
    if effective_use_prompt_cache and target_indices:
        first_idx = target_indices[0]
        try:
            chat_topic_list[first_idx] = fetch_chat_topic(
                target_index=first_idx,
                chat_df=target_df,
                topic_list_str=topic_list_str,
                topic_num=topic_num,
                prompt_dict=prompt_dict,
                model=model,
                former_chat_num=former_chat_num,
                use_prompt_cache=effective_use_prompt_cache,
                thinking_level=thinking_level,
            )
        except Exception as e:
            print(f"Error at index {first_idx}: {e}")
            chat_topic_list[first_idx] = -1
        target_indices = target_indices[1:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fetch_chat_topic,
                            target_index=idx,
                            chat_df=target_df,
                            topic_list_str=topic_list_str,
                            topic_num=topic_num,
                            prompt_dict=prompt_dict,
                            model=model,
                            former_chat_num=former_chat_num,
                            use_prompt_cache=effective_use_prompt_cache,
                            thinking_level=thinking_level,
                            ): idx
            for idx in target_indices
        }

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                chat_topic_list[idx] = result
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                chat_topic_list[idx] = -1

    target_df['topic_id'] = chat_topic_list

    topic_id_df = target_df[['timestamp', 'topic_id']]
    topic_id_df = topic_id_df[topic_id_df['topic_id'] != len(topic_list) + 1]
    df_topic_list = []
    for item in topic_id_df.itertuples():
        if int(item[2]) > INJECTION_SENTINEL_THRESHOLD:
            df_topic_list.append(-1)
        else:
            df_topic_list.append(topic_list[int(item[2]) - 1][0])
    topic_id_df['topic'] = df_topic_list

    return target_df, topic_id_df


def get_topic_summary(topic_list: list,
                      topic_tagged_df: pd.DataFrame,
                      prompt_dict: dict,
                      model: str,
                      topic_context_list: list | None = None) -> list:
    """トピックごとにチャットを束ねて並列で要約する。末尾は「その他」トピック。"""
    topic_chat_dict = {}

    for topic_idx in range(len(topic_list) + 1):
        topic_id = topic_idx + 1
        topic_chats_str = ""

        context_text = ""
        if topic_context_list and topic_idx < len(topic_context_list):
            context_text = topic_context_list[topic_idx].get("summary_hint", "")
        if context_text:
            topic_chats_str += f"補足情報:\n{context_text}\n"

        topic_chat_df = topic_tagged_df[topic_tagged_df['topic_id'] == topic_id]
        for chat_row in topic_chat_df.itertuples():
            topic_chats_str += f'{chat_row.integrated_content}\n'

        topic_chat_dict[topic_id] = topic_chats_str

    def summarize_topic(topic_id: int, topic_chats_str: str) -> tuple:
        user_prompt = prompt_dict["TOPIC_SUMMARY_CHAT_USER_PROMPT"].format(
            topic=topic_id,
            topic_chat=topic_chats_str
        )
        res = call_llm(
            model,
            prompt_dict.get("TOPIC_SUMMARY_CHAT_SYSTEM_PROMPT", ""),
            user_prompt,
            temperature=0.8
        )
        return (topic_id, res)

    topic_summary_list = [None] * (len(topic_list) + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_topic_id = {}
        for topic_id, topic_chats_str in topic_chat_dict.items():
            future = executor.submit(summarize_topic, topic_id, topic_chats_str)
            future_to_topic_id[future] = topic_id

        for future in concurrent.futures.as_completed(future_to_topic_id):
            tid = future_to_topic_id[future]
            try:
                topic_id, summary_text = future.result()
                topic_summary_list[topic_id - 1] = summary_text
            except Exception as e:
                print(f"Error during summarizing topic {tid}: {e}")
                topic_summary_list[tid - 1] = None

    return topic_summary_list


def condense_topic_summaries(summary_list: list,
                             prompt_dict: dict,
                             model: str,
                             max_chars: int = 200) -> list:
    """冗長になりがちなトピック要約を再度 LLM に通して短く整形する。"""
    if summary_list is None:
        return summary_list
    if len(summary_list) == 0:
        return summary_list

    system_prompt = prompt_dict.get(
        "TOPIC_SUMMARY_CONDENSE_SYSTEM_PROMPT",
        DEFAULT_CONDENSE_SYSTEM_PROMPT
    )
    user_template = prompt_dict.get(
        "TOPIC_SUMMARY_CONDENSE_USER_PROMPT",
        DEFAULT_CONDENSE_USER_PROMPT
    )

    def _condense_one(idx_summary):
        idx, summary = idx_summary
        if summary is None or summary.strip() == "":
            return idx, summary
        user_prompt = user_template.format(
            summary=summary,
            max_chars=max_chars
        )
        try:
            res = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2
            )
            condensed_text = res.strip()
            trim_limit = int(max_chars * 1.2)
            if len(condensed_text) > trim_limit:
                condensed_text = condensed_text[:trim_limit]
            return idx, condensed_text
        except Exception as e:
            print("condense error:", e)
            return idx, summary

    condensed_list = [None] * len(summary_list)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(summary_list))) as executor:
        futures = executor.map(_condense_one, enumerate(summary_list))
        for idx, result in futures:
            condensed_list[idx] = result

    return condensed_list


def generate_global_highlight(topic_summaries: list,
                              supplemental_info: list,
                              prompt_dict: dict,
                              model: str) -> str:
    """全トピックを俯瞰した50字以内のヘッドラインを生成する。"""
    if not topic_summaries:
        return ""
    system_prompt = prompt_dict.get("HIGHLIGHT_GLOBAL_SYSTEM", "")
    user_template = prompt_dict.get("HIGHLIGHT_GLOBAL_USER", "")

    topics_str = ""
    for idx, summary in enumerate(topic_summaries):
        topics_str += f"[topic{idx + 1}]\n{summary}\n"

    supplement_str = ""
    if supplemental_info:
        for idx, supp in enumerate(supplemental_info):
            if supp:
                supplement_str += f"[topic{idx + 1}] {supp}\n"

    user_prompt = user_template.format(
        all_topics=topics_str,
        all_supplements=supplement_str or "補足なし"
    )
    try:
        res = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0
        )
        highlight = res.strip().replace("\n", "")
        if len(highlight) > 50:
            highlight = highlight[:50]
        return highlight
    except Exception as e:
        print("global highlight error:", e)
        return ""


def generate_topic_titles_from_summaries(topic_summaries: list,
                                         prompt_dict: dict,
                                         model: str) -> list:
    if not topic_summaries:
        return []
    system_prompt = prompt_dict.get("TITLE_SYSTEM", "")
    user_template = prompt_dict.get("TITLE_USER", "")

    def _make_title(idx_summary):
        idx, summary = idx_summary
        user_prompt = user_template.format(topic_summary=summary)
        try:
            res = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0
            )
            title = res.strip().replace("\n", "")
            if len(title) > 40:
                title = title[:40]
            return idx, title
        except Exception as e:
            print("title gen error:", e)
            return idx, ""

    titles = [""] * len(topic_summaries)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(topic_summaries))) as executor:
        for idx, title in executor.map(_make_title, enumerate(topic_summaries)):
            titles[idx] = title
    return titles
