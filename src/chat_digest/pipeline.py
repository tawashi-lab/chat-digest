"""要約パイプラインのオーケストレーション。

fetch → preprocess → トピック抽出/投票 → 分類 → 要約 → (短縮/ハイライト/
ファクトチェック/参照リンク) → Discord Webhook 投稿 → ログ保存。
"""

import os
import pickle
from datetime import datetime
from pathlib import Path

from chat_digest.analysis.fact_check import (
    apply_fact_check_results,
    build_topic_contexts,
    default_topic_contexts,
    extract_fact_check_items,
    run_grok_fact_checks,
)
from chat_digest.analysis.references import add_summary_reference
from chat_digest.analysis.topics import (
    condense_topic_summaries,
    detect_topics,
    generate_global_highlight,
    generate_topic_titles_from_summaries,
    get_chats_topic,
    get_topic_summary,
    vote_topics,
)
from chat_digest.config import AppConfig
from chat_digest.llm.client import configure_llm
from chat_digest.output.graph import create_pivot_table, get_peaks_graph, get_peaks_urls
from chat_digest.output.webhook import (
    make_summary_dicts,
    send_embed_by_webhook,
    send_simple_webhook,
)
from chat_digest.preprocess import chat_df_to_str, get_topic_num
from chat_digest.prompts import load_prompt_profile
from chat_digest.sources import create_source


class PipelineError(RuntimeError):
    pass


def run_job(
    config: AppConfig,
    job_name: str,
    *,
    post_to_override: list[str] | None = None,
    simple_post_to_override: list[str] | None = None,
) -> bool:
    """1ジョブを実行する。要約を投稿したら True、閾値未満でスキップしたら False。

    post_to_override / simple_post_to_override を渡すと投稿先だけ差し替えられる
    (本番設定のままテスト用 Webhook に流す用途)。
    """
    if job_name not in config.jobs:
        raise PipelineError(
            f"Unknown job '{job_name}'. Available: {', '.join(sorted(config.jobs))}"
        )
    job = config.jobs[job_name]
    models = config.job_models(job_name)
    configure_llm(
        max_retries=config.llm.max_retries,
        timeout_seconds=config.llm.timeout_seconds,
    )

    prompt_dict = load_prompt_profile(
        job.prompt_profile,
        extra_dirs=config.prompt_dirs,
        overrides=job.prompt_overrides,
    )

    post_to = post_to_override if post_to_override is not None else job.post_to
    simple_post_to = (
        simple_post_to_override if simple_post_to_override is not None else job.simple_post_to
    )
    for out in [*post_to, *simple_post_to]:
        if out not in config.outputs:
            raise PipelineError(f"Unknown output '{out}'")

    source = create_source(job.source, config.sources[job.source])
    url_ctx = source.url_context(job.channel)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.log_dir) / job_name / timestamp
    os.makedirs(log_dir, exist_ok=True)

    # ---- 取得・前処理 ----
    fetch_result = source.fetch(
        job.channel,
        hours=job.hours,
        start_delta_hours=job.start_delta_hours,
        limit=job.message_limit,
    )
    effective_chat_df = fetch_result.effective
    _save_pickle(fetch_result.raw, log_dir / "log.pkl")

    if effective_chat_df.shape[0] <= job.cutoff_num:
        print(
            f"メッセージ数が少ないため、まとめをスキップします。 {effective_chat_df.shape[0]}"
        )
        return False

    # ---- トピック抽出・投票 ----
    topic_num = get_topic_num(
        effective_chat_df,
        base_num=job.topics.base_num,
        max_num=job.topics.max_num,
        denominator=job.topics.denominator,
    )
    chats_str = chat_df_to_str(effective_chat_df)

    topic_candidates = detect_topics(
        chats_str=chats_str,
        prompt_dict=prompt_dict,
        model=models.topic_detect,
        attempts=job.topics.candidate_num,
        topic_num=topic_num,
    )
    print(topic_candidates)

    topic_list = vote_topics(
        prompt_dict=prompt_dict,
        model=models.topic_vote,
        topic_candidate_list_str=topic_candidates,
        topic_num=topic_num,
        attempts=job.topics.vote_attempts,
    )
    print(topic_list)

    if len(topic_list) != topic_num:
        raise PipelineError(
            f"トピック数が足りません (expected {topic_num}, got {len(topic_list)})"
        )

    # ---- トピック補足情報(Grok) ----
    if job.features.topic_context:
        topic_contexts = build_topic_contexts(
            topic_list=topic_list,
            prompt_dict=prompt_dict,
            element_extract_model=models.topic_element,
            grok_model=models.grok,
            grok_api_key=os.getenv("GROK_API_KEY", ""),
        )
    else:
        topic_contexts = default_topic_contexts(topic_list)
    print(topic_contexts)

    # ---- チャット分類 ----
    id_tagged_df, topic_id_df = get_chats_topic(
        chat_df=effective_chat_df,
        topic_list=topic_list,
        topic_num=topic_num,
        prompt_dict=prompt_dict,
        model=models.topic_judge,
        former_chat_num=job.topics.judge_former_chat_num,
        max_workers=job.topics.judge_max_workers,
        topic_contexts=topic_contexts,
        use_prompt_cache=config.llm.use_prompt_cache,
        thinking_level=config.llm.topic_judge_thinking_level,
    )

    # ---- ピークグラフ・リンク ----
    pivot_df = create_pivot_table(id_df=topic_id_df)

    png_path = ""
    if job.features.graph:
        png_path = get_peaks_graph(pivot_df=pivot_df, png_path=str(log_dir / "peakgraph.png"))

    topic_peaks_urls = get_peaks_urls(
        chat_df=id_tagged_df,
        pivot_df=pivot_df,
        topic_list=topic_list,
        url_ctx=url_ctx,
    )

    # ---- 要約・タイトル ----
    topic_summary_list = get_topic_summary(
        topic_list=topic_list,
        topic_tagged_df=id_tagged_df,
        prompt_dict=prompt_dict,
        model=models.topic_summary,
        topic_context_list=topic_contexts,
    )

    # タイトルは短縮前の要約から生成する
    topic_titles = generate_topic_titles_from_summaries(
        topic_summaries=topic_summary_list[:-1] if job.topics.cut_others else topic_summary_list,
        prompt_dict=prompt_dict,
        model=models.title,
    )
    display_topic_list = []
    for idx, topic in enumerate(topic_list):
        new_title = topic_titles[idx] if idx < len(topic_titles) and topic_titles[idx] else topic[0]
        display_topic_list.append([new_title, topic[1]])

    if job.features.condense.enabled:
        print("トピック要約を短縮中...")
        topic_summary_list = condense_topic_summaries(
            summary_list=topic_summary_list,
            prompt_dict=prompt_dict,
            model=models.condense,
            max_chars=job.features.condense.max_chars,
        )

    # ---- 全体ハイライト ----
    global_highlight = ""
    if job.features.highlight:
        supplemental_texts = [ctx.get("summary_hint", "") for ctx in topic_contexts]
        if len(supplemental_texts) < len(topic_summary_list):
            supplemental_texts.extend([""] * (len(topic_summary_list) - len(supplemental_texts)))
        global_highlight = generate_global_highlight(
            topic_summaries=topic_summary_list,
            supplemental_info=supplemental_texts,
            prompt_dict=prompt_dict,
            model=models.highlight,
        )

    # ---- ファクトチェック ----
    if job.features.fact_check.enabled:
        print("ファクトチェック対象を抽出中...")
        fact_check_items_list = extract_fact_check_items(
            summary_list=topic_summary_list,
            prompt_dict=prompt_dict,
            model=models.fact_check_item,
            max_items=job.features.fact_check.max_items,
        )
        print("ファクトチェック対象:", fact_check_items_list)

        grok_api_key = os.getenv("GROK_API_KEY", "")
        if not grok_api_key:
            print("Grok API key が設定されていないため、ファクトチェックをスキップします。")
            fact_check_reports = [[] for _ in topic_summary_list]
        else:
            fact_check_reports = []
            for summary, items in zip(topic_summary_list, fact_check_items_list):
                fact_check_reports.append(
                    run_grok_fact_checks(
                        summary=summary,
                        items=items,
                        api_key=grok_api_key,
                        model=models.grok,
                    )
                )

        print("ファクトチェック結果を要約へ反映中...")
        topic_summary_list = apply_fact_check_results(
            summary_list=topic_summary_list,
            fact_check_reports=fact_check_reports,
            prompt_dict=prompt_dict,
            model=models.fact_check_rewrite,
        )

    # ---- 参照リンク付与 ----
    if job.features.references:
        topic_summary_list = add_summary_reference(
            summary_list=topic_summary_list,
            topic_list=topic_list,
            chat_df=id_tagged_df,
            url_ctx=url_ctx,
            prompt_dict=prompt_dict,
            extract_phrases_model=models.extract_phrases,
            judge_reference_model=models.judge_reference,
            emb_model=models.embedding,
        )

    # ---- 投稿 ----
    summary_dicts = make_summary_dicts(
        topic_summary_list=topic_summary_list,
        topic_list=display_topic_list,
        urls_list=topic_peaks_urls,
        chat_df=id_tagged_df,
        topic_num=topic_num,
        cut_others=job.topics.cut_others,
    )

    branding = config.branding
    main_message_url = None
    failures = []
    for idx, output_name in enumerate(post_to):
        webhook_url = config.outputs[output_name].resolve_url()
        res, url = send_embed_by_webhook(
            webhook_url,
            summary_dicts,
            job.title,
            png_path,
            description_off=job.description_off,
            url_off=job.url_off,
            color=job.embed_color,
            highlight_all=global_highlight,
            wait=bool(simple_post_to) and idx == 0,
            username=branding.username,
            avatar_url=branding.avatar_url,
            footer=branding.footer,
            server_id=url_ctx.server_part or None,
            channel_id=url_ctx.channel_part or None,
        )
        if idx == 0:
            main_message_url = url
        if not res.ok:
            failures.append(output_name)
            print(f"Webhook送信失敗({output_name}):", res.text)

    # ---- ログ保存 ----
    _save_pickle(summary_dicts, log_dir / "summarydicts.pkl")
    _save_pickle(id_tagged_df, log_dir / "id_tagged_df.pkl")

    # ---- 簡易版(ヘッドライン)投稿 ----
    if simple_post_to:
        fields = _build_simple_fields(
            summary_dicts, display_topic_list, global_highlight, main_message_url
        )
        for output_name in simple_post_to:
            webhook_url = config.outputs[output_name].resolve_url()
            res_simple = send_simple_webhook(
                webhook_url,
                job.title + "ヘッドライン",
                fields,
                color=job.embed_color,
                username=branding.username,
                avatar_url=branding.avatar_url,
            )
            if not res_simple.ok:
                failures.append(output_name)
                print(f"ヘッドライン版送信失敗({output_name}):", res_simple.text)

    if failures:
        raise PipelineError(f"Webhook送信に失敗した投稿先: {', '.join(failures)}")
    return True


def _build_simple_fields(summary_dicts, display_topic_list, global_highlight, main_message_url):
    """ヘッドライン投稿の fields を組み立てる(投稿数が多いトピック順)。"""
    fields = []
    if global_highlight:
        fields.append({"name": "ワンポイント", "value": global_highlight, "inline": False})

    sorted_indices = sorted(
        range(len(summary_dicts)),
        key=lambda i: summary_dicts[i].get("percentage", 0),
        reverse=True,
    )
    sorted_headlines = []
    for idx in sorted_indices:
        title = (
            display_topic_list[idx][0]
            if idx < len(display_topic_list)
            else summary_dicts[idx].get("title", "")
        )
        if not title:
            continue
        percent = summary_dicts[idx].get("percentage", 0)
        sorted_headlines.append(f"- {title} ({percent}%)")

    titles_text = "\n".join(sorted_headlines) if sorted_headlines else ""
    if titles_text:
        fields.append({"name": "トピック", "value": titles_text, "inline": False})
    if main_message_url:
        fields.append({"name": "本編リンク", "value": main_message_url, "inline": False})
    return fields


def _save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
