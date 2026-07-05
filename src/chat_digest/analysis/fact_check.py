"""Grok(xAI)の Web/X 検索を使った補足情報取得とファクトチェック。

xai-sdk は optional 依存(extras: grok)。未インストールまたは API キー未設定の
場合は各関数が空の結果を返してスキップする。
"""

import concurrent.futures
import datetime
import json

from chat_digest.llm.client import call_llm

try:
    from xai_sdk import Client as XaiClient
    from xai_sdk.chat import system as grok_system_message, user as grok_user_message
    from xai_sdk.tools import web_search, x_search
except ImportError:  # pragma: no cover - optional dependency
    XaiClient = None
    grok_system_message = None
    grok_user_message = None
    web_search = x_search = None

DEFAULT_TOPIC_ELEMENT_SYSTEM_PROMPT = ""
DEFAULT_TOPIC_ELEMENT_USER_PROMPT = """あなたはトレーダーのためのリサーチャーです。
下のトピック情報を読み、情報の中から、トレード判断するために知りたい要素や固有名詞を最大{max_items}件抽出してください。
トピック名: {topic_name}
説明: {topic_description}

抽出ルール:
- 株式や仮想通貨など銘柄名が含まれる場合、その概要や直近の値動きを把握する必要があるか判断する
- マクロ指標やイベントも、価格への影響が大きいもののみ選定
- それぞれ「title」「reason」「detail」を含む

以下の形式のみで出力してください:
[{{"title":"要素名","reason":"確認が必要な理由","detail":"補足説明"}}]
"""
DEFAULT_TOPIC_ELEMENT_GROK_PROMPT = """You are a research assistant for crypto traders.
Topic: {topic_name}
Description: {topic_description}

Elements requiring updates:
{elements_json}

For each element:
- summarize what it is (e.g., the project/company/indicator’s role or product) in one line
- if the element is a tradable asset (stock or crypto), add its latest price move or notable change in the last 12h (include % if available), citing the freshest source

Respond ONLY in the following array format:
[{{"title":"", "insight":""}}]
"""

DEFAULT_FACTCHECK_ITEM_SYSTEM_PROMPT = ""
DEFAULT_FACTCHECK_ITEM_USER_PROMPT = """
あなたは事実検証の専門家です。文章の中で、外部情報による確認が必要な要素を抽出してください。
あなたには、あるコミュニティのチャット内容の要約が与えられます。
以下の要約から、ファクトチェックが必要な要素を抽出してください。
- それぞれに短く一意なタイトル(title)を付与する
- なぜ検証が必要なのか(reason)と背景(context)を記載する
- 要約中で、事実として扱われている要素をファクトチェックの対象とする。
- ただし、要約の中でチャットにおける、意見や見込みとして扱われているものはチェックの対象外とする
- 次のような配列のみを出力する: [{{"title":"", "reason":"", "context":""}}, ...]
- 最大{max_items}件

要約:
{summary}
"""
DEFAULT_FACTCHECK_REWRITE_SYSTEM_PROMPT = ""
DEFAULT_FACTCHECK_REWRITE_USER_PROMPT = """
あなたは編集者です。ファクトチェック結果を反映して要約を更新してください
元要約:
{summary}

ファクトチェック結果(JSON):
{fact_check_report}

指示:
- 要約中で、ファクトチェック結果に基づいて誤っていると確実に判断できる記述のみを修正する
- 正しいと判定された記述は維持する
- ファクトチェック結果から確実に判断できない要素は修正しない
- 元の要約の形式を維持する。
- ファクトチェック後の要約のみを出力する。「元要約:」などの接頭辞は付けず、要約から開始する。
"""
DEFAULT_GROK_FACTCHECK_PROMPT = """You are a fact-checking assistant with access to web search.
Using the provided summary and list of statements, verify each element with up-to-date sources.
Return ONLY a JSON array like:
[{{"title":"", "status":"accurate|inaccurate|uncertain", "evidence":"", "suggested_fix":""}}, ...]
- Provide concrete evidence
- Consider the "context" when verifying statements
- suggested_fix should describe how to correct inaccurate or uncertain statements

Summary:
{summary}

Statements to verify:
{items_json}
"""


def extract_topic_elements(topic_name: str,
                           topic_description: str,
                           prompt_dict: dict,
                           model: str,
                           max_items: int = 6) -> list[dict]:
    system_prompt = prompt_dict.get(
        "TOPIC_ELEMENT_SYSTEM_PROMPT",
        DEFAULT_TOPIC_ELEMENT_SYSTEM_PROMPT
    )
    user_template = prompt_dict.get(
        "TOPIC_ELEMENT_USER_PROMPT",
        DEFAULT_TOPIC_ELEMENT_USER_PROMPT
    )
    if not topic_description:
        return []

    user_prompt = user_template.format(
        topic_name=topic_name,
        topic_description=topic_description,
        max_items=max_items
    )
    try:
        res = call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0
        )
        elements = json.loads(res)
        if isinstance(elements, list):
            sanitized = []
            for elem in elements[:max_items]:
                if isinstance(elem, dict):
                    sanitized.append({
                        "title": elem.get("title", "")[:80],
                        "reason": elem.get("reason", ""),
                        "detail": elem.get("detail", "")
                    })
            return sanitized
    except Exception as e:
        print("topic element extraction error:", e)
    return []


def fetch_topic_elements_info(topic_name: str,
                              topic_description: str,
                              elements: list[dict],
                              api_key: str,
                              model: str = "grok-4-fast-reasoning") -> list[dict]:
    if not api_key or not elements:
        return []
    if XaiClient is None or grok_system_message is None or web_search is None:
        return []

    def _fetch(idx_elem):
        idx, element = idx_elem
        user_prompt = DEFAULT_TOPIC_ELEMENT_GROK_PROMPT.format(
            topic_name=topic_name,
            topic_description=topic_description,
            elements_json=json.dumps([element], ensure_ascii=False, indent=2)
        )
        tools = []
        now = datetime.datetime.now(datetime.timezone.utc)
        time_from = now - datetime.timedelta(hours=12)
        if web_search:
            tools.append(web_search())
        if x_search:
            tools.append(x_search(from_date=time_from, to_date=now))
        if not tools:
            return idx, None
        try:
            with XaiClient(api_key=api_key) as client:
                chat_instance = client.chat.create(
                    model=model,
                    messages=[
                        grok_system_message("You provide concise, factual updates for crypto traders."),
                        grok_user_message(user_prompt)
                    ],
                    tools=tools,
                    temperature=0,
                )
                response = chat_instance.sample()
                info = json.loads(response.content)
                if isinstance(info, list) and info:
                    first = info[0]
                    if isinstance(first, dict):
                        record = {
                            "title": first.get("title", element.get("title", "")),
                            "insight": first.get("insight", ""),
                            "source": first.get("source", "")
                        }
                        if hasattr(response, "citations") and response.citations:
                            print("トピック補足 参照元:", response.citations)
                        return idx, record
        except Exception as e:
            print("topic element info fetch error:", e)
        return idx, None

    results = [None] * len(elements)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(elements))) as executor:
        for idx, info in executor.map(_fetch, enumerate(elements)):
            results[idx] = info

    return [item for item in results if item]


def build_topic_contexts(topic_list: list,
                         prompt_dict: dict,
                         element_extract_model: str,
                         grok_model: str,
                         grok_api_key: str,
                         max_items: int = 4) -> list[dict]:
    """各トピックについて Grok 検索で補足情報を集め、判定/要約用のコンテキストを作る。"""
    contexts = []
    if not topic_list:
        return contexts
    if not grok_api_key:
        for topic in topic_list:
            contexts.append({
                "judge_text": topic[1],
                "summary_hint": "",
                "elements": []
            })
        return contexts

    def _process(idx_topic):
        idx, topic = idx_topic
        topic_name = topic[0]
        topic_desc = topic[1]
        elements = extract_topic_elements(
            topic_name=topic_name,
            topic_description=topic_desc,
            prompt_dict=prompt_dict,
            model=element_extract_model,
            max_items=max_items
        )
        info_list = fetch_topic_elements_info(
            topic_name=topic_name,
            topic_description=topic_desc,
            elements=elements,
            api_key=grok_api_key,
            model=grok_model
        ) if elements else []

        supplemental_lines = []
        for info in info_list:
            title = info.get("title", "")
            insight = info.get("insight", "")
            supplemental_lines.append(f"{title}: {insight}")
        supplemental_text = "\n".join(supplemental_lines)

        judge_text = topic_desc
        if supplemental_text:
            judge_text = f"{topic_desc}\n補足情報:\n{supplemental_text}"

        return idx, {
            "judge_text": judge_text,
            "summary_hint": supplemental_text,
            "elements": info_list
        }

    contexts = [None] * len(topic_list)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(topic_list))) as executor:
        future_to_idx = {
            executor.submit(_process, (idx, topic)): idx
            for idx, topic in enumerate(topic_list)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                idx, ctx = future.result()
                contexts[idx] = ctx
            except Exception as e:
                print("topic context build error:", e)
                contexts[future_to_idx[future]] = {
                    "judge_text": topic_list[future_to_idx[future]][1],
                    "summary_hint": "",
                    "elements": []
                }

    return contexts


def default_topic_contexts(topic_list: list) -> list[dict]:
    """Grok 補足なしの素のトピックコンテキストを返す。"""
    return [
        {"judge_text": topic[1], "summary_hint": "", "elements": []}
        for topic in topic_list
    ]


def extract_fact_check_items(summary_list: list,
                             prompt_dict: dict,
                             model: str,
                             max_items: int = 5) -> list[list[dict]]:
    """要約からファクトチェックすべき要素を抽出する。"""
    system_prompt = prompt_dict.get(
        "FACT_CHECK_ITEM_SYSTEM_PROMPT",
        DEFAULT_FACTCHECK_ITEM_SYSTEM_PROMPT
    )
    user_template = prompt_dict.get(
        "FACT_CHECK_ITEM_USER_PROMPT",
        DEFAULT_FACTCHECK_ITEM_USER_PROMPT
    )

    extracted_items = []
    for summary in summary_list:
        if summary is None or summary.strip() == "":
            extracted_items.append([])
            continue
        user_prompt = user_template.format(
            summary=summary,
            max_items=max_items
        )
        try:
            res = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0
            )
            items = json.loads(res)
            if isinstance(items, list):
                sanitized = []
                for item in items[:max_items]:
                    if isinstance(item, dict):
                        sanitized.append({
                            "title": item.get("title", "")[:140],
                            "reason": item.get("reason", ""),
                            "context": item.get("context", ""),
                        })
                extracted_items.append(sanitized)
            else:
                extracted_items.append([])
        except Exception as e:
            print("fact check item extraction error:", e)
            extracted_items.append([])
    return extracted_items


def run_grok_fact_checks(summary: str,
                         items: list[dict],
                         api_key: str,
                         model: str = "grok-4-fast-reasoning",
                         live_sources: list[str] | None = None) -> list[dict]:
    """xai_sdk の Sync Client を使って Grok のファクトチェックを実施する。"""
    if not api_key:
        print("Grok API key is not configured. Skipping fact check.")
        return []
    if XaiClient is None or grok_system_message is None or web_search is None:
        print("xai_sdk がインストールされていないため、ファクトチェックをスキップします。")
        return []
    if not items:
        return []

    live_sources = [src.lower() for src in (live_sources or ["news", "x", "web"])]
    now = datetime.datetime.now(datetime.timezone.utc)
    time_from = now - datetime.timedelta(hours=12)
    tools = []
    # news は専用ソースが無いため web_search で近似
    if "web" in live_sources or "news" in live_sources:
        tools.append(web_search())
    if "x" in live_sources and x_search:
        tools.append(x_search(from_date=time_from, to_date=now))
    if not tools:
        print("利用可能な検索ツールが見つからないため、ファクトチェックをスキップします。")
        return []

    user_prompt = DEFAULT_GROK_FACTCHECK_PROMPT.format(
        summary=summary,
        items_json=json.dumps(items, ensure_ascii=False, indent=2)
    )

    try:
        with XaiClient(api_key=api_key) as client:
            chat_instance = client.chat.create(
                model=model,
                messages=[
                    grok_system_message("You are a factual accuracy checker with live search enabled."),
                    grok_user_message(user_prompt),
                ],
                tools=tools,
                temperature=0,
            )
            response = chat_instance.sample()
            content = response.content

        reports = json.loads(content)
        if isinstance(reports, list):
            sanitized = []
            for report in reports:
                if isinstance(report, dict):
                    sanitized.append({
                        "title": report.get("title", ""),
                        "status": report.get("status", "uncertain"),
                        "evidence": report.get("evidence", ""),
                        "suggested_fix": report.get("suggested_fix", "")
                    })
            print("ファクトチェック結果:", sanitized)
            if hasattr(response, "citations") and response.citations:
                print("参照元URL:", response.citations)
            return sanitized
    except Exception as e:
        print("Grok fact check failed:", e)
    return []


def apply_fact_check_results(summary_list: list,
                             fact_check_reports: list,
                             prompt_dict: dict,
                             model: str) -> list:
    """ファクトチェック結果を踏まえて要約を書き直す。"""
    system_prompt = prompt_dict.get(
        "FACT_CHECK_REWRITE_SYSTEM_PROMPT",
        DEFAULT_FACTCHECK_REWRITE_SYSTEM_PROMPT
    )
    user_template = prompt_dict.get(
        "FACT_CHECK_REWRITE_USER_PROMPT",
        DEFAULT_FACTCHECK_REWRITE_USER_PROMPT
    )

    updated_summaries = []
    for summary, report in zip(summary_list, fact_check_reports):
        if not report:
            updated_summaries.append(summary)
            continue
        user_prompt = user_template.format(
            summary=summary,
            fact_check_report=json.dumps(report, ensure_ascii=False, indent=2)
        )
        try:
            res = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2
            )
            updated_summaries.append(res.strip())
        except Exception as e:
            print("Failed to apply fact check result:", e)
            updated_summaries.append(summary)
    return updated_summaries
