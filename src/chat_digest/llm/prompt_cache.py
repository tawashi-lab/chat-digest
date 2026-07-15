from typing import Any, Callable, Optional


FORMER_CHATS_PLACEHOLDER = "{former_chats}"
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}
GEMINI_PROMPT_CACHE_MIN_TOKENS = 1024


def model_supports_cache_control(model: str) -> bool:
    if model.startswith(("anthropic/", "claude-")):
        return True
    if not is_gemini_model(model):
        return False

    gemini_model = model.split("/", 1)[1] if model.startswith("gemini/") else model
    return gemini_model.startswith(("gemini-2.5-", "gemini-3"))


def is_gemini_model(model: str) -> bool:
    return model.startswith(("gemini/", "gemini-"))


def split_topic_judge_user_prompt(
    user_template: str,
    *,
    topic_list: str,
    topic_num_plusone: int,
    former_chats: str,
    target_chat: str,
) -> tuple[Optional[str], str, str]:
    format_values = {
        "topic_list": topic_list,
        "topic_num_plusone": topic_num_plusone,
        "former_chats": former_chats,
        "target_chat": target_chat,
    }
    full_prompt = user_template.format(**format_values)

    if FORMER_CHATS_PLACEHOLDER not in user_template:
        return None, full_prompt, full_prompt

    cached_prefix_template, suffix_template = user_template.split(
        FORMER_CHATS_PLACEHOLDER,
        1,
    )
    cached_prefix = cached_prefix_template.format(**format_values)
    suffix = former_chats + suffix_template.format(**format_values)

    if cached_prefix + suffix != full_prompt:
        raise ValueError("Split topic judge prompt does not reconstruct the full prompt.")

    return cached_prefix, suffix, full_prompt


def build_cacheable_prefix_text(
    *,
    model: str,
    system_prompt: str,
    cached_prefix: str,
) -> str:
    if is_gemini_model(model) and system_prompt:
        return f"{system_prompt}\n\n{cached_prefix}"
    return cached_prefix


def build_topic_judge_cacheable_text(
    *,
    model: str,
    system_prompt: str,
    user_template: str,
    topic_list: str,
    topic_num_plusone: int,
    former_chats: str,
    target_chat: str,
    use_prompt_cache: bool,
) -> Optional[str]:
    cached_prefix, _, _ = split_topic_judge_user_prompt(
        user_template,
        topic_list=topic_list,
        topic_num_plusone=topic_num_plusone,
        former_chats=former_chats,
        target_chat=target_chat,
    )

    if not use_prompt_cache or cached_prefix is None:
        return None
    if not model_supports_cache_control(model):
        return None

    return build_cacheable_prefix_text(
        model=model,
        system_prompt=system_prompt,
        cached_prefix=cached_prefix,
    )


def build_cached_topic_judge_messages(
    *,
    model: str,
    system_prompt: str,
    cached_prefix: str,
    suffix: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if is_gemini_model(model):
        # Gemini rejects cachedContent when GenerateContent also has system_instruction.
        # Keep the instruction in the cached prefix instead of sending a system message.
        cached_prefix = build_cacheable_prefix_text(
            model=model,
            system_prompt=system_prompt,
            cached_prefix=cached_prefix,
        )
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": dict(CACHE_CONTROL_EPHEMERAL),
                },
                {"type": "text", "text": suffix},
            ],
        }
    )
    return messages


def build_topic_judge_prompt_payload(
    *,
    model: str,
    system_prompt: str,
    user_template: str,
    topic_list: str,
    topic_num_plusone: int,
    former_chats: str,
    target_chat: str,
    use_prompt_cache: bool,
) -> tuple[str, Optional[list[dict[str, Any]]]]:
    cached_prefix, suffix, full_prompt = split_topic_judge_user_prompt(
        user_template,
        topic_list=topic_list,
        topic_num_plusone=topic_num_plusone,
        former_chats=former_chats,
        target_chat=target_chat,
    )

    if not use_prompt_cache or cached_prefix is None:
        return full_prompt, None
    if not model_supports_cache_control(model):
        return full_prompt, None

    return full_prompt, build_cached_topic_judge_messages(
        model=model,
        system_prompt=system_prompt,
        cached_prefix=cached_prefix,
        suffix=suffix,
    )


def call_topic_judge_llm(
    *,
    model: str,
    system_prompt: str,
    full_prompt: str,
    cache_messages: Optional[list[dict[str, Any]]],
    call_llm_fn: Callable[..., str],
    call_llm_messages_fn: Callable[..., str],
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    on_cache_error: Optional[Callable[[Exception], None]] = None,
) -> str:
    if cache_messages is not None:
        try:
            message_kwargs = {"model": model, "messages": cache_messages}
            if thinking_level is not None:
                message_kwargs["thinking_level"] = thinking_level
            if temperature is not None:
                message_kwargs["temperature"] = temperature
            return call_llm_messages_fn(**message_kwargs)
        except Exception as exc:
            if on_cache_error is not None:
                on_cache_error(exc)

    plain_kwargs = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": full_prompt,
    }
    if thinking_level is not None:
        plain_kwargs["thinking_level"] = thinking_level
    if temperature is not None:
        plain_kwargs["temperature"] = temperature
    return call_llm_fn(**plain_kwargs)
