"""LiteLLM ラッパー。

モデル名のプロバイダ解決・GPT-5系エイリアス(-hh/-hm/-hl/-mh)の展開・
プロンプトキャッシュ関連パラメータの組み立てを一箇所に集約する。
API キーは環境変数(OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY)から読む。
"""

import base64
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import litellm as _litellm
except ImportError:
    litellm_completion = None
    litellm_responses = None

    class APIError(Exception):  # type: ignore
        pass

    class AuthenticationError(APIError):  # type: ignore
        pass

    class BadRequestError(APIError):  # type: ignore
        pass

    class RateLimitError(APIError):  # type: ignore
        pass

    class APIConnectionError(APIError):  # type: ignore
        pass

    class Timeout(APIError):  # type: ignore
        pass
else:
    APIError = getattr(_litellm, "APIError", Exception)
    AuthenticationError = getattr(_litellm, "AuthenticationError", APIError)
    BadRequestError = getattr(_litellm, "BadRequestError", APIError)
    RateLimitError = getattr(_litellm, "RateLimitError", APIError)
    APIConnectionError = getattr(_litellm, "APIConnectionError", APIError)
    Timeout = getattr(_litellm, "Timeout", APIError)
    litellm_completion = getattr(_litellm, "completion", None)
    litellm_responses = getattr(_litellm, "responses", None)


class LLMCallError(RuntimeError):
    """Raised when an LLM call cannot be completed safely."""


DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "max_retries": 3,
    "timeout_seconds": 120,
    "reasoning_effort": None,
    "service_tier": None,
    "log_prompt_cache_usage": False,
}
LLM_CONFIG: dict[str, Any] = dict(DEFAULT_LLM_CONFIG)


def configure_llm(**overrides: Any) -> None:
    """LLM 呼び出しの共通設定(リトライ回数・タイムアウト等)を上書きする。"""
    unknown = set(overrides) - set(DEFAULT_LLM_CONFIG)
    if unknown:
        raise ValueError(f"Unknown LLM config keys: {sorted(unknown)}")
    LLM_CONFIG.update(overrides)


_SUPPORTED_PREFIXES = {"openai", "anthropic", "gemini"}
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_GPT5_ALIAS_PATTERN = re.compile(r"^(gpt-5(?:\.\d+)?)(?:-(hh|hm|hl|mh))?$")
_GPT5_ALIAS_OPTIONS: dict[Optional[str], tuple[str, str]] = {
    None: ("none", "low"),
    "hh": ("high", "high"),
    "hm": ("high", "medium"),
    "hl": ("high", "low"),
    "mh": ("medium", "high"),
}


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    model_name: str
    request_model: str
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None


def resolve_model_name(model: str) -> str:
    return resolve_model(model).request_model


def resolve_model(model: str) -> ResolvedModel:
    provider, model_name = _split_provider(model)
    model_name, reasoning_effort, verbosity = _resolve_openai_alias(provider, model_name)
    return ResolvedModel(
        provider=provider,
        model_name=model_name,
        request_model=f"{provider}/{model_name}",
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )


def build_messages(
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_paths:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            user_content.append(_build_image_part(image_path))
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    return messages


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    thinking_budget: int | None = None,
    thinking_level: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    service_tier: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
) -> str:
    return call_llm_messages(
        model=model,
        messages=build_messages(system_prompt, user_prompt, image_paths=image_paths),
        temperature=temperature,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        service_tier=service_tier,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )


def call_llm_messages(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    thinking_budget: int | None = None,
    thinking_level: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    service_tier: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
) -> str:
    if litellm_completion is None and litellm_responses is None:
        raise LLMCallError("LiteLLM is not installed. Install the 'litellm' package.")

    resolved_model = resolve_model(model)
    _require_api_key(resolved_model.provider)

    try:
        if _use_responses_api(resolved_model) and litellm_responses is not None:
            request_kwargs = _build_responses_request_kwargs(
                resolved_model=resolved_model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                service_tier=service_tier,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            response = litellm_responses(**request_kwargs)
        elif litellm_completion is not None:
            request_kwargs = _build_request_kwargs(
                resolved_model=resolved_model,
                messages=messages,
                temperature=temperature,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                service_tier=service_tier,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            response = litellm_completion(**request_kwargs)
        else:
            raise LLMCallError("LiteLLM completion API is unavailable.")
    except LLMCallError:
        raise
    except (AuthenticationError, BadRequestError, RateLimitError, APIConnectionError, Timeout, APIError) as exc:
        raise LLMCallError(f"LLM request failed for '{resolved_model.request_model}': {exc}") from exc
    except Exception as exc:
        raise LLMCallError(f"Unexpected LLM failure for '{resolved_model.request_model}': {exc}") from exc

    _log_prompt_cache_usage(response)
    return _extract_text_content(response)


def count_gemini_tokens(model: str, text: str) -> int:
    resolved_model = resolve_model(model)
    if resolved_model.provider != "gemini":
        raise LLMCallError(f"Token counting is only supported for Gemini models: '{model}'")

    _require_api_key(resolved_model.provider)
    genai = _load_google_genai()
    api_key = os.getenv(_PROVIDER_ENV_VARS[resolved_model.provider])

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.count_tokens(
            model=resolved_model.model_name,
            contents=text,
        )
    except Exception as exc:
        raise LLMCallError(f"Gemini token count failed for '{resolved_model.request_model}': {exc}") from exc

    total_tokens = _get_value(response, "total_tokens")
    if total_tokens is None:
        total_tokens = _get_value(response, "totalTokens")
    if total_tokens is None:
        raise LLMCallError("Gemini token count response did not include total_tokens.")

    try:
        return int(total_tokens)
    except (TypeError, ValueError) as exc:
        raise LLMCallError(f"Gemini token count response had invalid total_tokens: {total_tokens!r}") from exc


def _load_google_genai() -> Any:
    try:
        from google import genai
    except Exception as exc:
        raise LLMCallError("google-genai is not installed. Install the 'google-genai' package.") from exc
    return genai


def _split_provider(model: str) -> tuple[str, str]:
    if not model:
        raise LLMCallError("Model name is required.")

    if "/" in model:
        provider, model_name = model.split("/", 1)
        if provider not in _SUPPORTED_PREFIXES:
            raise LLMCallError(f"Unsupported model provider prefix: '{provider}'.")
        if not model_name:
            raise LLMCallError("Model name is required.")
        return provider, model_name

    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", model
    if model.startswith("claude-"):
        return "anthropic", model
    if model.startswith("gemini-"):
        return "gemini", model

    raise LLMCallError(f"Unsupported model '{model}'.")


def _resolve_openai_alias(
    provider: str,
    model_name: str,
) -> tuple[str, Optional[str], Optional[str]]:
    if provider != "openai":
        return model_name, None, None

    match = _GPT5_ALIAS_PATTERN.fullmatch(model_name)
    if not match:
        return model_name, None, None

    base_model, alias = match.groups()
    reasoning_effort, verbosity = _GPT5_ALIAS_OPTIONS[alias]
    return base_model, reasoning_effort, verbosity


def _build_request_kwargs(
    *,
    resolved_model: ResolvedModel,
    messages: list[dict[str, Any]],
    temperature: float | None,
    thinking_budget: int | None,
    thinking_level: Optional[str],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    service_tier: Optional[str],
    prompt_cache_key: Optional[str],
    prompt_cache_retention: Optional[str],
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": resolved_model.request_model,
        "messages": messages,
        "num_retries": LLM_CONFIG["max_retries"],
        "timeout": LLM_CONFIG["timeout_seconds"],
    }

    effective_temperature = _effective_temperature(resolved_model, temperature)
    if effective_temperature is not None:
        request_kwargs["temperature"] = effective_temperature

    effective_reasoning_effort = _first_present(
        reasoning_effort,
        resolved_model.reasoning_effort,
        LLM_CONFIG.get("reasoning_effort"),
    )
    if effective_reasoning_effort and _supports_openai_reasoning(resolved_model):
        request_kwargs["reasoning_effort"] = effective_reasoning_effort

    effective_verbosity = _first_present(verbosity, resolved_model.verbosity)
    if effective_verbosity and _supports_openai_verbosity(resolved_model):
        request_kwargs["text"] = {"verbosity": effective_verbosity}

    effective_service_tier = _first_present(service_tier, LLM_CONFIG.get("service_tier"))
    if effective_service_tier:
        request_kwargs["service_tier"] = effective_service_tier

    if _supports_gemini_thinking_budget(resolved_model) and thinking_budget is not None:
        request_kwargs["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if _supports_gemini_thinking_level(resolved_model) and thinking_level is not None:
        request_kwargs["thinkingConfig"] = {"thinkingLevel": thinking_level}

    _add_openai_prompt_cache_kwargs(
        request_kwargs,
        resolved_model=resolved_model,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )

    return request_kwargs


def _build_responses_request_kwargs(
    *,
    resolved_model: ResolvedModel,
    messages: list[dict[str, Any]],
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    service_tier: Optional[str],
    prompt_cache_key: Optional[str],
    prompt_cache_retention: Optional[str],
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": resolved_model.request_model,
        "input": messages,
        "num_retries": LLM_CONFIG["max_retries"],
        "timeout": LLM_CONFIG["timeout_seconds"],
    }

    effective_reasoning_effort = _first_present(
        reasoning_effort,
        resolved_model.reasoning_effort,
        LLM_CONFIG.get("reasoning_effort"),
    )
    if effective_reasoning_effort:
        request_kwargs["reasoning_effort"] = effective_reasoning_effort

    effective_verbosity = _first_present(verbosity, resolved_model.verbosity)
    if effective_verbosity:
        request_kwargs["text"] = {"verbosity": effective_verbosity}

    effective_service_tier = _first_present(service_tier, LLM_CONFIG.get("service_tier"))
    if effective_service_tier:
        request_kwargs["service_tier"] = effective_service_tier

    _add_openai_prompt_cache_kwargs(
        request_kwargs,
        resolved_model=resolved_model,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )

    return request_kwargs


def _add_openai_prompt_cache_kwargs(
    request_kwargs: dict[str, Any],
    *,
    resolved_model: ResolvedModel,
    prompt_cache_key: Optional[str],
    prompt_cache_retention: Optional[str],
) -> None:
    if resolved_model.provider != "openai":
        return
    if prompt_cache_key:
        request_kwargs["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        request_kwargs["prompt_cache_retention"] = prompt_cache_retention


def _log_prompt_cache_usage(response: Any) -> None:
    if not LLM_CONFIG.get("log_prompt_cache_usage") and os.getenv("LOG_PROMPT_CACHE_USAGE") != "1":
        return

    usage = _get_value(response, "usage")
    prompt_details = _get_value(usage, "prompt_tokens_details")
    cached_tokens = _get_value(prompt_details, "cached_tokens")
    cache_creation_tokens = _get_value(usage, "cache_creation_input_tokens")
    cache_read_tokens = _get_value(usage, "cache_read_input_tokens")

    values = {
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
    }
    values = {key: value for key, value in values.items() if value is not None}
    if not values or all(value == 0 for value in values.values()):
        print("prompt cache usage: cache hit なし")
        return

    formatted = ", ".join(f"{key}={value}" for key, value in values.items())
    print(f"prompt cache usage: {formatted}")


def _first_present(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None:
            return value
    return None


def _effective_temperature(
    resolved_model: ResolvedModel,
    temperature: float | None,
) -> float | None:
    if temperature is None:
        return None
    if resolved_model.provider == "openai" and resolved_model.model_name.startswith("gpt-5"):
        return None
    if resolved_model.provider == "openai" and resolved_model.model_name.startswith(("o1", "o3", "o4")):
        return 1
    return temperature


def _supports_openai_reasoning(resolved_model: ResolvedModel) -> bool:
    return resolved_model.provider == "openai" and resolved_model.model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_openai_verbosity(resolved_model: ResolvedModel) -> bool:
    return resolved_model.provider == "openai" and resolved_model.model_name.startswith("gpt-5")


def _supports_gemini_thinking_budget(resolved_model: ResolvedModel) -> bool:
    return resolved_model.provider == "gemini" and resolved_model.model_name.startswith("gemini-2.5-")


def _supports_gemini_thinking_level(resolved_model: ResolvedModel) -> bool:
    return resolved_model.provider == "gemini" and resolved_model.model_name.startswith("gemini-3")


def _use_responses_api(resolved_model: ResolvedModel) -> bool:
    return resolved_model.provider == "openai" and resolved_model.model_name.startswith("gpt-5")


def _require_api_key(provider: str) -> None:
    env_var = _PROVIDER_ENV_VARS[provider]
    if not os.getenv(env_var):
        raise LLMCallError(f"Missing required environment variable: {env_var}")


def _build_image_part(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise LLMCallError(f"Image file not found: {image_path}")

    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("image/"):
        raise LLMCallError(f"Unsupported image type for '{image_path}'.")

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise LLMCallError(f"Failed to read image file '{image_path}': {exc}") from exc

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _extract_text_content(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    choices = _get_value(response, "choices")
    if choices:
        first_choice = choices[0]
        message = _get_value(first_choice, "message")
        content = _get_value(message, "content")
        text = _normalize_content(content)
        if text:
            return text

    output = _get_value(response, "output")
    text = _normalize_content(output)
    if text:
        return text

    raise LLMCallError("LiteLLM returned an empty message content.")


def _normalize_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _normalize_content(_get_value(item, "text"))
            if text:
                parts.append(text)
                continue
            nested_content = _get_value(item, "content")
            text = _normalize_content(nested_content)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return _normalize_content(content.get("content"))
    return None


def _get_value(obj: Any, field: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)
