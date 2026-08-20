from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Literal, Unpack

import httpx
import structlog.stdlib
import tiktoken
import tenacity
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import ConfigDict, Field

from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    RetryConfig,
    get_model_context_window_length,
    warn_about_non_empty_params,
)

logger = structlog.stdlib.get_logger(component=__name__)


_SEM_LOCK = asyncio.Lock()
_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_CALL_LOCK = asyncio.Lock()
_CALL_COUNTER = 0


class RetryableHTTPError(RuntimeError):
    pass


def _not_given(value: Any) -> bool:
    return isinstance(value, NotGiven)


async def _get_semaphore(limit: int) -> asyncio.Semaphore:
    async with _SEM_LOCK:
        if limit not in _SEMAPHORES:
            _SEMAPHORES[limit] = asyncio.Semaphore(limit)
        return _SEMAPHORES[limit]


async def _next_call_id() -> int:
    global _CALL_COUNTER
    async with _CALL_LOCK:
        _CALL_COUNTER += 1
        return _CALL_COUNTER


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
        return "\n".join(p for p in parts if p)
    return str(content)


def _conversation_to_responses_input(
    conversation: TurnCompleter.RuntimeConversation,
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in conversation:
        role = str(message.get("role", "user"))
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        converted.append(
            {
                "type": "message",
                "role": role,
                "content": _message_content_to_text(message.get("content")),
            }
        )
    return converted


def _conversation_to_chat_messages(
    conversation: TurnCompleter.RuntimeConversation,
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in conversation:
        role = str(message.get("role", "user"))
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        converted.append(
            {
                "role": role,
                "content": _message_content_to_text(message.get("content")),
            }
        )
    return converted


def _message_token_count(
    message: dict[str, str],
    token_encoder: tiktoken.Encoding,
) -> int:
    return len(
        token_encoder.encode(
            str(message.get("content", "")),
            disallowed_special=(),
        )
    )


def _messages_token_count(
    messages: list[dict[str, str]],
    token_encoder: tiktoken.Encoding,
) -> int:
    return sum(_message_token_count(message, token_encoder) for message in messages)


def _judge_input_token_cap() -> int | None:
    raw_cap = os.getenv("PAPERBENCH_JUDGE_INPUT_TOKEN_CAP")
    if not raw_cap:
        return None
    try:
        cap = int(raw_cap)
    except ValueError:
        logger.warning(
            "Ignoring invalid PAPERBENCH_JUDGE_INPUT_TOKEN_CAP",
            value=raw_cap,
        )
        return None
    if cap <= 0:
        logger.warning(
            "Ignoring non-positive PAPERBENCH_JUDGE_INPUT_TOKEN_CAP",
            value=raw_cap,
        )
        return None
    return cap


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _truncate_text_to_tokens(
    text: str,
    token_encoder: tiktoken.Encoding,
    max_tokens: int,
) -> str:
    if max_tokens <= 0:
        return ""
    tokens = token_encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    marker = "\n\n[PaperBench judge input truncated to avoid a gateway failure on an oversized request.]\n"
    marker_tokens = token_encoder.encode(marker, disallowed_special=())
    if len(marker_tokens) >= max_tokens:
        return token_encoder.decode(tokens[:max_tokens])
    return token_encoder.decode(tokens[: max_tokens - len(marker_tokens)]) + marker


def _cap_chat_messages_by_tokens(
    messages: list[dict[str, str]],
    token_encoder: tiktoken.Encoding,
    token_cap: int,
) -> tuple[list[dict[str, str]], int, int]:
    original_tokens = _messages_token_count(messages, token_encoder)
    if original_tokens <= token_cap:
        return messages, original_tokens, original_tokens

    capped_messages = [dict(message) for message in messages]
    # Preserve the system prompt and the final rubric/answer-format messages.
    protected_indexes = {0}
    if len(capped_messages) >= 2:
        protected_indexes.add(len(capped_messages) - 2)
    if len(capped_messages) >= 1:
        protected_indexes.add(len(capped_messages) - 1)
    truncatable_indexes = [
        i for i in range(len(capped_messages)) if i not in protected_indexes
    ]
    if not truncatable_indexes:
        return capped_messages, original_tokens, original_tokens

    current_tokens = original_tokens
    while current_tokens > token_cap:
        token_counts = {
            i: _message_token_count(capped_messages[i], token_encoder)
            for i in truncatable_indexes
        }
        target_index, target_tokens = max(token_counts.items(), key=lambda item: item[1])
        if target_tokens <= 0:
            break
        excess_tokens = current_tokens - token_cap
        new_target_tokens = max(0, target_tokens - excess_tokens)
        capped_messages[target_index]["content"] = _truncate_text_to_tokens(
            capped_messages[target_index].get("content", ""),
            token_encoder,
            new_target_tokens,
        )
        next_tokens = _messages_token_count(capped_messages, token_encoder)
        if next_tokens >= current_tokens:
            break
        current_tokens = next_tokens

    return capped_messages, original_tokens, current_tokens


def _strict_json_schema(model: type[Any]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "name": schema.get("title", model.__name__),
        "schema": schema,
        "strict": True,
    }


def _chat_json_schema(model: type[Any]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("title", model.__name__),
            "schema": schema,
            "strict": True,
        },
    }


def _extract_output_text(response_json: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response_json.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                texts.append(str(content.get("text", "")))
    return "".join(texts)


def _extract_chat_output_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _message_content_to_text(content)
    return "" if content is None else str(content)


def _chat_finish_reason(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    return str(first_choice.get("finish_reason") or "")


def _usage_from_response(response_json: dict[str, Any]) -> CompletionUsage | None:
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return CompletionUsage(
        completion_tokens=output_tokens,
        prompt_tokens=input_tokens,
        total_tokens=int(usage.get("total_tokens") or input_tokens + output_tokens),
    )


def _usage_from_chat_response(response_json: dict[str, Any]) -> CompletionUsage | None:
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    return CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=int(
            usage.get("total_tokens") or prompt_tokens + completion_tokens
        ),
    )


def _response_json_from_http_response(response: httpx.Response) -> dict[str, Any]:
    text = response.text
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type and not text.lstrip().startswith(("event:", "data:")):
        return response.json()

    completed_response: dict[str, Any] | None = None
    deltas: list[str] = []
    done_texts: list[str] = []
    last_response: dict[str, Any] | None = None
    event_name: str | None = None
    data_lines: list[str] = []

    def flush_event() -> None:
        nonlocal completed_response, data_lines, event_name, last_response
        if not data_lines:
            event_name = None
            return
        raw_data = "\n".join(data_lines).strip()
        data_lines = []
        if not raw_data or raw_data == "[DONE]":
            event_name = None
            return
        try:
            event = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unable to parse Responses SSE event: {raw_data[:500]}") from exc

        event_type = str(event.get("type") or event_name or "")
        response_obj = event.get("response")
        if isinstance(response_obj, dict):
            last_response = response_obj
        if event_type == "response.completed" and isinstance(response_obj, dict):
            completed_response = response_obj
        elif event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text_value = event.get("text")
            if isinstance(text_value, str):
                done_texts.append(text_value)
        event_name = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush_event()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    flush_event()

    if completed_response is not None:
        return completed_response
    content = done_texts[-1] if done_texts else "".join(deltas)
    if content:
        return {
            "output": [{"content": [{"type": "output_text", "text": content}]}],
            "usage": (last_response or {}).get("usage"),
        }
    raise RuntimeError(f"Responses SSE stream did not contain output text: {text[:500]}")


def _request_headers(api_key_env: str) -> dict[str, str]:
    api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"{api_key_env} or OPENAI_API_KEY must be set")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "codex-cli/0.138.0",
    }


def _split_base_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [
        item.strip().rstrip("/")
        for item in raw.replace("\n", ",").split(",")
        if item.strip()
    ]
    return parts


def _dedupe_base_urls(urls: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _candidate_base_urls(primary: str) -> list[str]:
    urls = [primary.rstrip("/")]
    urls.extend(_split_base_urls(os.getenv("PAPERBENCH_RESPONSES_ALT_BASE_URLS")))

    if os.getenv("PAPERBENCH_RESPONSES_DISABLE_IMPLICIT_SU8_FALLBACK") != "1":
        if "cn2.su8.codes" in primary:
            urls.append("https://www.su8.codes/v1")
        elif "www.su8.codes" in primary:
            urls.append("https://cn2.su8.codes/v1")

    return _dedupe_base_urls(urls)


def _model_context_window_length(model: str) -> int:
    try:
        return get_model_context_window_length(model)
    except ValueError as exc:
        raw_context_tokens = os.getenv("PAPERBENCH_JUDGE_CONTEXT_TOKENS")
        if raw_context_tokens:
            try:
                context_tokens = int(raw_context_tokens)
            except ValueError as parse_exc:
                raise ValueError(
                    "PAPERBENCH_JUDGE_CONTEXT_TOKENS must be an integer token count "
                    f"for model {model!r}; got {raw_context_tokens!r}"
                ) from parse_exc
            if context_tokens <= 0:
                raise ValueError(
                    "PAPERBENCH_JUDGE_CONTEXT_TOKENS must be positive "
                    f"for model {model!r}; got {context_tokens}"
                )
            logger.warning(
                "Using explicitly configured judge context window for model unknown "
                "to preparedness_turn_completer",
                model=model,
                n_ctx=context_tokens,
                env_var="PAPERBENCH_JUDGE_CONTEXT_TOKENS",
            )
            return context_tokens
        raise ValueError(
            f"Model {model!r} is not known to preparedness_turn_completer. "
            "Set PAPERBENCH_JUDGE_CONTEXT_TOKENS in sota/.env to the model's real "
            "context window before running SimpleJudge, because this value controls "
            "judge context budgeting and truncation."
        ) from exc


class HTTPResponsesTurnCompleter(OpenAICompletionsTurnCompleter):
    """OpenAI-compatible completer using direct HTTP requests.

    The su8 gateway is much more reliable for large PaperBench judge payloads on
    `/chat/completions` than `/responses`. This direct HTTP adapter keeps the
    existing PaperBench judge entrypoint and retry behavior while avoiding SDK
    compatibility issues.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: Literal["low", "medium", "high"] | None | NotGiven = NOT_GIVEN,
        response_format: type[Any] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig | None = None,
        concurrency: int = 8,
        timeout: float = 600.0,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://cn2.su8.codes/v1").rstrip(
            "/"
        )
        self.base_urls = _candidate_base_urls(self.base_url)
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.tools = NOT_GIVEN
        self.tool_choice = NOT_GIVEN
        self.retry_config = retry_config or RetryConfig()
        self.concurrency = concurrency
        self.timeout = timeout
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            self.encoding_name = "o200k_base"
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.n_ctx = _model_context_window_length(model)

    class Config(OpenAICompletionsTurnCompleter.Config):
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        base_url: str | None = None
        api_key_env: str = "OPENAI_API_KEY"
        concurrency: int = 8
        timeout: float = 600.0
        retry_config: RetryConfig = Field(default_factory=RetryConfig)

        def build(self) -> HTTPResponsesTurnCompleter:
            return HTTPResponsesTurnCompleter(
                model=self.model,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                reasoning_effort=self.reasoning_effort,
                response_format=self.response_format,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                retry_config=self.retry_config,
                concurrency=self.concurrency,
                timeout=self.timeout,
            )

    class Completion(OpenAICompletionsTurnCompleter.Completion):
        pass

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> HTTPResponsesTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> HTTPResponsesTurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)
        call_id = await _next_call_id()

        converted_messages = _conversation_to_chat_messages(conversation)
        token_cap = _judge_input_token_cap()
        if token_cap is not None:
            converted_messages, original_input_tokens, capped_input_tokens = (
                _cap_chat_messages_by_tokens(
                    converted_messages,
                    self.token_encoder,
                    token_cap,
                )
            )
            if capped_input_tokens < original_input_tokens:
                logger.warning(
                    "Capped oversized judge input before chat completion request",
                    call_id=call_id,
                    model=self.model,
                    input_tokens_before=original_input_tokens,
                    input_tokens_after=capped_input_tokens,
                    token_cap=token_cap,
                    env_var="PAPERBENCH_JUDGE_INPUT_TOKEN_CAP",
                )
            input_tokens = capped_input_tokens
        else:
            input_tokens = _messages_token_count(converted_messages, self.token_encoder)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }
        if not _not_given(self.reasoning_effort) and self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if not _not_given(self.response_format) and not _env_flag(
            "PAPERBENCH_RESPONSES_DISABLE_RESPONSE_FORMAT"
        ):
            payload["response_format"] = _chat_json_schema(self.response_format)
        elif not _not_given(self.response_format):
            logger.warning(
                "Skipping chat completion response_format because it is disabled",
                call_id=call_id,
                env_var="PAPERBENCH_RESPONSES_DISABLE_RESPONSE_FORMAT",
            )
        if not _not_given(self.temperature):
            payload["temperature"] = self.temperature
        if not _not_given(self.max_tokens):
            payload["max_tokens"] = self.max_tokens
        if not _not_given(self.top_p):
            payload["top_p"] = self.top_p

        payload_json = json.dumps(payload)
        payload_bytes = len(payload_json.encode("utf-8"))

        semaphore = await _get_semaphore(max(1, self.concurrency))
        async with semaphore:
            retrying = self.retry_config.build()
            retrying.retry = retrying.retry | tenacity.retry_if_exception_type(
                (RetryableHTTPError, httpx.TimeoutException, httpx.TransportError)
            )
            async for attempt in retrying:
                with attempt:
                    start = time.monotonic()
                    logger.info(
                        "Chat completions call started",
                        call_id=call_id,
                        attempt=attempt.retry_state.attempt_number,
                        model=self.model,
                        base_url=self.base_url,
                        input_messages=len(conversation),
                        input_tokens=input_tokens,
                        payload_bytes=payload_bytes,
                        n_ctx=self.n_ctx,
                    )
                    last_retryable_error: Exception | None = None
                    for base_url_index, base_url in enumerate(self.base_urls, start=1):
                        try:
                            async with httpx.AsyncClient(timeout=self.timeout) as client:
                                response = await asyncio.wait_for(
                                    client.post(
                                        f"{base_url}/chat/completions",
                                        headers=_request_headers(self.api_key_env),
                                        content=payload_json,
                                    ),
                                    timeout=self.timeout + 30.0,
                                )
                        except (httpx.TimeoutException, httpx.TransportError) as exc:
                            last_retryable_error = exc
                            if base_url_index < len(self.base_urls):
                                logger.warning(
                                    "Chat completions endpoint transport failure; trying fallback base URL",
                                    call_id=call_id,
                                    attempt=attempt.retry_state.attempt_number,
                                    base_url=base_url,
                                    fallback_base_url=self.base_urls[base_url_index],
                                    error=repr(exc),
                                )
                                continue
                            raise
                        except asyncio.TimeoutError as exc:
                            last_retryable_error = httpx.ReadTimeout(
                                f"Timed out waiting for {base_url}/chat/completions after {self.timeout + 30.0:.0f}s"
                            )
                            if base_url_index < len(self.base_urls):
                                logger.warning(
                                    "Chat completions endpoint hard timeout; trying fallback base URL",
                                    call_id=call_id,
                                    attempt=attempt.retry_state.attempt_number,
                                    base_url=base_url,
                                    fallback_base_url=self.base_urls[base_url_index],
                                    error=repr(exc),
                                )
                                continue
                            raise last_retryable_error

                        elapsed = time.monotonic() - start
                        logger.info(
                            "Chat completions call returned",
                            call_id=call_id,
                            attempt=attempt.retry_state.attempt_number,
                            base_url=base_url,
                            status_code=response.status_code,
                            elapsed_seconds=round(elapsed, 2),
                            content_type=response.headers.get("content-type", ""),
                            response_bytes=len(response.content),
                        )

                        retryable_forbidden = (
                            response.status_code == 403
                            and "quota_exceeded" in response.text
                        )
                        if (
                            response.status_code in {408, 409, 429}
                            or retryable_forbidden
                            or response.status_code >= 500
                        ):
                            last_retryable_error = RetryableHTTPError(
                                f"Retryable HTTP {response.status_code} from {base_url}/chat/completions: "
                                f"{response.text[:500]}"
                            )
                            if base_url_index < len(self.base_urls):
                                logger.warning(
                                    "Chat completions endpoint returned retryable status; trying fallback base URL",
                                    call_id=call_id,
                                    attempt=attempt.retry_state.attempt_number,
                                    base_url=base_url,
                                    fallback_base_url=self.base_urls[base_url_index],
                                    status_code=response.status_code,
                                )
                                continue
                            raise last_retryable_error

                        if response.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {response.status_code} from {base_url}/chat/completions: "
                                f"{response.text[:500]}"
                            )

                        response_json = response.json()
                        if response_json.get("error"):
                            raise RuntimeError(response_json["error"])
                        content = _extract_chat_output_text(response_json)
                        usage = _usage_from_chat_response(response_json)
                        if not content.strip() and usage and usage.completion_tokens > 0:
                            last_retryable_error = RetryableHTTPError(
                                "Chat completions response had empty assistant content "
                                f"after {usage.completion_tokens} completion tokens "
                                f"(finish_reason={_chat_finish_reason(response_json)!r})"
                            )
                            if base_url_index < len(self.base_urls):
                                logger.warning(
                                    "Chat completions endpoint returned empty content; trying fallback base URL",
                                    call_id=call_id,
                                    attempt=attempt.retry_state.attempt_number,
                                    base_url=base_url,
                                    fallback_base_url=self.base_urls[base_url_index],
                                    completion_tokens=usage.completion_tokens,
                                    finish_reason=_chat_finish_reason(response_json),
                                )
                                continue
                            raise last_retryable_error
                        break
                    else:
                        if last_retryable_error is not None:
                            raise last_retryable_error
                        raise RuntimeError("No API base URL succeeded")

        content = _extract_chat_output_text(response_json)
        return HTTPResponsesTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[ChatCompletionMessage(role="assistant", content=content)],
            usage=_usage_from_chat_response(response_json),
        )
