from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _dedupe_base_urls(primary: str, alternates: list[str] | None = None) -> list[str]:
    urls: list[str] = []
    for raw_url in [primary, *(alternates or [])]:
        url = _normalize_base_url(str(raw_url or "").strip())
        if url and url not in urls:
            urls.append(url)
    return urls


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            if "text" in part and part.get("text") is not None:
                parts.append(str(part["text"]))
                continue
            if "content" in part and part.get("content") is not None:
                parts.append(str(part["content"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_message_text(content_parts: Any) -> str:
    if isinstance(content_parts, str):
        return content_parts
    if not isinstance(content_parts, list):
        return _extract_text_from_content(content_parts)
    parts: list[str] = []
    for part in content_parts:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append(str(part))
            continue
        text = part.get("text")
        if text is not None:
            parts.append(str(text))
            continue
        content = part.get("content")
        if content is not None:
            parts.append(str(content))
    return "\n".join(part for part in parts if part)


def _stringify_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _chat_tools_from_responses_tools(tools: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "function"),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _chat_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str) and tool_choice in {"auto", "required", "none"}:
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function_name = tool_choice.get("name")
            if not function_name and isinstance(tool_choice.get("function"), dict):
                function_name = tool_choice["function"].get("name")
            if function_name:
                return {
                    "type": "function",
                    "function": {
                        "name": str(function_name),
                    },
                }
    return "auto"


def _messages_from_responses_request(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = str(payload.get("instructions") or "").strip()
    if instructions:
        messages.append({"role": "system", "content": instructions})

    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()

    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "message":
            flush_pending_tool_calls()
            role = str(item.get("role") or "user")
            if role in {"developer", "system"}:
                role = "system"
            elif role not in {"user", "assistant", "tool"}:
                role = "user"
            messages.append(
                {
                    "role": role,
                    "content": _extract_message_text(item.get("content")),
                }
            )
            continue
        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or f"call_{uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or "function"),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
            continue
        if item_type == "function_call_output":
            flush_pending_tool_calls()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _stringify_output(item.get("output")),
                }
            )
            continue

    flush_pending_tool_calls()
    return messages


def _extract_chat_response_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            if part.get("type") in {"text", "output_text"}:
                parts.append(str(part.get("text") or ""))
                continue
            if "text" in part:
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return _extract_text_from_content(content)


def _usage_from_chat_response(chat_response: dict[str, Any]) -> dict[str, Any]:
    usage = chat_response.get("usage")
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        details = {"reasoning_tokens": 0}
    elif "reasoning_tokens" not in details:
        details = dict(details)
        details["reasoning_tokens"] = 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_details": details,
        "total_tokens": total_tokens,
    }


def _responses_output_from_chat_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    output_items: list[dict[str, Any]] = []
    text = _extract_chat_response_text(message)
    if text:
        output_items.append(
            {
                "id": f"msg_{uuid4().hex}",
                "status": "completed",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            output_items.append(
                {
                    "id": f"fc_{uuid4().hex}",
                    "status": "completed",
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or f"call_{uuid4().hex}"),
                    "name": str(function.get("name") or "function"),
                    "arguments": str(function.get("arguments") or "{}"),
                }
            )
    return output_items


def _responses_payload_from_chat(
    request_payload: dict[str, Any],
    chat_response: dict[str, Any],
) -> dict[str, Any]:
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Upstream chat/completions response had no choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Upstream choice entry was not an object.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Upstream choice message was missing.")

    output_items = _responses_output_from_chat_message(message)
    if not output_items:
        output_items = [
            {
                "id": f"msg_{uuid4().hex}",
                "status": "completed",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    }
                ],
            }
        ]

    now = int(time.time())
    response_id = str(chat_response.get("id") or f"resp_{uuid4().hex}")
    tool_choice = request_payload.get("tool_choice")
    if tool_choice is None:
        tool_choice = "auto"
    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "input": [],
        "instructions": None,
        "max_output_tokens": request_payload.get("max_output_tokens"),
        "model": str(chat_response.get("model") or request_payload.get("model") or ""),
        "output": output_items,
        "parallel_tool_calls": bool(request_payload.get("parallel_tool_calls", False)),
        "previous_response_id": request_payload.get("previous_response_id"),
        "reasoning": {"effort": None, "summary": None},
        "store": bool(request_payload.get("store", False)),
        "temperature": 1,
        "text": {"format": {"type": "text"}},
        "tool_choice": tool_choice,
        "tools": [],
        "top_p": 1,
        "truncation": "disabled",
        "usage": _usage_from_chat_response(chat_response),
        "user": None,
        "metadata": {},
    }


def _response_in_progress_snapshot(response_payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(response_payload)
    snapshot["status"] = "in_progress"
    snapshot["completed_at"] = None
    snapshot["output"] = []
    return snapshot


class ChatResponsesBridge:
    def __init__(
        self,
        *,
        upstream_base_url: str,
        alternate_base_urls: list[str] | None = None,
        upstream_api_key: str | None = None,
        timeout_seconds: float = 1800.0,
        max_retries: int = 8,
        retry_initial_sleep_seconds: float = 5.0,
        retry_max_sleep_seconds: float = 120.0,
    ):
        self.upstream_base_urls = _dedupe_base_urls(upstream_base_url, alternate_base_urls)
        if not self.upstream_base_urls:
            raise ValueError("At least one upstream base URL is required.")
        self.upstream_base_url = self.upstream_base_urls[0]
        self.upstream_api_key = str(upstream_api_key or "").strip() or None
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.retry_initial_sleep_seconds = max(0.0, float(retry_initial_sleep_seconds))
        self.retry_max_sleep_seconds = max(
            self.retry_initial_sleep_seconds,
            float(retry_max_sleep_seconds),
        )

    @staticmethod
    def _retry_after_seconds(headers: Any) -> float | None:
        try:
            raw = headers.get("Retry-After")
        except Exception:
            raw = None
        if raw is None:
            return None
        try:
            return max(0.0, float(str(raw).strip()))
        except ValueError:
            return None

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code in {408, 409, 429} or status_code >= 500

    @staticmethod
    def _should_try_alternate_http_status(status_code: int) -> bool:
        return status_code in {402, 408, 409, 429} or status_code >= 500

    def chat_completion(self, request_payload: dict[str, Any], incoming_headers: dict[str, str]) -> dict[str, Any]:
        chat_payload: dict[str, Any] = {
            "model": str(request_payload.get("model") or ""),
            "messages": _messages_from_responses_request(request_payload),
            "stream": False,
        }

        tools = _chat_tools_from_responses_tools(request_payload.get("tools"))
        if tools:
            chat_payload["tools"] = tools

        tool_choice = _chat_tool_choice(request_payload.get("tool_choice"))
        if tool_choice is not None:
            chat_payload["tool_choice"] = tool_choice

        if "parallel_tool_calls" in request_payload:
            chat_payload["parallel_tool_calls"] = bool(request_payload.get("parallel_tool_calls"))

        body = json.dumps(chat_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        }
        authorization = None
        if self.upstream_api_key:
            authorization = f"Bearer {self.upstream_api_key}"
        else:
            authorization = incoming_headers.get("Authorization") or incoming_headers.get("authorization")
        if authorization:
            headers["Authorization"] = authorization

        deadline = time.monotonic() + self.timeout_seconds
        last_error: Exception | None = None
        last_url = f"{self.upstream_base_url}/chat/completions"

        for attempt in range(1, self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            retry_after: float | None = None
            should_sleep_before_retry = False

            for base_url_index, base_url in enumerate(self.upstream_base_urls):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                url = f"{base_url}/chat/completions"
                last_url = url
                request = urllib.request.Request(
                    url=url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=max(1.0, remaining)) as response:
                        raw = response.read()
                    return json.loads(raw.decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    status_code = int(exc.code)
                    has_alternate = base_url_index + 1 < len(self.upstream_base_urls)
                    if has_alternate and self._should_try_alternate_http_status(status_code):
                        sys.stdout.write(
                            "[chat-bridge] upstream "
                            f"{base_url} returned HTTP {status_code}; trying alternate base URL\n"
                        )
                        sys.stdout.flush()
                        continue
                    if not self._is_retryable_http_status(status_code) or attempt >= self.max_retries:
                        raise
                    retry_after = self._retry_after_seconds(exc.headers)
                    should_sleep_before_retry = True
                    break
                except Exception as exc:
                    last_error = exc
                    has_alternate = base_url_index + 1 < len(self.upstream_base_urls)
                    if has_alternate:
                        sys.stdout.write(
                            "[chat-bridge] upstream "
                            f"{base_url} failed with {type(exc).__name__}; trying alternate base URL\n"
                        )
                        sys.stdout.flush()
                        continue
                    if attempt >= self.max_retries:
                        raise
                    retry_after = None
                    should_sleep_before_retry = True
                    break

            if not should_sleep_before_retry:
                continue

            sleep_seconds = (
                retry_after
                if retry_after is not None
                else min(
                    self.retry_max_sleep_seconds,
                    self.retry_initial_sleep_seconds * (2 ** (attempt - 1)),
                )
            )
            sleep_seconds = min(max(0.0, sleep_seconds), max(0.0, deadline - time.monotonic()))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if last_error is not None:
            raise last_error
        raise TimeoutError(f"Timed out waiting for {last_url}")


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    bridge: ChatResponsesBridge

    def do_GET(self) -> None:
        if self.path == "/healthz":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/responses":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        raw_body = self.rfile.read(content_length)
        try:
            request_payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON body")
            return

        try:
            chat_response = self.bridge.chat_completion(request_payload, dict(self.headers))
            response_payload = _responses_payload_from_chat(request_payload, chat_response)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header(
                "Content-Type",
                exc.headers.get_content_type() if exc.headers else "text/plain; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if bool(request_payload.get("stream", True)):
            self._write_sse_response(response_payload)
            return

        body = json.dumps(response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse_response(self, response_payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        sequence_number = 1

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            nonlocal sequence_number
            envelope = {"type": event_type, **payload, "sequence_number": sequence_number}
            sequence_number += 1
            chunk = f"event: {event_type}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        emit("response.created", {"response": _response_in_progress_snapshot(response_payload)})
        emit("response.in_progress", {"response": _response_in_progress_snapshot(response_payload)})

        output_items = response_payload.get("output") or []
        for output_index, item in enumerate(output_items):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "message":
                message_id = str(item.get("id") or f"msg_{uuid4().hex}")
                in_progress_item = {
                    "id": message_id,
                    "status": "in_progress",
                    "type": "message",
                    "role": str(item.get("role") or "assistant"),
                    "content": [],
                }
                emit("response.output_item.added", {"output_index": output_index, "item": in_progress_item})
                content_list = item.get("content") or []
                first_content = content_list[0] if isinstance(content_list, list) and content_list else {}
                text = str(first_content.get("text") or "")
                part = {"type": "output_text", "text": "", "annotations": []}
                emit(
                    "response.content_part.added",
                    {
                        "item_id": message_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": part,
                    },
                )
                if text:
                    emit(
                        "response.output_text.delta",
                        {
                            "item_id": message_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
                emit(
                    "response.output_text.done",
                    {
                        "item_id": message_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                    },
                )
                emit(
                    "response.content_part.done",
                    {
                        "item_id": message_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        },
                    },
                )
                emit("response.output_item.done", {"output_index": output_index, "item": item})
                continue

            if item_type == "function_call":
                function_call_id = str(item.get("id") or f"fc_{uuid4().hex}")
                arguments = str(item.get("arguments") or "{}")
                in_progress_item = {
                    "id": function_call_id,
                    "status": "in_progress",
                    "type": "function_call",
                    "call_id": str(item.get("call_id") or f"call_{uuid4().hex}"),
                    "name": str(item.get("name") or "function"),
                    "arguments": "",
                }
                emit("response.output_item.added", {"output_index": output_index, "item": in_progress_item})
                if arguments:
                    emit(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": function_call_id,
                            "output_index": output_index,
                            "delta": arguments,
                        },
                    )
                emit(
                    "response.function_call_arguments.done",
                    {
                        "item_id": function_call_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    },
                )
                emit("response.output_item.done", {"output_index": output_index, "item": item})

        emit("response.completed", {"response": response_payload})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(
            "[chat-bridge] "
            + format % args
            + "\n"
        )
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    listen_host = str(config.get("listen_host") or "127.0.0.1")
    listen_port = int(config.get("listen_port") or 18911)
    upstream_base_url = str(config.get("upstream_base_url") or "").strip()
    raw_alternate_base_urls = config.get("alternate_base_urls") or []
    if isinstance(raw_alternate_base_urls, str):
        alternate_base_urls = [
            item.strip()
            for item in raw_alternate_base_urls.replace("\n", ",").split(",")
            if item.strip()
        ]
    elif isinstance(raw_alternate_base_urls, list):
        alternate_base_urls = [str(item).strip() for item in raw_alternate_base_urls if str(item).strip()]
    else:
        alternate_base_urls = []
    timeout_seconds = float(config.get("timeout_seconds") or 1800.0)
    upstream_api_key = str(config.get("upstream_api_key") or "").strip() or None
    max_retries = int(config.get("max_retries") or 8)
    retry_initial_sleep_seconds = float(config.get("retry_initial_sleep_seconds") or 5.0)
    retry_max_sleep_seconds = float(config.get("retry_max_sleep_seconds") or 120.0)
    if not upstream_base_url:
        raise SystemExit("Missing upstream_base_url in config.")

    handler = type(
        "ConfiguredBridgeHandler",
        (BridgeHandler,),
        {
            "bridge": ChatResponsesBridge(
                upstream_base_url=upstream_base_url,
                alternate_base_urls=alternate_base_urls,
                upstream_api_key=upstream_api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_initial_sleep_seconds=retry_initial_sleep_seconds,
                retry_max_sleep_seconds=retry_max_sleep_seconds,
            )
        },
    )
    server = ThreadingHTTPServer((listen_host, listen_port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
