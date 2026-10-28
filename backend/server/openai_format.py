"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        if m.role == "assistant" and getattr(m, "reasoning_content", None):
            lines.append(f"Assistant (reasoning): {m.reasoning_content}")
        lines.append(f"{label}: {_text_of(m.content)}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None, reasoning: str = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    message = {"role": "assistant", "content": content}
    if reasoning:
        # Non-standard extension, mirroring DeepSeek's own `reasoning_content`.
        message["reasoning_content"] = reasoning
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(model: str, stream, thinking: bool = False) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.
    `stream` is the client's stream object; after it's consumed we read its
    `.conversation_id` and attach it to the final chunk. When `thinking` is on and
    the client exposes typed events, reasoning is streamed as a non-standard
    `delta.reasoning_content` field (the same field DeepSeek's API uses).
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})
    iter_events = getattr(stream, "iter_events", None)
    if callable(iter_events):
        for kind, text in iter_events():
            if not text:
                continue
            if kind == "reasoning":
                if thinking:
                    yield frame({"reasoning_content": text})
            else:
                yield frame({"content": text})
    else:
        for d in stream:
            if d:
                yield frame({"content": d})
    conversation_id = getattr(stream, "conversation_id", None)
    extra = {"conversation_id": conversation_id}
    reasoning = getattr(stream, "reasoning", None)
    if thinking and reasoning:
        extra["reasoning_content"] = reasoning
    yield frame({}, finish="stop", extra=extra)
    yield "data: [DONE]\n\n"


def error_frame(message: str, err_type: str = "upstream_error",
                model: str = None) -> str:
    """A terminal SSE chunk that carries an error to the client.

    Emitted instead of crashing the ASGI task when the upstream call raises
    mid-stream. The object stays a valid ``chat.completion.chunk`` (one choice,
    an empty ``delta`` and ``finish_reason`` ``"error"``) so streaming parsers do
    not choke; the top-level ``error`` object is a non-standard extension the
    client reads to surface the failure. Always followed by ``data: [DONE]``.
    """
    obj = {
        "id": _id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        "error": {"message": message, "type": err_type},
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
