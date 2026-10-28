"""OpenAI Responses API adapter for the DeepSeek web chat.

Codex speaks the Responses API.  The DeepSeek web backend only accepts a single
prompt string, so this module:

* flattens Responses ``input`` items back into one role-labelled transcript;
* renders Codex's tool definitions as a prompted tool protocol;
* parses ``<tool_call>`` JSON envelopes and DeepSeek DSML calls;
* streams the Responses SSE events Codex understands.

The bridge is stateless: every request carries the full transcript, which also
makes reasoning replay explicit and robust across retries.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired

from .openai_format import MOCK_TITLE_JSON, is_title_request

from .toolcall import (
    StreamParser,
    extract_thinking,
    looks_like_tool_syntax,
    parse_tool_calls,
    strip_channel_tags,
    strip_tool_results,
)

log = logging.getLogger("uvicorn.error")

EMPTY_MAX_ATTEMPTS = 3
MAX_TOOL_CALLS = 16
MAX_TOOL_PROMPT_CHARS = 240_000
MAX_DESCRIPTION_CHARS = 4_000

BASE_PROMPT = """You are mike, an autonomous coding agent running locally on the user's machine.
You are powered by DeepSeek V4.1 Flash through a local bridge. You DO things with tools instead of explaining how.

Rules:
1. Read a file before editing it. Use the apply_patch tool for every file edit.
2. For multi-step tasks, use update_plan before starting and keep each step's status current.
3. Keep calling tools until the task is genuinely complete; don't stop after one step.
4. Never print raw JSON tool schemas or fake tool output to the user.
5. Use absolute or repo-relative paths. Never guess - use the provided file tools to find things.
6. After you finish, give a one-line summary of what changed and how to run/test it.

Before calling tools, write a short 1-2 sentence progress update as standard text.
Do not use [commentary] or [final] tags. When the task is fully complete, call task_complete.
"""

TOOL_PROTOCOL = """## Tool protocol

You can call tools. To call one, emit one or more blocks in EXACTLY this format:

<tool_call>
{"name": "<tool_name>", "arguments": { ... }}
</tool_call>

Rules for tool calls:
- Emit ONLY the envelope for the call; put any message to the user OUTSIDE the blocks.
- You may emit several <tool_call> blocks in one turn.
- Arguments must be a JSON object matching the tool's parameters.
- After the tools run you receive a <tool_result> message; keep going until the task is done.
- For multi-step tasks, call update_plan first and keep each step's status current.
- For file edits, ALWAYS use apply_patch; never use shell redirection, cat, or sed -i.
- Use ONLY this envelope for tools. Do NOT use DSML tags, bare function names, or fenced JSON.

Freeform tools (for example apply_patch) take one raw string argument: put it in
{"input": "<raw tool input>"}. apply_patch input is a Codex patch envelope:

*** Begin Patch
*** Add File: notes.txt
+hello
*** Update File: app.py
@@
-old
+new
*** End Patch

Available tools:
"""

COMPLETION_NUDGE = (
    "You have not called `task_complete`. Continue the task or call "
    "`task_complete` with a summary."
)
MALFORMED_NUDGE = (
    "Your last message contained tool syntax but no executable tool call. "
    "Emit a single valid <tool_call> envelope or a DSML invoke block now."
)

# Codex sends its native-tool system prompt (base instructions, permissions,
# collaboration mode, ...) as developer messages. Replaying it beside the
# prompted JSON protocol teaches DeepSeek a tool syntax it cannot call, so the
# model imitates Codex's native prompts (e.g. hallucinated DSML) instead of
# emitting <tool_call> envelopes. These fragments are stripped before the
# transcript is flattened; small project context (environment, user config)
# survives.
_NOISY_DEVELOPER_TAGS = (
    "permissions instructions",
    "collaboration_mode",
    "model_switch",
    "managed_developer_instructions",
    "persistent_mode",
    "apps_instructions",
    "plugins_instructions",
    "skills_instructions",
    "multi_agent_mode",
    "multi_agent_role",
    "environments_instructions",
    "git_attribution",
    "realtime_conversation",
    "personality_spec",
    "token_budget",
    "rollout_budget",
    "context_window",
)
_NATIVE_PROMPT_PREFIXES = (
    "You are Codex",
    "You are an AI assistant accessed via an API",
)
MAX_DEVELOPER_CONTEXT_CHARS = 6_000


def _clean_developer_text(text: str) -> str:
    """Strip Codex's native-tool prompt, keeping concise project context."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for tag in _NOISY_DEVELOPER_TAGS:
        pattern = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    low = cleaned.lower()
    if any(f"<{tag}>" in low for tag in _NOISY_DEVELOPER_TAGS):
        # Unclosed noisy block: do not risk leaking the rest of the prompt.
        return ""
    cleaned = cleaned.strip()
    if cleaned.startswith(_NATIVE_PROMPT_PREFIXES):
        return ""
    if len(cleaned) > MAX_DEVELOPER_CONTEXT_CHARS:
        return ""
    return cleaned


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _text_of_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(p for p in parts if p)


def _text_of_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    return _text_of_content(output)


def _reasoning_text(item: dict) -> str:
    parts: list[str] = []
    for key in ("content", "summary"):
        for part in item.get(key) or []:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n".join(parts)


def _prepare_tools(raw_tools: Any) -> tuple[list[dict], dict[str, dict]]:
    """Normalise Responses tools into prompt specs and a lookup map."""
    specs: list[dict] = []
    lookup: dict[str, dict] = {}
    seen: set[str] = set()

    def add(spec: dict) -> None:
        name = spec["name"]
        if not name or name in seen:
            return
        seen.add(name)
        specs.append(spec)
        lookup[name] = spec
        lookup[name.lower()] = spec

    unsupported_names = {"view_image", "web_search", "tool_search"}

    def walk(tool: Any) -> None:
        if not isinstance(tool, dict):
            return
        kind = tool.get("type")
        if kind == "function":
            name = str(tool.get("name") or "").strip()
            if not name or name in unsupported_names:
                return
            add(
                {
                    "name": name,
                    "kind": "function",
                    "description": str(tool.get("description") or "").strip(),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        elif kind == "custom":
            name = str(tool.get("name") or "").strip()
            if not name:
                return
            add(
                {
                    "name": name,
                    "kind": "custom",
                    "description": str(tool.get("description") or "").strip(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Complete raw input for this freeform tool.",
                            }
                        },
                        "required": ["input"],
                    },
                }
            )
        elif kind == "namespace":
            # Namespaced tools (multi-agent, etc.) are not part of the portable
            # Mike surface; flattening them would teach calls Codex can't route
            # reliably through this bridge.
            log.debug("responses: skipping namespaced tool set %s", tool.get("name"))
        elif kind in ("web_search", "tool_search"):
            log.debug("responses: skipping unsupported hosted tool %s", kind)
        # Unknown tool kinds are ignored rather than taught incorrectly.

    for tool in raw_tools or []:
        walk(tool)
    return specs, lookup


def _tool_prompt(specs: list[dict]) -> str:
    lines = [TOOL_PROTOCOL]
    for spec in specs:
        description = spec["description"] or "(no description)"
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS] + " ..."
        lines.append(f"- {spec['name']}: {description}")
        if spec["kind"] == "custom":
            lines.append(
                '  freeform tool; arguments: {"input": "<raw tool input>"}'
            )
        else:
            schema = json.dumps(spec["parameters"], ensure_ascii=False, separators=(",", ":"))
            lines.append(f"  parameters: {schema}")
    lines.append(
        '- task_complete: Call this ONLY when the user\'s task is fully done. '
        'arguments: {"summary": "one-paragraph summary of what was accomplished"}'
    )
    text = "\n".join(lines)
    if len(text) > MAX_TOOL_PROMPT_CHARS:
        log.warning("responses: tool prompt truncated (%d chars)", len(text))
        text = text[:MAX_TOOL_PROMPT_CHARS]
    return text


def _normalise_name(name: str, lookup: dict[str, dict]) -> str | None:
    raw = (name or "").strip()
    candidates = [
        raw,
        raw.rsplit(".", 1)[-1],
        raw.removeprefix("functions."),
        raw.replace("-", "_"),
    ]
    for candidate in candidates:
        if candidate in lookup:
            return candidate
    lowered = {key.lower(): key for key in lookup if key == key.lower()}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    return None


def _custom_input(args: dict) -> str:
    if "input" in args:
        value = args["input"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if "patch" in args:
        value = args["patch"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(args) == 1:
        value = next(iter(args.values()))
        if isinstance(value, str):
            return value
    return json.dumps(args, ensure_ascii=False)


def _input_items(body: dict) -> list[Any]:
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        return [{"type": "message", "role": "user", "content": raw_input}]
    if isinstance(raw_input, list):
        return raw_input
    return []


def _build_history(body: dict) -> tuple[list[str], list[str], dict[str, str], str]:
    """Return (developer_parts, dialogue, call_names, last_action)."""
    items = _input_items(body)

    dev_parts: list[str] = []
    dialogue: list[str] = []
    call_names: dict[str, str] = {}
    pending_reasoning: list[str] = []
    last_action = "user"

    def take_reasoning_prefix() -> str:
        if not pending_reasoning:
            return ""
        text = "\n".join(pending_reasoning).strip()
        pending_reasoning.clear()
        if not text:
            return ""
        return f"<reasoning>\n{text}\n</reasoning>\n"

    instructions = body.get("instructions")
    if isinstance(instructions, str):
        cleaned_instructions = _clean_developer_text(instructions)
        if cleaned_instructions:
            dev_parts.append(cleaned_instructions)

    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "message":
            role = str(item.get("role") or "user").lower()
            text = _text_of_content(item.get("content")).strip()
            if role in ("developer", "system"):
                cleaned = _clean_developer_text(text)
                if cleaned:
                    dev_parts.append(cleaned)
                continue
            if role == "user":
                if text:
                    if text.lstrip().startswith("<environment_context>"):
                        dev_parts.append(text)
                        continue
                    dialogue.append(f"User: {text}")
                    last_action = "user"
                continue
            if role == "assistant":
                if text:
                    prefix = take_reasoning_prefix()
                    dialogue.append(f"Assistant: {prefix}{text}")
                    last_action = "assistant"
                continue
            if role == "tool":
                call_id = str(item.get("call_id") or item.get("tool_call_id") or "tool")
                dialogue.append(
                    f'User:\n<tool_result name="tool" id="{call_id}">\n{text}\n</tool_result>'
                )
                last_action = "tool_output"
                continue
            continue

        if kind == "reasoning":
            text = _reasoning_text(item)
            if text:
                pending_reasoning.append(text)
            continue

        if kind in ("function_call", "custom_tool_call"):
            name = str(item.get("name") or "")
            call_id = str(item.get("call_id") or "")
            if call_id:
                call_names[call_id] = name
            prefix = take_reasoning_prefix()
            if kind == "function_call":
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except (json.JSONDecodeError, ValueError):
                        arguments = {"input": arguments}
                if not isinstance(arguments, dict):
                    arguments = {}
            else:
                raw_input = item.get("input")
                arguments = {"input": raw_input if isinstance(raw_input, str) else ""}
            envelope = json.dumps(
                {"name": name, "arguments": arguments}, ensure_ascii=False
            )
            dialogue.append(
                f"Assistant: {prefix}<tool_call>\n{envelope}\n</tool_call>"
            )
            last_action = "tool_call"
            continue

        if kind in (
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
        ):
            call_id = str(item.get("call_id") or "")
            name = str(item.get("name") or call_names.get(call_id) or "tool")
            output = _text_of_output(item.get("output"))
            dialogue.append(
                f'User:\n<tool_result name="{name}" id="{call_id}">\n{output}\n</tool_result>'
            )
            last_action = "tool_output"
            continue

        if kind in (
            "compaction",
            "context_compaction",
            "additional_tools",
            "compaction_trigger",
            "configuration_update",
        ):
            continue
        # Unknown item types are safely omitted from the flat transcript.

    return dev_parts, dialogue, call_names, last_action


def _build_prompt(body: dict, specs: list[dict]) -> tuple[str, str]:
    dev_parts, dialogue, _, last_action = _build_history(body)
    parts = [BASE_PROMPT.strip()]
    if dev_parts:
        parts.append("## Developer context\n\n" + "\n\n".join(dev_parts))
    if specs:
        parts.append(_tool_prompt(specs))
    conversation = "\n\n".join(dialogue) if dialogue else "(no messages)"
    parts.append("## Conversation\n\n" + conversation)
    return "\n\n".join(parts).strip() + "\n\nAssistant:", last_action


def _is_assistant_output(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    kind = item.get("type") or ("message" if "role" in item else "")
    if kind == "message":
        return str(item.get("role") or "").lower() == "assistant"
    return kind in (
        "reasoning",
        "function_call",
        "custom_tool_call",
        "local_shell_call",
    )


def _build_continuation_prompt(body: dict) -> tuple[str, str] | None:
    """Build the prompt for a resumed upstream thread.

    Codex resends the full transcript on every request, but the DeepSeek chat
    already holds everything up to the last assistant output.  Send only the
    messages that follow it (a new user turn, tool results, or both).
    """
    items = _input_items(body)
    start = 0
    for index, item in enumerate(items):
        if _is_assistant_output(item):
            start = index + 1
    if start == 0:
        # No assistant output in the transcript: this is the first request, or
        # the client's history does not match an existing upstream thread.
        return None
    tail = [item for item in items[start:] if isinstance(item, dict)]
    if not tail:
        return None
    _, dialogue, _, last_action = _build_history({"input": tail})
    if not dialogue:
        return None
    return "\n\n".join(dialogue) + "\n\nAssistant:", last_action




def _nudged_prompt(prompt: str, raw_response: str, nudge: str) -> str:
    return (
        prompt
        + "\n"
        + raw_response.strip()
        + "\n\nUser: "
        + nudge
        + "\n\nAssistant:"
    )


def _normalize_calls(
    raw_calls: Iterable[dict], lookup: dict[str, dict]
) -> tuple[list[dict], str | None]:
    normalised: list[dict] = []
    summary: str | None = None
    for call in list(raw_calls)[:MAX_TOOL_CALLS]:
        name = str(call.get("name") or "").strip()
        args = call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        if name.lower() == "task_complete":
            summary = str(args.get("summary") or "").strip() or summary
            continue
        resolved = _normalise_name(name, lookup)
        if resolved is None:
            log.warning("responses: dropping unknown tool call %r", name)
            continue
        spec = lookup[resolved]
        if spec["kind"] == "custom":
            normalised.append(
                {"kind": "custom", "name": resolved, "input": _custom_input(args)}
            )
        else:
            normalised.append(
                {
                    "kind": "function",
                    "name": resolved,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
    return normalised, summary


def _request_thinking(body: dict) -> bool:
    if "thinking" in body:
        return bool(body.get("thinking"))
    # The client's reasoning level doubles as the DeepThink toggle: any level
    # above `none` turns reasoning on.
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip():
            return effort.strip().lower() not in ("none", "minimal", "")
    return _env_bool("MIKE_BRIDGE_THINKING", False)


def _is_title_generation(body: dict) -> bool:
    texts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        texts.append(instructions)
    for item in _input_items(body):
        if isinstance(item, dict):
            texts.append(_text_of_content(item.get("content")))
    return is_title_request(texts)


def _mocked_title_events(response_id: str, prompt: str) -> Iterable[str]:
    """Answer Codex's title request locally so the upstream chat stays clean."""
    item_id = _new_id("msg")
    log.info("responses: answering title request locally")
    yield _sse({"type": "response.created", "response": {"id": response_id}})
    yield _begin_message(item_id)
    yield _sse(
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "delta": MOCK_TITLE_JSON,
            "content_index": 0,
        }
    )
    yield _sse(
        {
            "type": "response.output_item.done",
            "item": _message_item(item_id, MOCK_TITLE_JSON),
        }
    )
    yield _emit_completed(response_id, prompt, MOCK_TITLE_JSON, "")



def _emit_reasoning(reasoning: str) -> Iterable[str]:
    if not reasoning.strip():
        return
    item_id = _new_id("rs")
    item = {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": reasoning}],
        "encrypted_content": None,
    }
    yield _sse({"type": "response.output_item.added", "item": item})
    yield _sse({"type": "response.output_item.done", "item": item})


def _message_item(item_id: str, text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "id": item_id,
        "content": [{"type": "output_text", "text": text}],
    }


def _begin_message(item_id: str) -> str:
    return _sse(
        {
            "type": "response.output_item.added",
            "item": _message_item(item_id, ""),
        }
    )


def _emit_text(
    text: str,
    already_streamed: bool,
    item_id: str,
    added_sent: bool,
) -> Iterable[str]:
    if not text:
        if added_sent:
            yield _sse(
                {
                    "type": "response.output_item.done",
                    "item": _message_item(item_id, ""),
                }
            )
        return
    if not added_sent:
        yield _begin_message(item_id)
    if not already_streamed:
        # Chunk to keep the TUI responsive even for buffered turns.
        step = 800
        for start in range(0, len(text), step):
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "delta": text[start : start + step],
                    "content_index": 0,
                }
            )
    yield _sse(
        {
            "type": "response.output_item.done",
            "item": _message_item(item_id, text),
        }
    )


def _emit_tool_call(call: dict) -> str:
    call_id = _new_id("call")
    if call["kind"] == "custom":
        item = {
            "type": "custom_tool_call",
            "id": _new_id("ctc"),
            "call_id": call_id,
            "name": call["name"],
            "input": call["input"],
        }
    else:
        item = {
            "type": "function_call",
            "id": _new_id("fc"),
            "call_id": call_id,
            "name": call["name"],
            "arguments": call["arguments"],
        }
    return _sse({"type": "response.output_item.done", "item": item})


def _emit_completed(
    response_id: str,
    prompt: str,
    raw: str,
    reasoning: str,
    conversation_id: str | None = None,
) -> str:
    input_tokens = _est_tokens(prompt)
    output_tokens = _est_tokens(raw + reasoning)
    return _sse(
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "end_turn": True,
                "conversation_id": conversation_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {
                        "reasoning_tokens": _est_tokens(reasoning) if reasoning else 0
                    },
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        }
    )


def _emit_failed(response_id: str, message: str, code: str = "upstream_error") -> str:
    return _sse(
        {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "error": {"code": code, "message": message, "type": code},
            },
        }
    )


def _response_events(client, body: dict) -> Iterable[str]:
    response_id = _new_id("resp")
    # The web backend exposes one model; every accepted alias maps to it.
    specs, lookup = _prepare_tools(body.get("tools"))
    full_prompt, last_action = _build_prompt(body, specs)
    raw_cid = body.get("conversation_id")
    incoming_cid = raw_cid.strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
    upstream_cid = incoming_cid
    prompt = full_prompt
    if incoming_cid:
        continuation = _build_continuation_prompt(body)
        if continuation is not None:
            prompt = continuation[0]
            log.info("responses: resuming upstream thread %s", incoming_cid[:8])
        else:
            log.warning(
                "responses: conversation_id supplied without an assistant turn; "
                "starting a fresh upstream thread"
            )
            upstream_cid = None
    else:
        log.info("responses: starting a fresh upstream thread")
    thinking = _request_thinking(body)
    if thinking:
        log.info("responses: DeepThink enabled for this turn")
    live = last_action != "tool_output"
    after_tool = last_action == "tool_output"
    empty_attempts = 0
    malformed_retry = False
    nudge_count = 0
    fresh_fallback_used = not incoming_cid
    current_prompt = prompt
    message_item_id = _new_id("msg")
    message_added = False

    yield _sse({"type": "response.created", "response": {"id": response_id}})

    while True:
        raw_parts: list[str] = []
        reasoning_parts: list[str] = []
        parser = StreamParser()
        stream_error: Exception | None = None
        streamed_text = False
        emitted_any = False
        stream = None

        try:
            stream = client.stream(
                current_prompt,
                conversation_id=upstream_cid,
                model=None if upstream_cid else "default",
                thinking=thinking,
            )
            for kind_chunk in stream.iter_events():
                kind, chunk = kind_chunk
                if not chunk:
                    continue
                emitted_any = True
                if kind == "reasoning":
                    reasoning_parts.append(chunk)
                    continue
                raw_parts.append(chunk)
                if live:
                    for mode, text in parser.feed(chunk):
                        if mode == "content":
                            if not message_added:
                                message_added = True
                                yield _begin_message(message_item_id)
                            streamed_text = True
                            yield _sse(
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": message_item_id,
                                    "delta": text,
                                    "content_index": 0,
                                }
                            )
            if live:
                for mode, text in parser.finish():
                    if mode == "content":
                        if not message_added:
                            message_added = True
                            yield _begin_message(message_item_id)
                        streamed_text = True
                        yield _sse(
                            {
                                "type": "response.output_text.delta",
                                "item_id": message_item_id,
                                "delta": text,
                                "content_index": 0,
                            }
                        )
        except Exception as exc:  # includes auth/session failures
            stream_error = exc
            log.exception("responses: upstream stream failed")

        returned_cid = getattr(stream, "conversation_id", None) if stream else None
        raw = "".join(raw_parts)
        event_reasoning = "".join(reasoning_parts)
        calls, clean = parse_tool_calls(raw)
        tag_reasoning, clean = extract_thinking(clean)
        clean = strip_channel_tags(strip_tool_results(clean)).strip()
        reasoning = (event_reasoning or tag_reasoning or parser.reasoning).strip()

        if stream_error is not None:
            if raw or reasoning or emitted_any:
                message = f"DeepSeek stream failed: {stream_error}"
                yield _emit_failed(response_id, message)
                return
            # An expired or unknown upstream thread must not wedge the client:
            # fall back once to a fresh thread seeded with the full transcript.
            if upstream_cid is not None and not fresh_fallback_used:
                fresh_fallback_used = True
                upstream_cid = None
                current_prompt = full_prompt
                empty_attempts = 0
                log.warning(
                    "responses: failed to resume upstream thread; retrying fresh"
                )
                continue
            empty_attempts += 1
            if empty_attempts < EMPTY_MAX_ATTEMPTS:
                time.sleep(1.0)
                continue
            yield _emit_failed(
                response_id, f"DeepSeek stream failed: {stream_error}"
            )
            return

        if not raw and not reasoning and not calls:
            empty_attempts += 1
            log.warning(
                "responses: empty upstream response (attempt %d/%d)",
                empty_attempts,
                EMPTY_MAX_ATTEMPTS,
            )
            if empty_attempts < EMPTY_MAX_ATTEMPTS:
                time.sleep(1.0)
                if upstream_cid is not None:
                    current_prompt = "(continue)"
                continue
            yield _emit_failed(
                response_id,
                "DeepSeek returned an empty response after retries. "
                "Check the bridge login/session with `python -m deepseek.auth`.",
                code="empty_response",
            )
            return

        if returned_cid:
            upstream_cid = returned_cid
            log.info("responses: upstream thread %s", returned_cid[:8])

        # A model that hallucinates tool syntax but emits no parseable call gets
        # one corrective retry, unless text has already been shown (avoiding a
        # duplicated commentary stream).
        if (
            specs
            and not calls
            and not malformed_retry
            and not streamed_text
            and looks_like_tool_syntax(raw)
        ):
            malformed_retry = True
            current_prompt = (
                MALFORMED_NUDGE
                if upstream_cid
                else _nudged_prompt(prompt, raw, MALFORMED_NUDGE)
            )
            continue

        tool_calls, task_summary = _normalize_calls(calls, lookup)

        if task_summary is not None:
            if task_summary and clean and task_summary not in clean:
                final_text = f"{clean}\n\n{task_summary}"
            else:
                final_text = task_summary or clean
            if reasoning:
                yield from _emit_reasoning(reasoning)
            yield from _emit_text(final_text, streamed_text, message_item_id, message_added)
            yield _emit_completed(
                response_id, full_prompt, raw, reasoning, upstream_cid
            )
            return

        if tool_calls:
            if reasoning:
                yield from _emit_reasoning(reasoning)
            yield from _emit_text(clean, streamed_text, message_item_id, message_added)
            for call in tool_calls:
                yield _emit_tool_call(call)
            yield _emit_completed(
                response_id, full_prompt, raw, reasoning, upstream_cid
            )
            return

        # Text-only reply. After tool output it may be an early stop: give the
        # model exactly one nudge to keep working or finish explicitly.
        if after_tool and nudge_count == 0 and specs:
            nudge_count = 1
            current_prompt = (
                COMPLETION_NUDGE
                if upstream_cid
                else _nudged_prompt(prompt, raw, COMPLETION_NUDGE)
            )
            live = False
            continue

        if reasoning:
            yield from _emit_reasoning(reasoning)
        yield from _emit_text(clean, streamed_text, message_item_id, message_added)
        yield _emit_completed(
            response_id, full_prompt, raw, reasoning, upstream_cid
        )
        return


def build_responses_endpoint(app: FastAPI, get_client) -> None:
    """Register /v1/responses on the existing FastAPI app."""

    @app.post("/v1/responses")
    @app.post("/responses")
    async def responses(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "invalid JSON body", "type": "invalid_request"}},
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "body must be an object", "type": "invalid_request"}},
            )
        if not body.get("input"):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "`input` must not be empty", "type": "invalid_request"}},
            )

        if _is_title_generation(body):
            # Codex titles are generated on a hidden thread; answering locally
            # keeps the upstream DeepSeek chat free of orphan sessions.
            return StreamingResponse(
                _mocked_title_events(_new_id("resp"), body.get("instructions") or ""),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            client = await run_in_threadpool(get_client)
        except LoginRequired as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"DeepSeek login required: {exc}",
                        "type": "login_required",
                    }
                },
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Failed to initialise DeepSeek session: {exc}",
                        "type": "server_error",
                    }
                },
            )

        return StreamingResponse(
            _response_events(client, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
