"""Tests for the Responses API bridge (no browser or network required)."""

from __future__ import annotations

import json

from server import responses
from server.responses import _response_events, _build_prompt, _prepare_tools
from server.toolcall import parse_tool_calls


class FakeStream:
    def __init__(self, events, conversation_id=None):
        self.events = events
        self.conversation_id = conversation_id

    def iter_events(self):
        yield from self.events


class FakeClient:
    def __init__(self, scripts):
        self.scripts = [
            item if isinstance(item, dict) else {"events": list(item)}
            for item in scripts
        ]
        self.prompts: list[str] = []
        self.conversation_ids: list[str | None] = []

    def stream(self, prompt, model=None, thinking=False, conversation_id=None):
        self.prompts.append(prompt)
        self.conversation_ids.append(conversation_id)
        if not self.scripts:
            return FakeStream([("response", "No script left.")])
        script = self.scripts.pop(0)
        return FakeStream(script["events"], script.get("conversation_id"))


def events_of(gen):
    out = []
    for raw in gen:
        for block in raw.split("\n\n"):
            if block.startswith("data: "):
                out.append(json.loads(block[6:]))
    return out


def base_body(text="do the task", tools=None, history=None):
    items = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "You are mike."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context><cwd>/tmp/p</cwd></environment_context>",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    ]
    if history:
        items = history
    return {
        "model": "deepseek-v4.1-flash",
        "input": items,
        "tools": tools or [],
    }


TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "description": "Run a command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {"type": "custom", "name": "apply_patch", "description": "Patch files"},
    {"type": "function", "name": "view_image", "description": "Not supported"},
]


def test_dsml_function_call_streams_added_then_done():
    dsml = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n'
        '<｜DSML｜ parameter name="cmd" string="true">ls</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    client = FakeClient([[("response", "Checking.\n"), ("response", dsml)]])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    types = [event["type"] for event in events]
    assert types.index("response.output_item.added") < types.index(
        "response.output_text.delta"
    )
    calls = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    ]
    assert calls[0]["name"] == "exec_command"
    assert json.loads(calls[0]["arguments"]) == {"cmd": "ls"}
    assert "view_image" not in client.prompts[0]


def test_apply_patch_is_emitted_as_custom_tool_call():
    patch = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"
    envelope = json.dumps({"name": "apply_patch", "arguments": {"input": patch}})
    client = FakeClient([[("response", f"<tool_call>\n{envelope}\n</tool_call>")]])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    custom = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "custom_tool_call"
    ]
    assert custom[0]["name"] == "apply_patch"
    assert custom[0]["input"] == patch


def test_text_after_tool_output_gets_one_nudge():
    history = base_body()["input"] + [
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd":"ls"}',
            "call_id": "call_1",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    ]
    client = FakeClient(
        [
            [("response", "I think that is enough.")],
            [("response", "Done: verified.")],
        ]
    )
    events = events_of(_response_events(client, base_body(history=history, tools=TOOLS)))
    assert len(client.prompts) == 2
    assert "You have not called `task_complete`" in client.prompts[1]
    text = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.output_text.delta"
    )
    assert text == "Done: verified."


def test_task_complete_summary_becomes_final_text():
    envelope = json.dumps(
        {"name": "task_complete", "arguments": {"summary": "All done."}}
    )
    client = FakeClient([[("response", f"<tool_call>\n{envelope}\n</tool_call>")]])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    text = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.output_text.delta"
    )
    assert "All done." in text
    assert not any(
        event["type"] == "response.output_item.done"
        and event["item"]["type"] in ("function_call", "custom_tool_call")
        for event in events
    )


def test_conversation_id_is_returned_and_resumed_with_only_new_turns():
    history = base_body(text="My favorite color is blue.")["input"]
    client = FakeClient(
        [{"events": [("response", "Got it.")], "conversation_id": "conv-1"}]
    )
    events = events_of(_response_events(client, base_body(history=history, tools=TOOLS)))
    completed = [event for event in events if event["type"] == "response.completed"]
    assert completed[0]["response"]["conversation_id"] == "conv-1"
    assert client.conversation_ids == [None]

    history_with_answer = history + [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Got it."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is my favorite color?"}],
        },
    ]
    body = base_body(history=history_with_answer, tools=TOOLS)
    body["conversation_id"] = "conv-1"
    client.scripts.append({"events": [("response", "Blue.")], "conversation_id": "conv-2"})
    events = events_of(_response_events(client, body))
    completed = [event for event in events if event["type"] == "response.completed"]
    assert completed[0]["response"]["conversation_id"] == "conv-2"

    # Usage must describe the whole transcript, not just the continuation slice,
    # so the TUI's context counter reflects what is actually in the window.
    usage = completed[0]["response"]["usage"]
    full_prompt = _build_prompt(body, _prepare_tools(body["tools"])[0])[0]
    assert usage["input_tokens"] == responses._est_tokens(full_prompt)
    assert usage["input_tokens"] > len(client.prompts[1]) // 4

    assert client.conversation_ids == [None, "conv-1"]
    resumed_prompt = client.prompts[1]
    assert resumed_prompt == "User: What is my favorite color?\n\nAssistant:"
    assert "## Conversation" not in resumed_prompt


def test_tool_result_continuation_sends_only_the_new_result():
    tool_call = {
        "type": "function_call",
        "name": "exec_command",
        "arguments": '{"cmd":"pwd"}',
        "call_id": "call_1",
    }
    tool_output = {"type": "function_call_output", "call_id": "call_1", "output": "/tmp/p"}
    history = base_body(text="where am I?")["input"] + [tool_call, tool_output]
    envelope = json.dumps(
        {"name": "task_complete", "arguments": {"summary": "It is /tmp/p."}}
    )
    client = FakeClient(
        [{"events": [("response", f"<tool_call>\n{envelope}\n</tool_call>")],
          "conversation_id": "conv-3"}]
    )
    body = base_body(history=history, tools=TOOLS)
    body["conversation_id"] = "conv-1"
    events_of(_response_events(client, body))
    assert client.conversation_ids == ["conv-1"]
    assert client.prompts[0].startswith("User:\n<tool_result")
    assert "/tmp/p" in client.prompts[0]
    assert "## Conversation" not in client.prompts[0]


def test_conversation_id_without_assistant_history_starts_fresh():
    client = FakeClient([[("response", "Hello.")]])
    body = base_body(text="hi", tools=TOOLS)
    body["conversation_id"] = "stale-thread"
    events_of(_response_events(client, body))
    assert client.conversation_ids == [None]
    assert "## Conversation" in client.prompts[0]


def test_empty_stream_retries_then_fails(monkeypatch):
    monkeypatch.setattr(responses.time, "sleep", lambda _: None)
    client = FakeClient([[], [], []])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    assert len(client.prompts) == 3
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "empty_response"


def test_reasoning_items_are_replayed_in_the_prompt():
    history = base_body()["input"] + [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "I checked the file."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The file is fine."}],
        },
    ]
    specs, _ = _prepare_tools(TOOLS)
    prompt, _ = _build_prompt(base_body(history=history), specs)
    assert "<reasoning>\nI checked the file.\n</reasoning>" in prompt


def test_dsml_parser_matches_reference_spaced_form():
    block = (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="exec_command">\n'
        '<｜DSML｜ parameter name="cmd" string="true">pwd</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    calls, clean = parse_tool_calls(block)
    assert calls == [{"name": "exec_command", "arguments": {"cmd": "pwd"}}]
    assert clean == ""


def test_dsml_parser_tight_tags_raw_json_and_missing_string_attribute():
    block = (
        "<｜DSML｜tool_calls>"
        '<｜DSML｜invoke name="update_plan">'
        '<｜DSML｜parameter name="steps" string="false">'
        '[{"step": "a", "status": "pending"}]</｜DSML｜parameter>'
        '<｜DSML｜parameter name="note">plain text</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )
    calls, clean = parse_tool_calls(block)
    assert calls == [
        {
            "name": "update_plan",
            "arguments": {
                "steps": [{"step": "a", "status": "pending"}],
                "note": "plain text",
            },
        }
    ]
    assert clean == ""


def test_dsml_empty_invoke_has_empty_arguments():
    block = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="exec_command">'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    calls, _ = parse_tool_calls(block)
    assert calls == [{"name": "exec_command", "arguments": {}}]


def test_dsml_value_with_orphan_double_bar_tag_is_cleaned():
    block = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n'
        '<｜DSML｜ parameter name="cmd" string="true">'
        "pwd</｜｜DSML｜｜></｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    calls, clean = parse_tool_calls(block)
    assert calls == [{"name": "exec_command", "arguments": {"cmd": "pwd"}}]
    assert "DSML" not in clean


def test_dsml_ignores_trailing_garbage_after_block():
    block = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="exec_command">'
        '<｜DSML｜ parameter name="cmd" string="true">pwd</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls> done </｜｜DSML｜｜>"
    )
    calls, clean = parse_tool_calls(block)
    assert calls == [{"name": "exec_command", "arguments": {"cmd": "pwd"}}]
    assert "DSML" not in clean
    assert "done" in clean


def test_orphaned_dsml_in_json_arguments_is_stripped():
    envelope = json.dumps(
        {"name": "exec_command", "arguments": {"cmd": "pwd</｜｜DSML｜｜>"}}
    )
    calls, clean = parse_tool_calls(f"<tool_call>\n{envelope}\n</tool_call>")
    assert calls == [{"name": "exec_command", "arguments": {"cmd": "pwd"}}]
    assert clean == ""

    client = FakeClient([[("response", f"<tool_call>\n{envelope}\n</tool_call>")]])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    emitted = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    ]
    assert json.loads(emitted[0]["arguments"]) == {"cmd": "pwd"}
    text = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "response.output_text.delta"
    )
    assert "DSML" not in text


def test_dsml_accepts_doubled_and_tripled_bars():
    for bars in ("\uff5c", "\uff5c\uff5c", "\uff5c\uff5c\uff5c"):
        dsml = (
            f"<{bars}DSML{bars} calls>\n"
            f'<{bars}DSML{bars} invoke name="exec_command">\n'
            f'<{bars}DSML{bars} parameter name="cmd" string="true">ls'
            f"</{bars}DSML{bars} parameter>\n"
            f"</{bars}DSML{bars} invoke>\n"
            f"</{bars}DSML{bars} calls>"
        )
        calls, clean = parse_tool_calls(dsml)
        assert calls == [{"name": "exec_command", "arguments": {"cmd": "ls"}}]
        assert "DSML" not in clean

    double = (
        "<\uff5c\uff5cDSML\uff5c\uff5c calls>\n"
        '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="exec_command">\n'
        '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="cmd" string="true">pwd'
        "</\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
        "</\uff5c\uff5cDSML\uff5c\uff5c invoke>\n"
        "</\uff5c\uff5cDSML\uff5c\uff5c calls>"
    )
    client = FakeClient([[("response", double)]])
    events = events_of(_response_events(client, base_body(tools=TOOLS)))
    calls = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "function_call"
    ]
    assert calls[0]["name"] == "exec_command"
    assert json.loads(calls[0]["arguments"]) == {"cmd": "pwd"}


def test_native_codex_prompts_are_stripped_from_the_prompt():
    native = (
        "<permissions instructions>sandbox workspace-write</permissions instructions>\n"
        "<collaboration_mode>default</collaboration_mode>\n"
        "You are Codex, based on GPT-5. " + "Native tool instructions. " * 500
    )
    history = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": native}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "You are Codex, native prompt."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<environment_context><cwd>/tmp/p</cwd></environment_context>",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "do the task"}],
        },
    ]
    body = base_body(history=history, tools=TOOLS)
    body["instructions"] = "You are Codex, native system prompt."
    prompt, _ = _build_prompt(body, _prepare_tools(TOOLS)[0])
    assert "permissions instructions" not in prompt
    assert "collaboration_mode" not in prompt
    assert "You are Codex" not in prompt
    assert "<environment_context>" in prompt
    assert "<tool_call>" in prompt


def test_parser_handles_truncated_dsml_and_json_envelope():
    calls, clean = parse_tool_calls(
        'before\n<tool_call>{"name":"exec_command","arguments":{"cmd":"pwd"}}'
        "</tool_call>\nafter"
    )
    assert calls == [{"name": "exec_command", "arguments": {"cmd": "pwd"}}]
    assert "tool_call" not in clean
    truncated = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n'
        '<｜DSML｜ parameter name="cmd" string="true">ls'
    )
    calls, _ = parse_tool_calls(truncated)
    assert calls and calls[0]["name"] == "exec_command"


def test_title_request_is_answered_without_calling_deepseek():
    body = base_body(
        text=(
            "Generate a concise, single-line task title of at most 36 characters "
            "and under five words where possible.\n\nUser prompt:\nFix the login bug"
        )
    )
    client = FakeClient([["Should not be used."]])
    assert responses._is_title_generation(body)
    text = events_of(
        responses._mocked_title_events("resp_1", body["input"][-1]["content"][0]["text"])
    )
    assert text[-1]["type"] == "response.completed"
    delta = "".join(e.get("delta", "") for e in text)
    assert json.loads(delta) == {"title": "New Coding Session"}
    assert client.prompts == []


def test_completed_usage_is_fully_populated():
    client = FakeClient([("The answer is 42.",)])
    events = events_of(_response_events(client, base_body()))
    usage = events[-1]["response"]["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens_details"] == {"cached_tokens": 0}
    assert "reasoning_tokens" in usage["output_tokens_details"]


def test_thinking_follows_reasoning_effort_when_flag_absent():
    body = base_body()
    assert responses._request_thinking(body) is False
    body["reasoning"] = {"effort": "low"}
    assert responses._request_thinking(body) is True
    body["reasoning"] = {"effort": "none"}
    assert responses._request_thinking(body) is False
    body["thinking"] = True
    assert responses._request_thinking(body) is True
