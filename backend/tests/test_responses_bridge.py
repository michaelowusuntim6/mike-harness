"""Tests for the Responses API bridge (no browser or network required)."""

from __future__ import annotations

import json

from server import responses
from server.responses import _response_events, _build_prompt, _prepare_tools
from server.toolcall import parse_tool_calls


class FakeStream:
    def __init__(self, events):
        self.events = events

    def iter_events(self):
        yield from self.events


class FakeClient:
    def __init__(self, scripts):
        self.scripts = [list(script) for script in scripts]
        self.prompts: list[str] = []

    def stream(self, prompt, model=None, thinking=False):
        self.prompts.append(prompt)
        if not self.scripts:
            return FakeStream([("response", "No script left.")])
        return FakeStream(self.scripts.pop(0))


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
