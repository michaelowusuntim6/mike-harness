"""Regression tests for the hallucinated tool-call wrappers seen in the wild.

Each payload is a shape the web model has actually emitted. The contract is the
same for all of them: the right tool name and arguments come out, and nothing of
the wrapper reaches the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.toolcall import parse_tool_calls  # noqa: E402

BAR = "｜"  # U+FF5C, the full-width bar DeepSeek markup uses
OPEN = "<" + "tool_call>"  # split so the tag names read clearly here
CLOSE = "<" + "/tool_call>"

PAYLOADS = [
    (
        "DSML container tag directly around a JSON object",
        f'<{BAR}{BAR}DSML{BAR}{BAR} calls>{{"name": "apply_patch", '
        f'"arguments": {{"input": "*** Begin Patch\\n*** End Patch"}}}}',
        ("apply_patch", {"input": "*** Begin Patch\n*** End Patch"}),
    ),
    (
        "plain <call> wrapper",
        "<call> {"
        '"name": "exec_command", '
        '"arguments": {"cmd": "cd /home/mike/slnm && git status --short"}} '
        "</call>",
        ("exec_command", {"cmd": "cd /home/mike/slnm && git status --short"}),
    ),
    (
        "full-width bar prefixed <call> wrapper",
        f"<{BAR}call> {{\"name\": \"exec_command\", \"arguments\": "
        f'{{"cmd": "cd /home/mike/slnm && git status --short"}}}} </call>',
        ("exec_command", {"cmd": "cd /home/mike/slnm && git status --short"}),
    ),
    (
        "bar prefixed wrapper closed with a bar tag",
        f"<{BAR}call> {{\"name\": \"exec_command\", \"arguments\": "
        f'{{"cmd": "cd /home/mike/slnm && git status --short"}}}} </{BAR}call>',
        ("exec_command", {"cmd": "cd /home/mike/slnm && git status --short"}),
    ),
    (
        "canonical <tool_call> envelope",
        f"{OPEN}{{\"name\": \"exec_command\", \"arguments\": "
        f'{{"cmd": "cd /home/mike/slnm && git status --short"}}}}{CLOSE}',
        ("exec_command", "cd /home/mike/slnm && git status --short"),
    ),
    (
        "panic mode: bare JSON, no wrapper at all",
        '{"name": "exec_command", "arguments": '
        '{"cmd": "cd /home/mike/slnm && git status --short"}}',
        ("exec_command", "cd /home/mike/slnm && git status --short"),
    ),
]


def test_every_observed_wrapper_yields_the_call_and_no_leftover_text():
    for label, text, expected in PAYLOADS:
        calls, clean = parse_tool_calls(text)
        assert len(calls) == 1, f"{label}: got {calls!r}"
        name, args = expected
        assert calls[0]["name"] == name, label
        if isinstance(args, dict):
            assert calls[0]["arguments"] == args, label
        else:
            # Some payloads arrive with the arguments wrapped again; the value
            # that matters is the command itself.
            assert calls[0]["arguments"].get("cmd") == args, label
        assert clean == "", f"{label}: leaked {clean!r}"


def test_no_full_width_bar_survives_anywhere():
    for _, text, _ in PAYLOADS:
        calls, clean = parse_tool_calls(text)
        assert BAR not in clean
        for call in calls:
            assert BAR not in call["name"]
            for key, value in call["arguments"].items():
                assert BAR not in key
                if isinstance(value, str):
                    assert BAR not in value


def test_nested_braces_in_arguments_are_not_cut_short():
    text = (
        "<call>{\"name\": \"apply_patch\", \"arguments\": "
        '{"input": "line\\nwith {braces} inside"}}</call>'
    )
    calls, clean = parse_tool_calls(text)
    assert calls[0]["arguments"]["input"] == "line\nwith {braces} inside"
    assert clean == ""


def test_two_wrapped_calls_in_one_message():
    text = (
        "<call>{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"ls\"}}</call>\n"
        "<call>{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"pwd\"}}</call>"
    )
    calls, clean = parse_tool_calls(text)
    assert [call["arguments"]["cmd"] for call in calls] == ["ls", "pwd"]
    assert clean == ""


def test_prose_quoting_a_name_object_is_not_eaten():
    # An assistant message explaining the format must survive untouched; only a
    # bare payload that IS the whole message counts as a panic-mode call.
    text = (
        'The tool payload looks like {"name": "exec_command", "arguments": '
        '{"cmd": "ls"}} in this protocol. Shall I run it?'
    )
    calls, clean = parse_tool_calls(text)
    assert calls == []
    assert clean == text


def test_malformed_json_inside_wrapper_is_not_a_call():
    text = '<call>{"name": "exec_command", "arguments": {"cmd": "ls"</call>'
    calls, clean = parse_tool_calls(text)
    assert calls == []
