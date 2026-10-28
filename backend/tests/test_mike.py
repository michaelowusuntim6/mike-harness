"""Smoke tests for mike.py — run WITHOUT touching the network.

The OpenAI client is stubbed everywhere it is needed, so this suite is fully
hermetic. Run with:

    venv/bin/python -m pytest tests/test_mike.py -q
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mike  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Prompted-mode tool-call parser                                            #
# --------------------------------------------------------------------------- #

def test_parse_single_tool_call():
    text = (
        "Let me read that file.\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>\n'
        "One moment."
    )
    calls, clean = mike.parse_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "a.py"}}]
    assert "Let me read that file." in clean
    assert "One moment." in clean
    assert "tool_call" not in clean


def test_parse_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>'
    )
    calls, clean = mike.parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["read_file", "list_dir"]
    assert clean == ""


def test_parse_malformed_json_is_ignored():
    text = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        'some text <tool_call>{"name": "broken", "arguments": }</tool_call> more'
    )
    calls, clean = mike.parse_tool_calls(text)
    # Only the well-formed block is a call; the malformed one is dropped, not raised.
    assert [c["name"] for c in calls] == ["read_file"]
    assert "broken" not in clean
    assert "some text" in clean and "more" in clean


def test_parse_text_before_and_after():
    text = (
        "before the call "
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        " after the call"
    )
    calls, clean = mike.parse_tool_calls(text)
    assert calls[0]["name"] == "bash"
    assert clean == "before the call  after the call"


def test_parse_no_tool_calls():
    calls, clean = mike.parse_tool_calls("just a normal answer")
    assert calls == []
    assert clean == "just a normal answer"


def test_parse_arguments_as_json_string():
    text = '<tool_call>{"name": "glob", "arguments": "{\\"pattern\\": \\"*.py\\"}"}</tool_call>'
    calls, _ = mike.parse_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*.py"}}]


BAR = "\uff5c"  # full-width vertical bar used by DeepSeek's DSML tool syntax


def _dsml(name, params):
    """Build a DeepSeek DSML tool-call block (params: list of (name, value))."""
    p = f"<{BAR}DSML{BAR} parameter name=\"%s\" string=\"true\">%s</{BAR}DSML{BAR} parameter>"
    body = "\n".join(p % (k, v) for k, v in params)
    return (f"<{BAR}DSML{BAR} calls>\n"
            f"<{BAR}DSML{BAR} invoke name=\"{name}\">\n"
            f"{body}\n"
            f"</{BAR}DSML{BAR} invoke>\n"
            f"</{BAR}DSML{BAR} calls>")


def test_parse_dsml_tool_call():
    """The free web model emits DeepSeek's native DSML syntax; mike parses it too."""
    text = _dsml("write_file", [("path", "/tmp/mike_e2e_check.txt"),
                                ("content", "hello from mike")])
    calls, clean = mike.parse_tool_calls(text)
    assert calls == [{"name": "write_file",
                      "arguments": {"path": "/tmp/mike_e2e_check.txt",
                                    "content": "hello from mike"}}]
    assert "DSML" not in clean
    assert "parameter" not in clean


def test_dsml_hidden_from_content():
    text = _dsml("bash", [("command", "ls")])
    reasoning, clean = mike.extract_thinking(text)
    assert clean == ""
    assert "DSML" not in clean


# --------------------------------------------------------------------------- #
# 2. Thinking-block stripper                                                   #
# --------------------------------------------------------------------------- #

def test_strip_think_tag():
    open_tag = "<thinking>"
    close_tag = "</thinking>"
    reasoning, clean = mike.extract_thinking("Answer: " + open_tag + "secret reasoning here" + close_tag + "42")
    assert "secret reasoning here" in reasoning
    assert clean == "Answer: 42"
    assert "reasoning" not in clean


def test_strip_reasoning_tag():
    reasoning, clean = mike.extract_thinking("<reasoning>because</reasoning>final")
    assert reasoning == "because"
    assert clean == "final"


def test_strip_tool_call_from_content():
    _, clean = mike.extract_thinking(
        'hi <tool_call>{"name": "bash", "arguments": {}}</tool_call> bye'
    )
    assert clean == "hi  bye"


def test_stream_parser_incremental_reasoning():
    parser = mike.StreamParser()
    out = []
    for chunk in ["<thinking>", "thinking hard", "<｜end▁of▁thinking｜>", "the answer"]:
        out.extend(parser.feed(chunk))
    out.extend(parser.finish())
    reasoning = "".join(t for k, t in out if k == "reasoning")
    content = "".join(t for k, t in out if k == "content")
    assert reasoning == "thinking hard"
    assert content == "the answer"


def test_reasoning_content_on_mock_delta():
    """A streamed delta carrying `reasoning_content` surfaces as a reasoning event."""
    chunks = [
        _chunk(_delta(reasoning_content="step 1 … ")),
        _chunk(_delta(reasoning_content="step 2. ")),
        _chunk(_delta(content="The answer is 42")),
    ]
    llm = mike.DeepSeekLLM("http://127.0.0.1:1/v1")
    llm._openai = _FakeOpenAI(chunks)
    events = list(llm.chat_stream([{"role": "user", "content": "q"}], thinking=True))
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    content = "".join(e["text"] for e in events if e["type"] == "content")
    assert reasoning == "step 1 … step 2. "
    assert content == "The answer is 42"


def test_native_tool_calls_from_mock_delta():
    """Native mode: structured tool_calls on the deltas are accumulated."""
    chunks = [
        _chunk(_delta(tool_calls=[_tc(index=0, id="call_1", name="read_file",
                                      arguments='{"path": "x.py"}')])),
        _chunk(_delta()),
    ]
    llm = mike.DeepSeekLLM("http://127.0.0.1:1/v1")
    llm._openai = _FakeOpenAI(chunks)
    events = list(llm.chat_stream([{"role": "user", "content": "q"}],
                                  tools=[{"type": "function"}]))
    calls = [e for e in events if e["type"] == "tool_calls"][0]["calls"]
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"] == {"path": "x.py"}


# --------------------------------------------------------------------------- #
# 3. Token estimator                                                           #
# --------------------------------------------------------------------------- #

def test_token_estimator_deterministic():
    counter = mike.TokenCounter("heuristic")
    text = "the quick brown fox jumps over the lazy dog" * 3
    assert counter.count(text) == counter.count(text)
    assert counter.count("x" * 40) == 10


def test_token_counter_create_reports_name():
    counter = mike.TokenCounter.create()
    assert counter.name in ("tiktoken", "heuristic")
    assert counter.count("hello world") >= 1


# --------------------------------------------------------------------------- #
# 4. Compactor trigger / non-trigger                                           #
# --------------------------------------------------------------------------- #

class _MockCounter:
    name = "mock"

    def __init__(self, value):
        self.value = value

    def count(self, text):
        return 0

    def count_messages(self, messages):
        return self.value


def test_compactor_threshold_is_400k():
    comp = mike.Compactor(_MockCounter(0))
    assert comp.max_context == 500_000
    assert comp.trigger_tokens == 400_000


def test_compactor_triggers_at_400k():
    comp = mike.Compactor(_MockCounter(400_001))
    assert comp.should_compact([{"role": "user", "content": "x"}]) is True


def test_compactor_does_not_trigger_at_300k():
    comp = mike.Compactor(_MockCounter(300_000))
    assert comp.should_compact([{"role": "user", "content": "x"}]) is False


def test_summary_has_seven_sections():
    comp = mike.Compactor(_MockCounter(0))
    messages = [
        {"role": "user", "content": "Build a web scraper that must never hit prod."},
        {"role": "assistant", "content": "I decided to use httpx instead of requests."},
        {"role": "user", "content": "<tool_result>\n1. [ ] write scraper\n2. [ ] add tests\n</tool_result>"},
    ]
    summary = comp.build_summary(messages)
    for header in mike.SUMMARY_SECTIONS:
        assert header in summary, f"missing {header}"


# --------------------------------------------------------------------------- #
# 5. Session rollover                                                          #
# --------------------------------------------------------------------------- #

def test_rollover_creates_cont1_with_seeded_summary(tmp_path):
    store = mike.SessionStore(tmp_path)
    session = store.new_session(name="demo")
    session.messages = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "sure"},
    ]
    # Mock counter always reports above the threshold, so post-compaction the
    # estimate is still over budget and a rollover must happen.
    comp = mike.Compactor(_MockCounter(500_000), keep_recent=2)
    agent = mike.MikeAgent(
        llm=None, session=session, store=store, compactor=comp,
        toolbox=mike.Toolbox(str(tmp_path)), console=mike._PlainConsole(),
        cwd=str(tmp_path),
    )
    summary = agent.compact(force=True)

    assert "## Goal" in summary and "## Next Steps" in summary
    new = agent.session
    assert new.name == "demo-cont1"
    assert new.messages[0]["content"].startswith(
        "This is a continuation of session demo. Summary follows:"
    )
    assert "## Goal" in new.messages[0]["content"]
    created = list(tmp_path.glob("*__demo-cont1.json"))
    assert created, "expected a -cont1 session file on disk"
    data = json.loads(created[0].read_text())
    assert data["name"] == "demo-cont1"
    assert data["messages"][0]["content"].startswith("This is a continuation of session demo.")


def test_session_file_is_pretty_sorted_json(tmp_path):
    store = mike.SessionStore(tmp_path)
    session = store.new_session(name="fmt")
    session.messages.append({"role": "user", "content": "hi"})
    session.save()
    raw = session.path.read_text()
    assert raw.startswith("{\n")          # pretty-printed
    keys = list(json.loads(raw).keys())
    assert keys == sorted(keys)           # sorted keys


# --------------------------------------------------------------------------- #
# 6. Alias routing                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alias", [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v3",
    "deepseek-v4",
    "deepseek-v4.1-flash",
    "deepseek-flash",
    "deepseek-web",
    "default",
    "something-unknown",
    None,
])
def test_alias_routing(alias):
    assert mike.resolve_model(alias) == "deepseek-v4.1-flash"


def test_single_model_identity():
    assert mike.MODEL_ID == "deepseek-v4.1-flash"
    assert mike.MODEL_LABEL == "DeepSeek V4.1 Flash"


# --------------------------------------------------------------------------- #
# 7. End-to-end agent loop with a stubbed client                               #
# --------------------------------------------------------------------------- #

class _FakeLLM:
    """A stub LLM: first turn asks for a tool, second turn gives the answer."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0
        self.seen_messages = []
        self.seen_conversation_ids = []
        self.base_url = "http://stub/v1"

    def chat_stream(self, messages, tools=None, thinking=False, search=False,
                    conversation_id=None):
        self.seen_messages.append(messages)
        self.seen_conversation_ids.append(conversation_id)
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        for event in turn:
            yield event

    def complete(self, messages, thinking=False, max_tokens=2048):
        return "ok"


def test_agent_runs_tool_then_answers(tmp_path):
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch\n"
    tool_turn = [
        {"type": "content", "text":
            "<tool_call>" + json.dumps({
                "name": "apply_patch",
                "arguments": {"input": patch},
            }) + "</tool_call>"},
    ]
    answer_turn = [{"type": "content", "text": "Done — created hello.txt."}]
    llm = _FakeLLM([tool_turn, answer_turn])

    store = mike.SessionStore(tmp_path / "sessions")
    session = store.new_session(name="e2e")
    comp = mike.Compactor(mike.TokenCounter("heuristic"))
    agent = mike.MikeAgent(
        llm=llm, session=session, store=store, compactor=comp,
        toolbox=mike.Toolbox(str(tmp_path)), console=mike._PlainConsole(),
        tool_mode="prompted", cwd=str(tmp_path),
    )

    result = agent.run("create hello.txt")

    assert (tmp_path / "hello.txt").read_text() == "hi\n"
    assert result == "Done — created hello.txt."
    # Tool result was fed back as a user message wrapped in <tool_result>.
    assert any("<tool_result" in str(m.get("content")) for m in session.messages)
    # Reasoning/tool envelopes are not left in assistant content.
    for m in session.messages:
        if m["role"] == "assistant":
            assert "<tool_call>" not in (m.get("content") or "")


def test_thinking_never_becomes_assistant_content(tmp_path):
    turn = [{"type": "content", "text": "<thinking>thinking...<｜end▁of▁thinking｜>visible answer"}]
    llm = _FakeLLM([turn])
    store = mike.SessionStore(tmp_path / "s")
    session = store.new_session(name="t")
    agent = mike.MikeAgent(
        llm=llm, session=session, store=store,
        compactor=mike.Compactor(mike.TokenCounter("heuristic")),
        toolbox=mike.Toolbox(str(tmp_path)), console=mike._PlainConsole(),
        tool_mode="prompted", cwd=str(tmp_path),
    )
    agent.run("hi")
    assistant = [m for m in session.messages if m["role"] == "assistant"][-1]
    assert assistant["content"] == "visible answer"
    assert "thinking..." not in assistant["content"]


def test_thinking_on_uses_transient_live_and_stops_on_content(tmp_path, monkeypatch):
    reasoning = " ".join(["reason"] * 24)
    turns = [[
        {"type": "reasoning", "text": reasoning},
        {"type": "content", "text": "The poem starts here."},
        {"type": "finish", "finish_reason": "stop"},
    ]]

    class FakeLive:
        instances = []

        def __init__(self, renderable, **kwargs):
            self.renderable = renderable
            self.kwargs = kwargs
            self.updates = []
            self.stopped = False
            FakeLive.instances.append(self)

        def start(self):
            pass

        def update(self, renderable):
            self.renderable = renderable
            self.updates.append(renderable)

        def stop(self):
            self.stopped = True

    class LiveConsole(mike._PlainConsole):
        def set_live(self):
            pass

    monkeypatch.setattr(mike, "Live", FakeLive)
    agent, _llm, _console, _session = _make_roundtrip_agent(tmp_path, turns)
    agent.console = LiveConsole()
    agent.thinking_mode = "on"

    raw, saved_reasoning, calls, _finish, error = agent._stream_once()

    assert raw == "The poem starts here."
    assert saved_reasoning == reasoning
    assert calls == [] and error is None
    assert len(FakeLive.instances) == 1
    live = FakeLive.instances[0]
    assert live.kwargs["transient"] is True
    assert live.stopped is True
    assert isinstance(live.updates[-1], mike.Panel)
    assert reasoning in str(live.updates[-1].renderable)


def test_thinking_on_is_not_reprinted_after_live_stops(tmp_path):
    agent, _llm, console, _session = _make_roundtrip_agent(tmp_path, [[]])
    agent.thinking_mode = "on"

    agent._render_assistant_response("Final answer.", "Temporary reasoning.")

    assert "Final answer." in console.text
    assert "Temporary reasoning." not in console.text


# --------------------------------------------------------------------------- #
# Stub helpers                                                                 #
# --------------------------------------------------------------------------- #

def _delta(content=None, reasoning_content=None, tool_calls=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning_content,
                           tool_calls=tool_calls)


def _tc(index, id, name, arguments):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    def create(self, **kwargs):
        return iter(self._chunks)


class _FakeOpenAI:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


# --------------------------------------------------------------------------- #
# 8. Layered config (v0.2)                                                     #
# --------------------------------------------------------------------------- #

class TestConfig:
    def _clean_env(self, monkeypatch):
        for name in list(mike.ENV_MAP.values()) + ["HOST", "PORT", "MIKE_PORT"]:
            monkeypatch.delenv(name, raising=False)

    def test_defaults_when_no_toml(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch)
        cfg, src = mike.load_config({}, user_config=tmp_path / "u.toml",
                                    repo_config=tmp_path / "r.toml")
        assert cfg["max_context"] == 500_000
        assert cfg["compact_threshold"] == 0.80
        assert cfg["server_port"] == 8000
        assert cfg["model_id"] == "deepseek-v4.1-flash"
        assert set(src.values()) == {"default"}

    def test_precedence_default_user_repo_env_cli(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch)
        u = tmp_path / "u.toml"
        r = tmp_path / "r.toml"
        u.write_text("max_context = 100000\n")
        r.write_text("max_context = 300000\n")

        cfg, src = mike.load_config({}, user_config=u, repo_config=r)
        assert (cfg["max_context"], src["max_context"]) == (300_000, "repo")

        monkeypatch.setenv("MIKE_MAX_CONTEXT", "250000")
        cfg, src = mike.load_config({}, user_config=u, repo_config=r)
        assert (cfg["max_context"], src["max_context"]) == (250_000, "env")

        cfg, src = mike.load_config({"max_context": 200_000}, user_config=u, repo_config=r)
        assert (cfg["max_context"], src["max_context"]) == (200_000, "cli")

    def test_all_keys_optional_and_typed(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch)
        r = tmp_path / "r.toml"
        r.write_text('server_port = 8123\ncompact_threshold = 0.5\n'
                     'log_level = "debug"\nthinking_default = "on"\n')
        cfg, src = mike.load_config({}, user_config=tmp_path / "u.toml", repo_config=r)
        assert cfg["server_port"] == 8123 and src["server_port"] == "repo"
        assert cfg["compact_threshold"] == 0.5
        assert cfg["log_level"] == "debug"
        assert cfg["thinking_default"] == "on"

    def test_invalid_toml_is_ignored(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch)
        r = tmp_path / "r.toml"
        r.write_text("this is not = valid toml [[[\n")
        cfg, src = mike.load_config({}, user_config=tmp_path / "u.toml", repo_config=r)
        assert cfg["max_context"] == mike.DEFAULTS["max_context"]
        assert src["max_context"] == "default"

    def test_env_alias_host_port(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("PORT", "9123")
        cfg, src = mike.load_config({}, user_config=tmp_path / "u.toml",
                                    repo_config=tmp_path / "r.toml")
        assert cfg["server_port"] == 9123
        assert src["server_port"] == "env"

    def test_compact_at_derived_from_max(self):
        cfg = dict(mike.DEFAULTS)
        cfg["max_context"] = 500_000
        cfg["compact_threshold"] = 0.80
        assert mike.compact_at(cfg) == 400_000
        cfg["max_context"] = 200_000
        assert mike.compact_at(cfg) == 160_000

    def test_dir_resolution(self, tmp_path):
        cfg = {"session_dir": str(tmp_path / "s"), "export_dir": ""}
        assert mike.resolved_session_dir(cfg) == tmp_path / "s"
        assert mike.resolved_export_dir(cfg) == mike.ROOT / "exports"


# --------------------------------------------------------------------------- #
# 9. Markdown export (v0.2)                                                    #
# --------------------------------------------------------------------------- #

def _fixture_session():
    return mike.Session({
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "demo",
        "created_at": "2026-10-08T10:00:00",
        "updated_at": "2026-10-08T10:01:00",
        "model": "DeepSeek V4.1 Flash",
        "messages": [
            {"role": "user", "content": "create a file named a.txt containing hi"},
            {"role": "assistant", "content": "", "reasoning": "I should use write_file here.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "write_file",
                                          "arguments": json.dumps({"path": "a.txt",
                                                                   "content": "hi"})}}]},
            {"role": "user",
             "content": "<tool_result>\nwrote 1 line(s) to /tmp/a.txt\n</tool_result>"},
            {"role": "assistant", "content": "Done - created a.txt."},
        ],
        "compactions": [],
    })


def test_export_matches_golden():
    golden = (Path(__file__).parent / "golden_export.md").read_text(encoding="utf-8")
    assert mike.render_transcript(_fixture_session()) == golden


def test_export_writes_file_and_ends_with_newline(tmp_path):
    p = tmp_path / "out.md"
    written = mike.export_session(_fixture_session(), p, force=True,
                                  console=mike._PlainConsole())
    text = written.read_text(encoding="utf-8")
    assert text == mike.render_transcript(_fixture_session())
    assert text.endswith("\n")


def test_export_dir_semantics(tmp_path):
    session = _fixture_session()
    d = tmp_path / "exports"
    d.mkdir()
    cfg = {"export_dir": str(d)}
    p = mike.resolve_export_path(None, session, cfg)
    assert p.parent == d and p.name.startswith("demo__") and p.name.endswith(".md")
    assert mike.resolve_export_path(str(d) + "/", session, cfg).parent == d
    assert mike.resolve_export_path(str(tmp_path / "file.md"), session, cfg).name == "file.md"


def test_export_refuses_overwrite_without_force(tmp_path):
    p = tmp_path / "out.md"
    p.write_text("existing")
    with pytest.raises(mike.ExportCancelled):
        mike.export_session(_fixture_session(), p, interactive=False)
    assert p.read_text() == "existing"


def test_export_escapes_leading_hr(tmp_path):
    s = _fixture_session()
    s.messages = [{"role": "user", "content": "---\nnot a rule"}]
    md = mike.render_transcript(s)
    assert "\\---\nnot a rule" in md


# --------------------------------------------------------------------------- #
# 10. Hardened tool-call round-trip (v0.2) — the acceptance gate               #
# --------------------------------------------------------------------------- #

def _calls_open():
    return f"<{BAR}DSML{BAR} calls>"


def _calls_close():
    return f"</{BAR}DSML{BAR} calls>"


def _invoke_open(name):
    return f"<{BAR}DSML{BAR} invoke name=\"{name}\">"


def _invoke_close():
    return f"</{BAR}DSML{BAR} invoke>"


def _param(name, value, as_string=True):
    attr = ' string="true"' if as_string else ""
    return (f"<{BAR}DSML{BAR} parameter name=\"{name}\"{attr}>"
            f"{value}</{BAR}DSML{BAR} parameter>")


def _dsml_block(name, params):
    lines = [_calls_open(), _invoke_open(name)]
    lines += [_param(k, v) for k, v in params]
    lines += [_invoke_close(), _calls_close()]
    return "\n".join(lines)


def _selfclosing(name, **attrs):
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return (f"<{BAR}DSML{BAR} calls>\n"
            f"<{BAR}DSML{BAR} invoke name=\"{name}\"{extra} />\n"
            f"{_calls_close()}")


def _truncated(name, param_name, value):
    # No closing tags anywhere: the stream stops mid-invoke.
    return (f"<{BAR}DSML{BAR} calls>\n"
            f"<{BAR}DSML{BAR} invoke name=\"{name}\">\n"
            f"<{BAR}DSML{BAR} parameter name=\"{param_name}\" string=\"true\">"
            f"{value}")


class _RecordingConsole(mike._PlainConsole):
    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self):
        return "\n".join(self.lines)


def _assert_no_syntax(text):
    for bad in ("DSML", "<tool_call>", f"<{BAR}", f"</{BAR}"):
        assert bad not in text, f"tool syntax leaked into user-visible output: {bad!r}"


def _make_roundtrip_agent(tmp_path, turns, tool_mode="prompted", cwd=None):
    llm = _FakeLLM(turns)
    store = mike.SessionStore(tmp_path / "sessions")
    session = store.new_session(name="rt")
    comp = mike.Compactor(mike.TokenCounter("heuristic"))
    console = _RecordingConsole()
    agent = mike.MikeAgent(
        llm=llm, session=session, store=store, compactor=comp,
        toolbox=mike.Toolbox(str(cwd or tmp_path)), console=console,
        tool_mode=tool_mode, cwd=str(cwd or tmp_path),
    )
    return agent, llm, console, session


def _spy(agent):
    """Wrap Toolbox.call, recording (name, args) for every dispatch."""
    recorded = []
    original = agent.toolbox.call

    def wrapper(name, args):
        recorded.append((name, args))
        return original(name, args)

    agent.toolbox.call = wrapper
    return recorded


class TestToolCallRoundTrip:
    def test_dsml_standard_roundtrip(self, tmp_path):
        (tmp_path / "x.py").write_text("hello world\n")
        turns = [
            [{"type": "content", "text": _dsml_block("read_file", [("path", "x.py")])}],
            [{"type": "content", "text": "I read the file."}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        result = agent.run("read x.py")

        assert recorded == [("read_file", {"path": "x.py"})]
        assert result == "I read the file."
        fed = [m for m in session.messages if "<tool_result" in str(m.get("content"))]
        assert fed and "hello world" in str(fed[0]["content"])
        assert "DSML" not in result
        _assert_no_syntax(console.text)

    def test_dsml_selfclosing(self, tmp_path):
        (tmp_path / "sub").mkdir()
        turns = [
            [{"type": "content", "text": _selfclosing("list_dir")}],
            [{"type": "content", "text": "listed it"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        result = agent.run("list the dir")

        assert recorded == [("list_dir", {})]
        assert result == "listed it"
        assert any("<tool_result" in str(m.get("content")) for m in session.messages)
        _assert_no_syntax(console.text)

    def test_dsml_selfclosing_with_attrs(self, tmp_path):
        (tmp_path / "x.py").write_text("attr style\n")
        turns = [
            [{"type": "content", "text": _selfclosing("read_file", path="x.py")}],
            [{"type": "content", "text": "ok"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        agent.run("read it")

        assert recorded == [("read_file", {"path": "x.py"})]
        assert any("attr style" in str(m.get("content")) for m in session.messages)
        _assert_no_syntax(console.text)

    def test_dsml_truncated(self, tmp_path, caplog):
        (tmp_path / "x.py").write_text("partial\n")
        turns = [
            [{"type": "content", "text": _truncated("read_file", "path", "x.py")}],
            [{"type": "content", "text": "done"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        with caplog.at_level(logging.WARNING, logger="mike"):
            agent.run("read it")

        assert recorded == [("read_file", {"path": "x.py"})]
        assert any("tool_call_truncated" in r.getMessage() for r in caplog.records)
        assert any("partial" in str(m.get("content")) for m in session.messages)
        _assert_no_syntax(console.text)

    def test_malformed_param_json(self, tmp_path):
        (tmp_path / "x.py").write_text("malformed\n")
        # No string="true" -> claimed as JSON, but "{not json" is not valid JSON.
        text = _dsml_block("read_file", [("path", "{not json")])
        calls, _clean = mike.parse_tool_calls(text)
        assert calls == [{"name": "read_file", "arguments": {"path": "{not json"}}]

        turns = [
            [{"type": "content", "text": text}],
            [{"type": "content", "text": "handled"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        agent.run("go")
        assert recorded == [("read_file", {"path": "{not json"})]
        _assert_no_syntax(console.text)

    def test_multiple_invokes(self, tmp_path):
        (tmp_path / "a.py").write_text("AAA\n")
        (tmp_path / "b.py").write_text("BBB\n")
        text = (_invoke_open("read_file") + "\n" + _param("path", "a.py") + "\n"
                + _invoke_close() + "\n" + _invoke_open("read_file") + "\n"
                + _param("path", "b.py") + "\n" + _invoke_close())
        turns = [
            [{"type": "content", "text": text}],
            [{"type": "content", "text": "both read"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        agent.run("read both")

        assert [r[0] for r in recorded] == ["read_file", "read_file"]
        assert [r[1]["path"] for r in recorded] == ["a.py", "b.py"]
        _assert_no_syntax(console.text)

    def test_mixed_envelope(self, tmp_path):
        (tmp_path / "a.py").write_text("AAA\n")
        (tmp_path / "b.py").write_text("BBB\n")
        json_block = ('<tool_call>{"name": "read_file", '
                      '"arguments": {"path": "a.py"}}</tool_call>')
        text = json_block + "\n" + _dsml_block("read_file", [("path", "b.py")])
        turns = [
            [{"type": "content", "text": text}],
            [{"type": "content", "text": "both read"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        agent.run("read both")

        paths = sorted(r[1]["path"] for r in recorded)
        assert paths == ["a.py", "b.py"]
        assert all(r[0] == "read_file" for r in recorded)
        _assert_no_syntax(console.text)
        assert "<tool_call>" not in console.text

    def test_apply_patch_roundtrip(self, tmp_path):
        target = tmp_path / "code.txt"
        target.write_text("alpha\nbeta\ngamma\n")
        patch = (
            "*** Begin Patch\n"
            "*** Update File: code.txt\n"
            "@@\n"
            " alpha\n"
            "-beta\n"
            "+BETA\n"
            " gamma\n"
            "*** End Patch\n"
        )
        text = _dsml_block("apply_patch", [("input", patch)])
        turns = [
            [{"type": "content", "text": text}],
            [{"type": "content", "text": "Renamed beta to BETA."}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        result = agent.run("change beta to BETA")

        assert recorded[0][0] == "apply_patch"
        assert recorded[0][1]["input"].rstrip("\n") == patch.rstrip("\n")
        # The file on disk was really mutated.
        on_disk = target.read_text()
        assert "BETA" in on_disk and "beta" not in on_disk
        assert result == "Renamed beta to BETA."
        _assert_no_syntax(console.text)

    def test_apply_patch_bad_context_is_atomic(self, tmp_path):
        target = tmp_path / "dup.txt"
        original = "same\nsame\n"
        target.write_text(original)
        patch = (
            "*** Begin Patch\n"
            "*** Add File: should-not-exist.txt\n"
            "+partial\n"
            "*** Update File: dup.txt\n"
            "@@\n"
            "-different\n"
            "+changed\n"
            "*** End Patch\n"
        )
        text = _dsml_block("apply_patch", [("input", patch)])
        turns = [
            [{"type": "content", "text": text}],
            [{"type": "content", "text": "The patch could not be applied."}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        recorded = _spy(agent)
        result = agent.run("replace same")

        # The tool ran, returned an error, and did NOT write the file.
        assert recorded[0][0] == "apply_patch"
        assert target.read_text() == original
        assert not (tmp_path / "should-not-exist.txt").exists()
        fed = [m for m in session.messages if "<tool_result" in str(m.get("content"))]
        assert fed and "error" in str(fed[0]["content"]).lower()
        assert "context mismatch" in str(fed[0]["content"])
        assert result == "The patch could not be applied."
        _assert_no_syntax(console.text)

    def test_native_mode_still_works(self, tmp_path):
        (tmp_path / "x.py").write_text("native path\n")
        turns = [
            [{"type": "tool_calls", "calls": [
                {"id": "call_1", "name": "read_file", "arguments": {"path": "x.py"}}]}],
            [{"type": "content", "text": "native done"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(
            tmp_path, turns, tool_mode="native")
        recorded = _spy(agent)
        result = agent.run("read x.py")

        assert recorded == [("read_file", {"path": "x.py"})]
        assert result == "native done"
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
        assert "native path" in tool_msgs[0]["content"]
        _assert_no_syntax(console.text)

    def test_threads_one_conversation(self, tmp_path):
        """Tool results and follow-ups must reuse ONE upstream chat."""
        (tmp_path / "x.py").write_text("thread\n")
        turns = [
            [{"type": "content", "text": _dsml_block("read_file", [("path", "x.py")])},
             {"type": "conversation_id", "id": "conv-123"}],
            [{"type": "content", "text": "done"}],
        ]
        agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
        agent.run("read x.py")

        # First request opened the thread (no id); the tool-result follow-up
        # reused the same id instead of spawning a new chat.
        assert llm.seen_conversation_ids[0] is None
        assert llm.seen_conversation_ids[1] == "conv-123"
        assert session.conversation_id == "conv-123"

        # The follow-up carried ONLY the new tool result, not the whole history.
        follow_up = llm.seen_messages[1]
        assert len(follow_up) == 1
        assert "<tool_result" in str(follow_up[0]["content"])
        # The result is paired to the assistant's tool-call by its id.
        asst = next(m for m in session.messages if m["role"] == "assistant" and m.get("tool_calls"))
        call_id = asst["tool_calls"][0]["id"]
        assert f'id="{call_id}"' in str(follow_up[0]["content"])

        # And the thread id is persisted on the session for resume.
        assert session.to_dict()["conversation_id"] == "conv-123"


# --------------------------------------------------------------------------- #
# 11. Reference DSML parser + tool-result pairing (v0.3)                        #
# --------------------------------------------------------------------------- #

def _v41_block(name, params):
    """DeepSeek V4.1 DSML: leading space after the ｜DSML｜ token."""
    return _dsml_block(name, params)  # helpers already emit the V4.1 space form


def _v4_block(name, params):
    """DeepSeek V4 DSML: no space after the ｜DSML｜ token (<｜DSML｜tool_calls>)."""
    params_txt = "\n".join(
        f"<{BAR}DSML{BAR}parameter name=\"{k}\" string=\"true\">{v}</{BAR}DSML{BAR}parameter>"
        for k, v in params
    )
    return (f"<{BAR}DSML{BAR}tool_calls>\n"
            f"<{BAR}DSML{BAR}invoke name=\"{name}\">\n"
            f"{params_txt}\n"
            f"</{BAR}DSML{BAR}invoke>\n"
            f"</{BAR}DSML{BAR}tool_calls>")


def test_dsml_v41_space_form():
    calls, clean = mike.parse_tool_calls(_v41_block("read_file", [("path", "x.py")]))
    assert calls == [{"name": "read_file", "arguments": {"path": "x.py"}}]
    assert "DSML" not in clean and "invoke" not in clean


def test_dsml_v4_no_space_form():
    calls, clean = mike.parse_tool_calls(_v4_block("read_file", [("path", "x.py")]))
    assert calls == [{"name": "read_file", "arguments": {"path": "x.py"}}]
    assert "DSML" not in clean and "invoke" not in clean


def test_dsml_block_split_across_frames():
    """A DSML block cut mid-invoke-tag across two SSE frames still parses.

    mike parses on the reassembled raw stream text (StreamParser accumulates
    everything into .raw), so a tag split across frames is rejoined before the
    parser sees it.
    """
    block = _v41_block("read_file", [("path", "x.py")])
    cut = block.index("invoke") + 2  # inside the invoke tag
    sp = mike.StreamParser()
    sp.feed(block[:cut])
    # Nothing complete yet: parsing the partial frame yields no full call.
    partial, _ = mike.parse_tool_calls(sp.raw)
    sp.feed(block[cut:])
    sp.finish()
    calls, clean = mike.parse_tool_calls(sp.raw)
    assert calls == [{"name": "read_file", "arguments": {"path": "x.py"}}]
    assert "DSML" not in clean


def test_tool_call_id_paired_with_result(tmp_path):
    """The assistant tool-call id must equal the id in the paired tool result."""
    (tmp_path / "x.py").write_text("contents\n")
    turns = [
        [{"type": "content", "text": _dsml_block("read_file", [("path", "x.py")])}],
        [{"type": "content", "text": "done"}],
    ]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    agent.run("read x.py")

    asst = next(m for m in session.messages
                if m["role"] == "assistant" and m.get("tool_calls"))
    result_msg = next(m for m in session.messages
                      if m["role"] == "user" and "<tool_result" in str(m.get("content")))
    assert result_msg["content"].index(asst["tool_calls"][0]["id"]) != -1


def test_tool_result_uses_call_id(tmp_path):
    """_run_tools must carry the call's id into the result message."""
    (tmp_path / "x.py").write_text("payload\n")
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, [[]])
    agent._run_tools([{"name": "read_file", "arguments": {"path": str(tmp_path / "x.py")},
                      "id": "call_DEADBEEF"}])
    last = session.messages[-1]
    assert last["role"] == "user"
    assert 'id="call_DEADBEEF"' in last["content"]
    assert "payload" in last["content"]


# --------------------------------------------------------------------------- #
# 12. Empty / failed stream handling (invalid token mid-stream)                #
# --------------------------------------------------------------------------- #

def _assistant_count(session):
    return sum(1 for m in session.messages if m["role"] == "assistant")


def test_run_reports_empty_stream_after_retries(tmp_path, monkeypatch):
    """A blank upstream stream prints a visible error and does not append an
    empty assistant turn after all automatic retries are exhausted."""
    turns = [[{"type": "done"}]]  # nothing emitted by the model
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    sleeps = []
    monkeypatch.setattr(mike.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = agent.run("tell me what you can see")

    low = console.text.lower()
    assert llm.calls == 3
    assert sleeps == [2.0, 2.0]
    assert "retrying (attempt 1/2)" in low
    assert "retrying (attempt 2/2)" in low
    assert "no response from the model" in low
    assert "upstream stream was empty" in low
    assert result == ""
    assert _assistant_count(session) == 0   # no empty assistant message stored
    # The user turn is preserved on disk; only the assistant reply is dropped.
    assert any(m["role"] == "user" for m in session.messages)


def test_empty_stream_retry_recovers_with_same_messages(tmp_path, monkeypatch, caplog):
    task_call = (
        "<tool_call>"
        + json.dumps({
            "name": "task_complete",
            "arguments": {"summary": "Recovered after a transient empty stream."},
        })
        + "</tool_call>"
    )
    turns = [
        [{"type": "done"}],
        [{"type": "content", "text": task_call}],
    ]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    sleeps = []
    monkeypatch.setattr(mike.time, "sleep", lambda seconds: sleeps.append(seconds))

    with caplog.at_level(logging.WARNING, logger="mike"):
        result = agent.run("answer the existing prompt")

    assert result == "Recovered after a transient empty stream."
    assert llm.calls == 2
    assert llm.seen_messages[0] == llm.seen_messages[1]
    assert sleeps == [2.0]
    assert "retrying (attempt 1/2)" in console.text.lower()
    assert sum(1 for m in session.messages if m["role"] == "user") == 1
    assert sum(
        1 for record in caplog.records
        if record.getMessage() == "empty upstream stream; retrying..."
    ) == 1


def test_run_surfaces_sse_error_frame(tmp_path):
    """An SSE error frame (Fix 1) is shown verbatim and stops the turn without
    appending an assistant message."""
    msg = "DeepSeek API error: Authorization Failed (invalid token)"
    turns = [[{"type": "error", "message": msg}]]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)

    result = agent.run("tell me what you can see")

    assert msg in console.text
    assert result == msg
    assert _assistant_count(session) == 0


def test_run_surfaces_raised_stream_error(tmp_path):
    """When the LLM backend *raises* (as the OpenAI SDK does on an error frame),
    run() still reports the failure cleanly instead of crashing the turn."""
    msg = "DeepSeek API error: Authorization Failed (invalid token)"

    class _RaisingLLM:
        base_url = "http://stub/v1"

        def chat_stream(self, *a, **k):
            raise RuntimeError(msg)

        def complete(self, *a, **k):
            return "ok"

    store = mike.SessionStore(tmp_path / "sessions")
    session = store.new_session(name="raise")
    comp = mike.Compactor(mike.TokenCounter("heuristic"))
    console = _RecordingConsole()
    agent = mike.MikeAgent(
        llm=_RaisingLLM(), session=session, store=store, compactor=comp,
        toolbox=mike.Toolbox(str(tmp_path)), console=console,
        tool_mode="prompted", cwd=str(tmp_path),
    )

    result = agent.run("tell me what you can see")

    assert msg in console.text
    assert "This usually means the session was invalidated" in console.text
    assert result == msg
    assert _assistant_count(session) == 0


def test_stream_once_returns_error(tmp_path):
    """_stream_once must return the error message as its 5th element."""
    turns = [[{"type": "error", "message": "boom"}]]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    text, reasoning, calls, finish, err = agent._stream_once()
    assert err == "boom"
    assert text == "" and calls == []


# --------------------------------------------------------------------------- #
# 13. Invalid-token mid-stream: server error frame + client 401 retry          #
# --------------------------------------------------------------------------- #

class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        if self._p.get("code") != 0:
            raise RuntimeError(f"DeepSeek API error: {self._p.get('msg')}")

    def json(self):
        return self._p


class _FakeCookies:
    def clear(self):
        pass

    def update(self, d):
        pass


class _FakeHTTP:
    """httpx stand-in: first POST gets a 401 auth failure, then succeeds."""

    def __init__(self, fail_first=True):
        self.headers = {}
        self.cookies = _FakeCookies()
        self.calls = 0
        self.fail_first = fail_first

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return _FakeResp({"code": 1001,
                              "msg": "Authorization Failed (invalid token)",
                              "data": {}})
        return _FakeResp({"code": 0,
                          "data": {"biz_data": {"chat_session": {"id": "SID-NEW"}}}})


def _bare_client(http):
    """A DeepSeekClient built without touching network/Playwright/PoW."""
    import deepseek.client as dc
    import time as _t
    c = dc.DeepSeekClient.__new__(dc.DeepSeekClient)
    c._http = http
    c.session = dc.Session(token="stale", cookies={}, user_agent="UA",
                           captured_at=_t.time())
    return c


def test_is_auth_error_matches_only_auth_failures():
    import deepseek.client as dc
    assert dc.DeepSeekClient._is_auth_error(
        RuntimeError("DeepSeek API error: Authorization Failed (invalid token)"))
    assert dc.DeepSeekClient._is_auth_error(
        RuntimeError("deepseek api error: Invalid Token"))
    assert not dc.DeepSeekClient._is_auth_error(RuntimeError("DeepSeek API error: rate limited"))


def test_client_refreshes_and_retries_once_after_401(monkeypatch):
    import deepseek.client as dc
    import time as _t
    fresh = dc.Session(token="fresh-tok", cookies={"c": "1"}, user_agent="UA2",
                       captured_at=_t.time())
    refresh_calls = {"n": 0}

    def fake_get_session(**kwargs):
        refresh_calls["n"] += 1
        assert kwargs.get("max_age") == 0          # forces a headless re-capture
        assert kwargs.get("allow_interactive") is False
        return fresh

    monkeypatch.setattr(dc, "get_session", fake_get_session)
    http = _FakeHTTP(fail_first=True)
    c = _bare_client(http)

    sid = c.create_chat_session()

    assert sid == "SID-NEW"
    assert http.calls == 2                 # original + exactly one retry
    assert refresh_calls["n"] == 1         # refreshed once
    assert c.session.token == "fresh-tok"  # new session swapped in
    assert c._http.headers["authorization"] == "Bearer fresh-tok"


def test_client_propagates_when_refresh_cannot_recover(monkeypatch):
    import deepseek.client as dc
    from deepseek.auth import LoginRequired

    def boom(**kwargs):
        raise LoginRequired()

    monkeypatch.setattr(dc, "get_session", boom)
    http = _FakeHTTP(fail_first=True)
    c = _bare_client(http)

    with pytest.raises(LoginRequired):
        c.create_chat_session()
    assert http.calls == 1   # no retry once refresh fails


def test_client_does_not_retry_non_auth_errors():
    import deepseek.client as dc
    http = _FakeHTTP(fail_first=True)
    http._fail_msg = "some other failure"
    # Make the only post fail with a non-auth error.
    def always_fail(*a, **k):
        http.calls += 1
        return _FakeResp({"code": 2000, "msg": "some other failure", "data": {}})
    http.post = always_fail
    c = _bare_client(http)
    with pytest.raises(RuntimeError):
        c.create_chat_session()
    assert http.calls == 1   # non-auth error is not retried


def test_error_frame_shape_and_server_emits_on_failure():
    import json as _json
    from server.openai_format import error_frame
    frame = error_frame("kaboom", err_type="upstream_error", model="m1")
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    obj = _json.loads(frame[len("data: "):].strip())
    assert obj["object"] == "chat.completion.chunk"
    assert obj["model"] == "m1"
    assert obj["choices"][0]["delta"] == {}
    assert obj["choices"][0]["finish_reason"] == "error"
    assert obj["error"] == {"message": "kaboom", "type": "upstream_error"}

    from fastapi.testclient import TestClient
    import server.api as api
    client = TestClient(api.app)

    class _RaisingClient:
        def stream(self, *a, **k):
            raise RuntimeError("DeepSeek API error: Authorization Failed (invalid token)")

    original = api._client
    api._client = _RaisingClient()
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": api.MODEL_ID, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        api._client = original
    assert resp.status_code == 200
    assert '"finish_reason": "error"' in resp.text
    assert "Authorization Failed" in resp.text
    assert "data: [DONE]" in resp.text


# --------------------------------------------------------------------------- #
# 14. Stray <tool_result> wrapper echo must never be visible (assistant side)  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    "<tool_result>abc</tool_result>",   # full block
    "<tool_result>",                     # lone opening tag
    '<tool_result foo="bar">',           # lone opening tag with attributes
    "</tool_result>",                    # lone closing tag
    "< tool_result >",                   # whitespace inside brackets
    "</ tool_result >",                  # whitespace inside closing bracket
    "<TOOL_RESULT>",                     # case-insensitive open
    "</Tool_Result>",                    # case-insensitive close
    "<tool_result>x</TOOL_RESULT>",      # full block, mixed case
])
def test_strip_tool_results_removes_wrappers(text):
    assert mike.strip_tool_results(text) == ""


def test_strip_tool_results_preserves_surrounding_content():
    close = "<" + "/tool_result" + ">"
    # A stray closing tag between words is removed; the surrounding words stay.
    assert mike.strip_tool_results("hello " + close + " world") == "hello  world"
    # Text with no wrapper is returned unchanged (idempotent passthrough).
    assert mike.strip_tool_results("plain  text") == "plain  text"


def test_strip_tool_results_keeps_near_tags():
    # Plain <result> is not the tool_result wrapper.
    assert mike.strip_tool_results("<result>x</result>") == "<result>x</result>"
    # A tag that merely starts with the substring must not be stripped (the \b guard).
    assert mike.strip_tool_results(
        "<tool_result_log>x</tool_result_log>") == "<tool_result_log>x</tool_result_log>"


def test_strip_tool_results_empty_passthrough():
    assert mike.strip_tool_results("") == ""
    assert mike.strip_tool_results("plain text") == "plain text"


def test_stream_parser_hides_tool_result_blocks():
    sp = mike.StreamParser()
    events = []
    for chunk in ["Hi", "<tool_result>", " .git/", "</tool_result>", " done"]:
        events += sp.feed(chunk)
    events += sp.finish()
    content = "".join(t for k, t in events if k == "content")
    reasoning = "".join(t for k, t in events if k == "reasoning")
    # tool_result content is discarded entirely, never accumulated as reasoning.
    assert content == "Hi done"
    assert reasoning == ""


def test_stream_parser_hides_tool_result_split_across_chunks():
    sp = mike.StreamParser()
    events = sp.feed("Hi <tool_")            # partial tag held back by _safe_split
    events += sp.feed("result>HIDDEN</tool_result> bye")
    events += sp.finish()
    content = "".join(t for k, t in events if k == "content")
    reasoning = "".join(t for k, t in events if k == "reasoning")
    assert content == "Hi  bye"
    assert "HIDDEN" not in content and "HIDDEN" not in reasoning


def test_run_strips_tool_result_from_assistant(tmp_path):
    # Model echoes a stray wrapper around a tool call; it must not leak into
    # stored assistant content or the visible terminal output.
    (tmp_path / "x.py").write_text("contents\n")
    text = ("Looking.\n</tool_result>\n<tool_call>" + json.dumps({"name": "read_file", "arguments": {"path": "x.py"}}) + "</tool_call>")
    turn1 = [{"type": "content", "text": text}]
    agent, llm, console, session = _make_roundtrip_agent(
        tmp_path, [turn1, [{"type": "content", "text": "done"}]])
    recorded = _spy(agent)
    agent.run("read x.py")

    assert ("read_file", {"path": "x.py"}) in recorded
    asst = [m.get("content") or "" for m in session.messages if m.get("role") == "assistant"]
    assert asst and all("tool_result" not in t for t in asst)
    assert "Looking." in asst[0]
    assert "tool_result" not in console.text


# --------------------------------------------------------------------------- #
# 15. Codex-style completion and apply_patch protocol (v0.4)                  #
# --------------------------------------------------------------------------- #

def test_apply_patch_add_update_delete_move(tmp_path):
    (tmp_path / "edit.txt").write_text("one\ntwo\nthree\n")
    (tmp_path / "delete.txt").write_text("remove me\n")
    (tmp_path / "move.txt").write_text("move me\n")
    (tmp_path / "eof.txt").write_text("head\n")
    toolbox = mike.Toolbox(str(tmp_path))

    patch = (
        "*** Begin Patch\n"
        "*** Add File: nested/added.txt\n"
        "+hello\n"
        "+world\n"
        "*** Update File: edit.txt\n"
        "@@ optional function name\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
        "*** Delete File: delete.txt\n"
        "*** Update File: move.txt\n"
        "*** Move to: moved/moved.txt\n"
        "*** Update File: eof.txt\n"
        "@@\n"
        " head\n"
        "*** End of File\n"
        "+tail\n"
        "*** End Patch\n"
    )
    result = toolbox.call("apply_patch", {"input": patch})

    assert not result.startswith("error:")
    assert (tmp_path / "nested" / "added.txt").read_text() == "hello\nworld\n"
    assert (tmp_path / "edit.txt").read_text() == "one\nTWO\nthree\n"
    assert not (tmp_path / "delete.txt").exists()
    assert not (tmp_path / "move.txt").exists()
    assert (tmp_path / "moved" / "moved.txt").read_text() == "move me\n"
    assert (tmp_path / "eof.txt").read_text() == "head\ntail\n"
    assert "added nested/added.txt" in result
    assert "updated edit.txt (1 hunks)" in result
    assert "deleted delete.txt" in result
    assert "moved move.txt -> moved/moved.txt" in result

    before_edit = (tmp_path / "edit.txt").read_text()
    (tmp_path / "safe-delete.txt").write_text("keep me\n")
    bad = (
        "*** Begin Patch\n"
        "*** Add File: atomic-new.txt\n"
        "+should not exist\n"
        "*** Delete File: safe-delete.txt\n"
        "*** Update File: edit.txt\n"
        "@@\n"
        "-not present\n"
        "+replacement\n"
        "*** End Patch\n"
    )
    bad_result = toolbox.call("apply_patch", {"input": bad})

    assert bad_result.startswith("error:")
    assert "context mismatch" in bad_result and "edit.txt" in bad_result
    assert not (tmp_path / "atomic-new.txt").exists()
    assert (tmp_path / "safe-delete.txt").read_text() == "keep me\n"
    assert (tmp_path / "edit.txt").read_text() == before_edit


def test_apply_patch_rejects_bad_envelope(tmp_path):
    target = tmp_path / "safe.txt"
    target.write_text("unchanged\n")
    toolbox = mike.Toolbox(str(tmp_path))
    result = toolbox.call("apply_patch", {
        "input": (
            "*** Update File: safe.txt\n"
            "@@\n"
            "-unchanged\n"
            "+changed\n"
            "*** End Patch\n"
        ),
    })
    assert result.startswith("error:")
    assert "*** Begin Patch" in result
    assert target.read_text() == "unchanged\n"


def test_apply_patch_in_toolbox_and_not_write_file(tmp_path):
    specs = mike.Toolbox(str(tmp_path)).specs
    assert "apply_patch" in specs
    assert "update_plan" in specs
    assert "todo_write" not in specs
    assert "task_complete" in specs
    assert "write_file" not in specs
    assert "edit_file" not in specs


def test_task_complete_stops_loop(tmp_path):
    list_call = (
        "<tool_call>"
        + json.dumps({"name": "list_dir", "arguments": {"path": "."}})
        + "</tool_call>"
    )
    task_call = (
        "<tool_call>"
        + json.dumps({
            "name": "task_complete",
            "arguments": {"summary": "Implemented and verified the change."},
        })
        + "</tool_call>"
    )
    turns = [
        [{"type": "content", "text": "I will inspect the files first.\n" + list_call}],
        [{"type": "content", "text": task_call},
         {"type": "finish", "finish_reason": "stop"}],
    ]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    recorded = _spy(agent)

    result = agent.run("finish the task")

    assert llm.calls == 2
    assert result == "Implemented and verified the change."
    assert [name for name, _args in recorded] == ["list_dir"]
    assert any("inspect the files first" in str(m.get("content"))
               for m in session.messages if m["role"] == "assistant")
    assert all(
        mike._call_name(call) != "task_complete"
        for m in session.messages for call in (m.get("tool_calls") or [])
    )


def test_task_complete_summary_hidden_when_text_was_visible(tmp_path):
    visible = "Hi! I'm Mike. What would you like me to do?"
    summary = "No task was requested; the user only sent a greeting."
    task_call = (
        "<tool_call>"
        + json.dumps({
            "name": "task_complete",
            "arguments": {"summary": summary},
        })
        + "</tool_call>"
    )
    turns = [[{
        "type": "content",
        "text": visible + "\n" + task_call,
    }]]
    agent, _llm, console, session = _make_roundtrip_agent(tmp_path, turns)

    result = agent.run("Hi")

    assert result == visible
    assert console.text.count(visible) == 1
    assert summary not in console.text
    assert "task_complete" not in console.text
    assistant = [m for m in session.messages if m["role"] == "assistant"][-1]
    assert assistant["content"] == visible
    assert not assistant.get("tool_calls")


def test_commentary_nudge(tmp_path):
    list_call = (
        "<tool_call>"
        + json.dumps({"name": "list_dir", "arguments": {"path": "."}})
        + "</tool_call>"
    )
    turns = [
        [{"type": "content", "text": "Progress: reading the next file.\n" + list_call}],
        [{"type": "content", "text": "The task is done."}],
    ]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    nudge = (
        "You have not called `task_complete`. Continue the task or call "
        "`task_complete` with a summary."
    )

    result = agent.run("do the work")

    assert llm.calls == 3
    assert result == "The task is done."
    assert nudge in str(llm.seen_messages[2])
    assert sum(1 for m in session.messages if m.get("content") == nudge) == 1


def test_conversational_no_nudge(tmp_path):
    turns = [[{"type": "content", "text": "Hi! How can I help?"}]]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)
    nudge = "You have not called `task_complete`"

    result = agent.run("Hi")

    assert result == "Hi! How can I help?"
    assert llm.calls == 1
    assert "Hi! How can I help?" in console.text
    assert nudge not in console.text
    assert all(nudge not in str(m.get("content", "")) for m in session.messages)


def test_hallucinated_channel_tags_are_stripped(tmp_path):
    list_call = (
        "<tool_call>"
        + json.dumps({"name": "list_dir", "arguments": {"path": "."}})
        + "</tool_call>"
    )
    task_call = (
        "<tool_call>"
        + json.dumps({
            "name": "task_complete",
            "arguments": {"summary": "[final] Hi! I'm Mike."},
        })
        + "</tool_call>"
    )
    turns = [
        [{"type": "content", "text": "[COMMENTARY] Hello! I'm Mike.\n" + list_call}],
        [{"type": "content", "text": task_call}],
    ]
    agent, _llm, console, session = _make_roundtrip_agent(tmp_path, turns)

    result = agent.run("Hi")

    assert result == "Hi! I'm Mike."
    assert "[commentary]" not in console.text.lower()
    assert "[final]" not in console.text.lower()
    for message in session.messages:
        assert "[commentary]" not in str(message.get("content", "")).lower()
        assert "[final]" not in str(message.get("content", "")).lower()


def _reasoning_replay_agent(tmp_path, tool_mode):
    agent, _llm, _console, session = _make_roundtrip_agent(
        tmp_path, [[]], tool_mode=tool_mode)
    session.messages = [
        {"role": "user", "content": "inspect it"},
        {
            "role": "assistant",
            "content": "I inspected the file.",
            "reasoning": "I should read before editing.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "x.py"}),
                },
            }],
        },
    ]
    agent.context = list(session.messages)
    return agent, session


def test_reasoning_replayed_prompted(tmp_path):
    agent, session = _reasoning_replay_agent(tmp_path, "prompted")
    messages = agent._request_messages()
    assistant = next(m for m in messages if m["role"] == "assistant")

    assert "<reasoning>\nI should read before editing.\n</reasoning>" in assistant["content"]
    assert "reasoning_content" not in assistant
    assert session.messages[1]["content"] == "I inspected the file."


def test_reasoning_replayed_native(tmp_path):
    agent, _session = _reasoning_replay_agent(tmp_path, "native")
    messages = agent._request_messages()
    assistant = next(m for m in messages if m["role"] == "assistant")

    assert assistant["reasoning_content"] == "I should read before editing."
    assert "<reasoning>" not in (assistant["content"] or "")
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_server_prompt_and_completion_replay_reasoning():
    from server.openai_format import completion_response, messages_to_prompt
    from server.schemas import ChatMessage

    prompt = messages_to_prompt([
        ChatMessage(role="user", content="inspect it"),
        ChatMessage(
            role="assistant",
            content="I inspected it.",
            reasoning_content="I should read before editing.",
        ),
        ChatMessage(role="user", content="continue"),
    ])
    response = completion_response(
        "deepseek-v4.1-flash", "done", prompt,
        reasoning="Final reasoning.",
    )

    assert "Assistant (reasoning): I should read before editing." in prompt
    assert response["choices"][0]["message"]["reasoning_content"] == "Final reasoning."


def test_compaction_header_and_awareness(tmp_path):
    agent, _llm, _console, session = _make_roundtrip_agent(tmp_path, [[]])
    session.conversation_id = "conv-compact"
    agent.context = [
        {"role": "user", "content": "Implement the feature."},
        {"role": "assistant", "content": "Working on it."},
    ]
    pending = {"role": "user", "content": "Continue with tests."}
    agent._unsent = [pending]

    summary = agent.compact(force=True)

    assert summary
    assert agent.context[0]["content"].startswith(mike.COMPACTION_HEADER)
    assert agent._unsent[0]["content"].startswith(mike.COMPACTION_HEADER)
    assert sum(
        1 for m in agent._unsent
        if str(m.get("content", "")).startswith(mike.COMPACTION_HEADER)
    ) == 1
    assert pending in agent._unsent
    assert "Compaction awareness" in agent._system
    only_compaction = agent.compactor.build_summary([agent.context[0]])
    assert "The conversation was compacted." not in only_compaction


def test_dsml_stop_retry(tmp_path, caplog):
    raw_dsml = (
        "Trying a tool call.\n"
        f"<{BAR}DSML{BAR}tool_calls>\n"
        f"<{BAR}DSML{BAR} truncated"
    )
    task_call = (
        "<tool_call>"
        + json.dumps({
            "name": "task_complete",
            "arguments": {"summary": "Retried and completed."},
        })
        + "</tool_call>"
    )
    turns = [
        [
            {"type": "content", "text": raw_dsml},
            {"type": "finish", "finish_reason": "stop"},
        ],
        [{"type": "content", "text": task_call}],
    ]
    agent, llm, console, session = _make_roundtrip_agent(tmp_path, turns)

    with caplog.at_level(logging.WARNING, logger="mike"):
        result = agent.run("try the tool")

    assert result == "Retried and completed."
    assert llm.calls == 2
    assert "Your last message contained tool syntax" in str(llm.seen_messages[1])
    assert "tool-call syntax detected in content but not parsed" in console.text
    assert sum(1 for r in caplog.records
               if r.getMessage() == "dsml_unparsed_retry") == 1


# --------------------------------------------------------------------------- #
# 16. Codex-style TUI helpers                                                  #
# --------------------------------------------------------------------------- #

def test_mike_completer_commands_and_arguments():
    from prompt_toolkit.document import Document

    completer = mike.MikeCompleter()
    commands = list(completer.get_completions(Document("/th"), None))
    assert [item.text for item in commands] == ["/thinking"]
    assert "reasoning" in commands[0].display_meta_text.lower()

    arguments = list(completer.get_completions(Document("/thinking o"), None))
    assert {item.text for item in arguments} == {"on", "off"}


def test_prompt_session_uses_file_history_and_multiline():
    class _Agent:
        pass

    session = mike._build_prompt_session(_Agent())
    assert session.multiline is True
    assert isinstance(session.completer, mike.MikeCompleter)
    assert session.history.filename == str(mike.ROOT / ".mike_history")


def test_update_plan_schema_and_rendering(tmp_path):
    console = _RecordingConsole()
    toolbox = mike.Toolbox(str(tmp_path), console=console)
    result = toolbox.call("update_plan", {
        "plan": [
            {"step": "Inspect the repo", "status": "completed"},
            {"step": "Edit app.py", "status": "in_progress"},
            {"step": "Run tests", "status": "pending"},
        ],
    })

    assert "Plan · 1/3 complete" in result
    assert "✓ Inspect the repo" in result
    assert "⠋ Edit app.py" in result
    assert "○ Run tests" in result
    assert "✓ Inspect the repo" in console.text
    assert toolbox.load_plan()[1]["status"] == "in_progress"
    assert toolbox.call("todo_read", {}) == result


def test_status_bar_reports_context_plan_and_cwd(tmp_path):
    agent, _llm, _console, _session = _make_roundtrip_agent(tmp_path, [[]])
    agent.toolbox.call("update_plan", {
        "plan": [
            {"step": "One", "status": "completed"},
            {"step": "Two", "status": "pending"},
        ],
    })
    status = str(mike.render_status_bar(agent))

    assert "DeepSeek V4.1 Flash" in status
    assert "Ctx " in status and "500,000" in status
    assert "Plan 1/2" in status
    assert "cwd" not in status


def test_apply_patch_renders_unified_diff(tmp_path):
    target = tmp_path / "code.py"
    target.write_text("old\n")
    console = _RecordingConsole()
    toolbox = mike.Toolbox(str(tmp_path), console=console)
    patch = (
        "*** Begin Patch\n"
        "*** Update File: code.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )

    result = toolbox.call("apply_patch", {"input": patch})

    assert "updated code.py (1 hunks) +1 -1" in result
    assert "--- a/code.py" in console.text
    assert "+++ b/code.py" in console.text
    assert "-old" in console.text and "+new" in console.text
    assert target.read_text() == "new\n"
