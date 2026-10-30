#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mike - a self-contained, zero-cost DeepSeek coding harness.

`mike.py` is a single file that:

  * launches (or attaches to) the local DeepSeek-API server on 127.0.0.1,
  * talks to it over the OpenAI-compatible ``/v1`` endpoints,
  * and runs an interactive coding-agent CLI with tools, thinking-block
    handling, inline compaction, resumable JSON sessions and auto-rollover.

It surfaces exactly ONE model to the user - the single free DeepSeek web model,
whose display name and wire id live in the ``DEFAULTS`` config block below - no
matter what the backend advertises. Every legacy model name is accepted and
routed to that one model.

Everything mike needs (agent loop, tool registry, compactor, session manager,
thinking parser, tool-call parser, server supervisor) lives in this file. It has
no runtime dependency on ``~/nanocode`` or ``~/cipher-harness``.

--------------------------------------------------------------------------------
Licence / provenance (original headers preserved)

  * mike.py is derived from ``nanocode.py`` - released under the MIT License.
  * Tool logic and the compaction approach are ported from ``cipher-harness``
    (compact.py, tools/*) - released under the MIT License.
  * The surrounding DeepSeek-API project is MIT, Copyright (c) 2026 !(Not Sumit).

  Runtime dependencies that are *not* in the standard library are declared in
  ``requirements.txt`` (rich, openai/httpx, tiktoken - all optional at import
  time with graceful fallbacks).
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import difflib
import glob as _globlib
import html as _html
import json
import logging
import os
import re
import select as _select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# 0. Optional third-party imports (graceful fallbacks keep mike runnable)      #
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - exercised only when rich is installed
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except Exception:  # pragma: no cover
    _RICH = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    _PROMPT_TOOLKIT = True
except Exception:  # pragma: no cover
    PromptSession = None
    Completer = object
    Completion = None
    FormattedText = None
    FileHistory = None
    KeyBindings = None
    Style = None
    _PROMPT_TOOLKIT = False


def _make_console():  # pragma: no cover - trivial
    if _RICH:
        return Console()
    return _PlainConsole()


class _PlainConsole:
    """A tiny rich.Console stand-in used when rich is unavailable."""

    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        print(text, end=end)

    def rule(self, title=""):
        print(f"--- {title} " + "-" * max(0, 60 - len(title)))


# --------------------------------------------------------------------------- #
# 1. Configuration                                                            #
# --------------------------------------------------------------------------- #
#
# Layered config, resolved once at startup. Later layers override earlier ones:
#   1. DEFAULTS (below)
#   2. ~/.config/mike/config.toml   (source: "user")
#   3. ./mike.toml                  (source: "repo")
#   4. environment MIKE_* variables (source: "env")
#   5. CLI flags                    (source: "cli")
# Merge is a shallow per-key dict.update, not a deep merge. `mike.py config`
# prints the resolved value and winning source for every key.

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".mike_state.json"
USER_CONFIG = Path(os.path.expanduser("~/.config/mike/config.toml"))
REPO_CONFIG = ROOT / "mike.toml"

# Every key is optional; these are the defaults. This block is the ONLY place
# these literals occur in mike.py.
DEFAULTS = {
    "max_context": 120_000,        # total tokens before compaction is possible
    "compact_threshold": 0.80,     # fraction of max_context that triggers compaction
    "keep_recent_turns": 6,        # turns preserved verbatim across a compaction
    "server_host": "127.0.0.1",
    "server_port": 8000,
    "server_cmd": "",              # empty => auto-detect (python app.py)
    "model": "DeepSeek V4.1 Flash",
    "model_id": "deepseek-v4.1-flash",   # wire id sent to the server
    "thinking_default": "off",     # off | on | spinner | last
    "tool_call_mode": "auto",      # "auto" | "native" | "prompted"
    "session_dir": "",             # empty => <repo>/sessions/
    "export_dir": "",              # empty => <repo>/exports/
    "log_level": "info",           # "debug" | "info" | "warn" | "error"
}

# Types used to coerce string values from the environment.
_INT_KEYS = {"max_context", "keep_recent_turns", "server_port"}
_FLOAT_KEYS = {"compact_threshold"}

ENV_MAP = {
    "max_context": "MIKE_MAX_CONTEXT",
    "compact_threshold": "MIKE_COMPACT_THRESHOLD",
    "keep_recent_turns": "MIKE_KEEP_RECENT_TURNS",
    "server_host": "MIKE_SERVER_HOST",
    "server_port": "MIKE_SERVER_PORT",
    "model": "MIKE_MODEL",
    "thinking_default": "MIKE_THINKING_DEFAULT",
    "tool_call_mode": "MIKE_TOOL_CALL_MODE",
    "session_dir": "MIKE_SESSION_DIR",
    "export_dir": "MIKE_EXPORT_DIR",
    "log_level": "MIKE_LOG_LEVEL",
}

# Backwards-compatible env aliases (v0.1 documented HOST / PORT / MIKE_PORT).
ENV_ALIASES = {
    "server_host": ["HOST"],
    "server_port": ["MIKE_PORT", "PORT"],
}

# Convenience constants derived from the defaults block (no duplicated literals).
MODEL_ID = DEFAULTS["model_id"]
MODEL_LABEL = DEFAULTS["model"]
THINKING_OFF = DEFAULTS["thinking_default"]
THINKING_CHOICES = [THINKING_OFF, "on", "spinner", "last"]

# Every one of these aliases is accepted and transparently routed to MODEL_ID.
MODEL_ALIASES = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v3",
    "deepseek-v4",
    MODEL_ID,
    "deepseek-flash",
    "deepseek-web",
    "default",
]

API_KEY = os.getenv("MIKE_API_KEY", "sk-local")
STARTUP_TIMEOUT = 30.0

# Module logger. Configured by `configure_logging()` at startup; tests capture
# it directly. Kept handler-free here so importing mike stays side-effect free.
log = logging.getLogger("mike")


def _read_toml(path: Path) -> dict:
    """Load a TOML file, returning {} if it is missing or unreadable."""
    try:
        if not Path(path).exists():
            return {}
        raw = Path(path).read_bytes()
    except Exception:
        return {}
    parser = None
    try:
        import tomllib as parser  # Python 3.11+
    except Exception:
        try:
            import tomli as parser  # type: ignore  # Python < 3.11 fallback
        except Exception:
            parser = None
    if parser is None:
        return {}
    try:
        data = parser.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce(key: str, value):
    if value is None:
        return None
    if key in _INT_KEYS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return DEFAULTS[key]
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULTS[key]
    return value if isinstance(value, str) else str(value)


def load_config(cli_overrides: Optional[dict] = None,
                user_config: Optional[Path] = None,
                repo_config: Optional[Path] = None) -> tuple:
    """Resolve the layered config. Returns ``(values, sources)`` dicts."""
    values = dict(DEFAULTS)
    sources = {k: "default" for k in DEFAULTS}

    for path, label in ((user_config or USER_CONFIG, "user"),
                        (repo_config or REPO_CONFIG, "repo")):
        for key, val in _read_toml(path).items():
            if key in DEFAULTS:
                values[key] = _coerce(key, val)
                sources[key] = label

    for key, env_name in ENV_MAP.items():
        if os.environ.get(env_name, "") != "":
            values[key] = _coerce(key, os.environ[env_name])
            sources[key] = "env"
    for key, aliases in ENV_ALIASES.items():
        if sources.get(key) == "default":
            for alias in aliases:
                if os.environ.get(alias, "") != "":
                    values[key] = _coerce(key, os.environ[alias])
                    sources[key] = "env"
                    break

    for key, val in (cli_overrides or {}).items():
        if val is None or key not in DEFAULTS:
            continue
        values[key] = _coerce(key, val)
        sources[key] = "cli"

    return values, sources


def compact_at(cfg: dict) -> int:
    """The token count at which compaction triggers (threshold × max)."""
    return int(cfg["max_context"] * cfg["compact_threshold"])


def resolved_session_dir(cfg: dict) -> Path:
    d = (cfg.get("session_dir") or "").strip()
    return Path(os.path.expanduser(d)) if d else (ROOT / "sessions")


def resolved_export_dir(cfg: dict) -> Path:
    d = (cfg.get("export_dir") or "").strip()
    return Path(os.path.expanduser(d)) if d else (ROOT / "exports")


def configure_logging(level: str = "info") -> None:
    """Attach a stderr handler to the `mike` logger at the configured level."""
    levels = {"debug": logging.DEBUG, "info": logging.INFO,
              "warn": logging.WARNING, "warning": logging.WARNING,
              "error": logging.ERROR}
    log.setLevel(levels.get(str(level).lower(), logging.INFO))
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(handler)
        log.propagate = False

SUMMARY_SECTIONS = [
    "## Goal",
    "## Constraints",
    "## Files Touched",
    "## Decisions Made",
    "## Open Todos",
    "## Last User Intent",
    "## Next Steps",
]
COMPACTION_PREFIX = "The conversation was compacted."
COMPACTION_HEADER = (
    "The conversation was compacted. This is a summary of the work so far. "
    "Continue the task from where it left off. Do not restart or redo finished "
    "work. The last user intent is the current instruction."
)


# --------------------------------------------------------------------------- #
# 2. Model alias routing                                                       #
# --------------------------------------------------------------------------- #

def resolve_model(name: Optional[str]) -> str:
    """Route *any* model name to the single surfaced backend model.

    mike only has one model. Aliases are accepted for compatibility; everything -
    including unknown names - resolves to ``MODEL_ID`` so a stale config can
    never turn into a hard error.
    """
    return MODEL_ID


# --------------------------------------------------------------------------- #
# 3. State file (.mike_state.json) with estimator choice + tool-mode cache     #
# --------------------------------------------------------------------------- #

def load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_state(state: dict) -> None:
    try:
        atomic_write_json(STATE_FILE, state)
    except Exception:
        pass


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON pretty-printed with sorted keys, atomically.

    We never hand a partially-written file to the user: write to a temp file in
    the same directory, fsync, then ``os.replace`` (atomic on POSIX). This is
    what keeps a crash from corrupting an on-disk session.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    data = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 4. Token estimation (tiktoken if available, else len//4)                     #
# --------------------------------------------------------------------------- #

class TokenCounter:
    """Deterministic token estimator with a persisted implementation choice."""

    def __init__(self, name: str, encoder: Any = None):
        self.name = name
        self._enc = encoder

    @classmethod
    def create(cls) -> "TokenCounter":
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            enc.encode("hello world")  # sanity check before committing to it
            return cls("tiktoken", enc)
        except Exception:
            return cls("heuristic", None)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_messages(self, messages: Iterable[dict]) -> int:
        total = 0
        for msg in messages:
            total += 4  # per-message framing overhead
            total += self._count_content(msg.get("content", ""))
            reasoning = msg.get("reasoning")
            if reasoning:
                total += self.count(str(reasoning))
            for call in msg.get("tool_calls", []) or []:
                total += self.count(json.dumps(call, ensure_ascii=False))
        return total

    def _count_content(self, content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return self.count(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    total += self.count(str(block.get("text") or block.get("content") or ""))
                else:
                    total += self.count(str(block))
            return total
        return self.count(str(content))


# --------------------------------------------------------------------------- #
# 5. Thinking-block + tool-call stream parsing                                 #
# --------------------------------------------------------------------------- #

# Matches both XML-ish tags (<think>, <thinking>, <reasoning>, <tool_call>) and
# DeepSeek's full-width special tokens (<｜begin▁of▁thinking｜> /
# <｜end▁of▁thinking｜>), tolerating stray whitespace.
_TAG_RE = re.compile(
    r"<｜(?:begin|end)▁of▁(?:thinking|reasoning)｜>"
    r"|<\s*/?\s*(?:think|thinking|reasoning|tool_call)\s*>"
    r"|<[^<>]*?(?:DSML|dsml)[^<>]*>"
    r"|<\s*tool_result\b[^>]*>|<\s*/\s*tool_result\s*>",
    re.IGNORECASE,
)
_REASONING_TAGS = {"think", "thinking", "reasoning"}


def _tag_name(raw_tag: str) -> tuple[str, bool]:
    raw = raw_tag.strip()
    low = raw.lower()
    # mike's tool_result wrapper echoed by the model: hide it entirely (never
    # treated as reasoning). The \b guard keeps near-tags like <tool_result_log>
    # from being classified as tool_result.
    if re.match(r"<\s*/?\s*tool_result\b", low):
        return "tool_result", raw.startswith("</")
    # DeepSeek's native DSML tool-call tags. The container tags mention "calls";
    # the inner invoke/parameter tags do not, so they are no-ops for the parser
    # (their text is swallowed while the container keeps us in "hidden" mode).
    if "dsml" in low:
        if "calls" in low:
            return "tool_call", raw.startswith("</")
        return "dsml_inner", raw.startswith("</")
    if "▁of▁" in raw:  # DeepSeek special token, e.g. <｜end▁of▁thinking｜>
        closing = "end▁of▁" in low
        if "thinking" in low:
            return "thinking", closing
        if "reasoning" in low:
            return "reasoning", closing
        return "think", closing
    closing = raw.startswith("</")
    name = re.sub(r"[<>/\s]", "", raw).lower()
    return name, closing


def extract_thinking(text: str) -> tuple[str, str]:
    """Split *text* into (reasoning, clean_content).

    Understands ``< -think->...`` style blocks, ``<thinking>...</thinking>`` and
    ``<reasoning>...</reasoning>``. Any ``<tool_call>...</tool_call>`` blocks are
    stripped from the clean content as well.
    """
    if not text:
        return "", ""
    reasoning_parts: list[str] = []
    out: list[str] = []
    mode = "content"
    pos = 0
    for m in _TAG_RE.finditer(text):
        chunk = text[pos:m.start()]
        if chunk:
            if mode == "reasoning":
                reasoning_parts.append(chunk)
            elif mode != "hidden":
                out.append(chunk)
        pos = m.end()
        name, closing = _tag_name(m.group(0))
        if name in _REASONING_TAGS:
            mode = "content" if closing else "reasoning"
        elif name == "tool_call":
            mode = "content" if closing else "hidden"
        elif name == "tool_result":
            mode = "content" if closing else "hidden"
    tail = text[pos:]
    if tail:
        if mode == "reasoning":
            reasoning_parts.append(tail)
        elif mode != "hidden":
            out.append(tail)
    return "".join(reasoning_parts).strip(), "".join(out).strip()


def strip_thinking(text: str) -> str:
    """Return assistant content with reasoning and tool_call blocks removed."""
    return extract_thinking(text)[1]


# mike feeds tool output back to the model wrapped as <tool_result>...</tool_result>
# inside USER messages (the reference encoding shape). The model sometimes mimics that
# wrapper inside its own assistant reply; it must never reach the user's terminal. These
# remove a full block (content and all), a lone opening tag (with or without attributes),
# or a lone closing tag — including whitespace/case variants — while never matching a
# plain <result> or a near-tag like <tool_result_log> (the `\b` guard).
_TOOL_RESULT_RE = re.compile(
    r"<\s*tool_result\b[^>]*>.*?<\s*/\s*tool_result\s*>"   # full block (with attrs)
    r"|<\s*/\s*tool_result\s*>"                            # lone closing tag
    r"|<\s*tool_result\b[^>]*>",                           # lone opening tag (with attrs)
    re.DOTALL | re.IGNORECASE,
)


def strip_tool_results(text: str) -> str:
    """Remove tool_result tags from the assistant's visible output.

    The tool_result wrapper is a marker mike sends INSIDE user messages to feed tool
    output back to the model. The model sometimes mimics the wrapper in its own
    responses. That wrapper must never be visible in the assistant's reply.
    """
    if not text:
        return text
    return _TOOL_RESULT_RE.sub("", text)


def strip_channel_tags(text: str) -> str:
    """Remove hallucinated Codex channel labels from visible assistant text."""
    if not text:
        return text
    text = re.sub(r"\[commentary\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[final\]\s*", "", text, flags=re.IGNORECASE)
    return text


class StreamParser:
    """Incremental parser for streamed deltas.

    Classifies incoming text as ``content``, ``reasoning`` or hidden (inside a
    ``<tool_call>`` block), holding back a small tail so a tag split across two
    chunks is not leaked to the screen.
    """

    MAX_TAG_LEN = 32

    def __init__(self):
        self._buf = ""
        self._mode = "content"
        self.reasoning = ""
        self.content = ""
        self.raw = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self.raw += chunk
        self._buf += chunk
        emitted: list[tuple[str, str]] = []
        while True:
            m = _TAG_RE.search(self._buf)
            if m is None:
                safe = self._safe_split()
                if safe:
                    emitted.append(self._emit(self._buf[:safe]))
                    self._buf = self._buf[safe:]
                break
            if m.start() > 0:
                emitted.append(self._emit(self._buf[:m.start()]))
            name, closing = _tag_name(m.group(0))
            if name in _REASONING_TAGS:
                self._mode = "content" if closing else "reasoning"
            elif name == "tool_call":
                self._mode = "content" if closing else "hidden"
            elif name == "tool_result":
                # Echoed tool_result wrapper: discard its content entirely (it is
                # feedback the model saw, not reasoning), never accumulate it.
                self._mode = "content" if closing else "hidden"
            self._buf = self._buf[m.end():]
        return [e for e in emitted if e[1]]

    def _emit(self, text: str) -> tuple[str, str]:
        if self._mode == "reasoning":
            self.reasoning += text
            return ("reasoning", text)
        if self._mode == "hidden":
            return ("hidden", "")
        self.content += text
        return ("content", text)

    def _safe_split(self) -> int:
        """Length of the buffer that is safe to emit (excludes a partial tag)."""
        idx = self._buf.rfind("<")
        if idx == -1:
            return len(self._buf)
        if ">" in self._buf[idx:]:
            return len(self._buf)
        if len(self._buf) - idx <= self.MAX_TAG_LEN:
            return idx
        return len(self._buf)

    def finish(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        out = [self._emit(self._buf)]
        self._buf = ""
        return [e for e in out if e[1]]


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Parser adapted from the DeepSeek reference encoding.py
# (https://huggingface.co/msuiche/DeepSeek-V4.1-Flash-NVFP4)
# Handles V4 and V4.1 DSML, full-width bar U+FF5C.
#
# The reference tag token is "<｜DSML｜" (the bar is the full-width character
# U+FF5C, never ASCII '|'). V4 tags carry no space after the token
# ("<｜DSML｜tool_calls>", "<｜DSML｜invoke>"); V4.1 tags add a leading space
# ("<｜DSML｜ calls>", "<｜DSML｜ invoke>"). Both forms are accepted here, keyed
# off the "｜DSML｜" token plus the ASCII words (calls / invoke / parameter),
# exactly as the Rust reference stream parser (deepseek-recipe) treats them.
#
#   container tag ...calls...
#     <... invoke name="tool"> <... parameter name="k" string="true|false">v</... parameter> </... invoke>
#     <... invoke name="tool" />            (self-closing -> {} args)
#     <... invoke name="tool" path="x" />   (attribute-style args, superset)
_DSML_TOKEN = "\uff5cDSML\uff5c"  # full-width bar U+FF5C, not ASCII '|'
_DSML_OPEN_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*invoke\b([^<>]*?)(/?)>", re.IGNORECASE
)
_DSML_CLOSE_RE = re.compile(
    r"</\s*\uff5cDSML\uff5c\s*invoke\s*>", re.IGNORECASE
)
_DSML_PARAM_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*parameter\b([^<>]*?)>(.*?)(?:"
    r"</\s*\uff5cDSML\uff5c\s*parameter\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)
_DSML_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*)\s*=\s*"([^"]*)"')
# The container block spans from a (V4 or V4.1) calls-open tag to its close.
_DSML_BLOCK_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>.*?"
    r"</\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_CALLS_OPEN_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>", re.IGNORECASE
)
_DSML_INVOKE_OPEN_BARE_RE = re.compile(r"<\s*\uff5cDSML\uff5c\s*invoke\b", re.IGNORECASE)
_DSML_PARAM_ANY_RE = re.compile(
    r"</?\s*\uff5cDSML\uff5c\s*parameter\b[^<>]*>", re.IGNORECASE
)


def _attr_value(attrs: str, key: str) -> Optional[str]:
    for name, value in _DSML_ATTR_RE.findall(attrs or ""):
        if name.lower() == key.lower():
            return value
    return None


def _attrs_to_args(attrs: str) -> dict:
    """Attribute-style arguments, used by self-closing invoke tags.

    The reference encoder never emits attribute-style args (parameters are
    always child tags), so this is a tolerant superset, not a reference path.
    """
    args: dict = {}
    for name, value in _DSML_ATTR_RE.findall(attrs or ""):
        if name.lower() in ("name", "string"):
            continue
        args[name] = value
    return args


def _parse_param_tags(body: str) -> dict:
    """Collect child ``<｜DSML｜ parameter>`` tags into an arguments dict."""
    args: dict = {}
    for pm in _DSML_PARAM_RE.finditer(body or ""):
        attrs = pm.group(1)
        args[_attr_value(attrs, "name") or "value"] = _dsml_value(pm.group(2), attrs)
    return args


def parse_dsml_calls(text: str) -> list[dict]:
    """Parse DeepSeek's native DSML invoke/parameter tool-call syntax.

    Handles, in one pass and in emission order: standard open/close invokes,
    self-closing invokes with no arguments, attribute-style arguments on a
    self-closing invoke, multiple invokes per message, and a truncated invoke
    with no closing tag (the partial call is still dispatched, and a WARN is
    logged as ``tool_call_truncated name=<name>``). Both the V4 (no space after
    the token) and V4.1 (leading space) tag forms are accepted.
    """
    text = text or ""
    calls: list[dict] = []
    for m in _DSML_OPEN_RE.finditer(text):
        attrs = m.group(1) or ""
        self_closing = m.group(2) == "/"
        name = _attr_value(attrs, "name")
        if not name:
            continue
        if self_closing:
            calls.append({"name": name, "arguments": _attrs_to_args(attrs)})
            continue
        close = _DSML_CLOSE_RE.search(text, m.end())
        if close is not None:
            body = text[m.end():close.start()]
            calls.append({"name": name, "arguments": _parse_param_tags(body)})
        else:
            # Truncated invoke: no close tag. Keep the partial call, parse what
            # arrived, and warn — the reference stream parser flushes on finish().
            body = text[m.end():]
            calls.append({"name": name, "arguments": _parse_param_tags(body)})
            log.warning("tool_call_truncated name=%s", name)
    return calls


def _dsml_value(raw: str, attrs: str):
    """Decode a parameter value using the reference rule.

    The reference (`decode_dsml_to_arguments`) treats ``string="true"`` values
    as verbatim strings and everything else as JSON. JSON that fails to parse
    is kept as the raw string rather than dropped, so a malformed argument
    still dispatches the call.
    """
    if re.search(r'string\s*=\s*"true"', attrs or "", re.IGNORECASE):
        value = raw
        if value.startswith("\n"):
            value = value[1:]
        return value[:-1] if value.endswith("\n") else value
    try:
        return json.loads(raw)
    except Exception:
        return raw


def strip_dsml(text: str) -> str:
    """Remove every DSML tool-call fragment so nothing leaks to the terminal."""
    if not text:
        return text
    out = _DSML_BLOCK_RE.sub("", text)
    # Bare invoke blocks (with or without the surrounding container).
    out = re.sub(
        r"<\s*\uff5cDSML\uff5c\s*invoke\b[^<>]*>.*?"
        r"</\s*\uff5cDSML\uff5c\s*invoke\s*>",
        "", out, flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(
        r"<\s*\uff5cDSML\uff5c\s*invoke\b[^<>]*/>",
        "", out, flags=re.IGNORECASE,
    )
    # Drop a truncated (unclosed) container or invoke too, so nothing leaks.
    m = _DSML_CALLS_OPEN_RE.search(out)
    if m is not None:
        out = out[:m.start()]
    m = _DSML_INVOKE_OPEN_BARE_RE.search(out)
    if m is not None:
        out = out[:m.start()]
    # Any leftover parameter tags.
    out = _DSML_PARAM_ANY_RE.sub("", out)
    return out


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """Parse ``<tool_call>`` envelopes out of an assistant message.

    Returns ``(calls, clean_text)`` where each call is
    ``{"name": str, "arguments": dict}``. Text outside the blocks is the
    assistant's natural-language reply. Malformed JSON inside a block is ignored
    (the block is still removed from the reply) rather than raising.
    """
    if not text:
        return [], ""

    calls: list[dict] = []
    for raw in _TOOL_CALL_BLOCK_RE.findall(text):
        parsed = _loads_tool_obj(raw)
        if parsed is not None:
            calls.append(parsed)

    clean = _TOOL_CALL_BLOCK_RE.sub("", text)

    # Fallback: some models wrap the envelope in a ```json fence without the
    # <tool_call> tags. Only accept a fenced object that clearly names a tool.
    if not calls:
        for raw in _FENCED_JSON_RE.findall(clean):
            parsed = _loads_tool_obj(raw)
            if parsed is not None and parsed.get("name"):
                calls.append(parsed)
                clean = clean.replace(raw, "")
                clean = re.sub(r"```(?:json)?\s*```", "", clean)

    # Native DeepSeek DSML syntax (what the free web model actually emits).
    calls.extend(parse_dsml_calls(text))
    clean = strip_dsml(clean)

    return calls, clean.strip()


def _loads_tool_obj(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    if not name:
        return None
    if "arguments" in obj:
        args = obj.get("arguments")
    else:
        args = obj.get("parameters", obj.get("input", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": str(name), "arguments": args}


# --------------------------------------------------------------------------- #
# 6. Tool registry (ported from nanocode + cipher-harness)                     #
# --------------------------------------------------------------------------- #

class ToolSpec:
    def __init__(self, name: str, description: str, parameters: dict,
                 fn: Callable[[dict, "Toolbox"], str],
                 read_only: bool = True, concurrency_safe: bool = True,
                 destructive: bool = False):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn
        self.read_only = read_only
        self.concurrency_safe = concurrency_safe
        self.destructive = destructive

    def openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _schema(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _is_rich_console(console) -> bool:
    return _RICH and not isinstance(console, _PlainConsole)


def _diff_counts(old_lines: list[str], new_lines: list[str]) -> tuple[int, int]:
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added += new_end - new_start
        if tag in ("delete", "replace"):
            removed += old_end - old_start
    return added, removed


class Toolbox:
    """The set of tools mike exposes, plus the dispatcher."""

    BLOCKED_BASH = [
        "rm -rf /",
        ":(){ :|:& };:",
        "mkfs",
        "dd if=/dev/zero",
        "chmod -R 777 /",
        "> /dev/sda",
    ]

    def __init__(self, cwd: str, console=None):
        self.cwd = cwd
        self.todos_path = os.path.join(cwd, ".mike", "todos.json")
        self.console = console or _make_console()
        self.specs: "OrderedDict[str, ToolSpec]" = OrderedDict()
        self._register()

    # -- registry ---------------------------------------------------------

    def _register(self) -> None:
        self._add(ToolSpec(
            "read_file",
            "Read a file from disk. Returns numbered lines; use offset/limit for large files.",
            _schema({
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "offset": {"type": "integer", "description": "1-indexed first line (default 1)"},
                "limit": {"type": "integer", "description": "Max number of lines"},
            }, ["path"]),
            self._read_file, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "apply_patch",
            "Apply a Codex-style patch to add, update, delete, or move files. "
            "The input MUST be a single string beginning with '*** Begin Patch' "
            "and ending with '*** End Patch'. Use this for ALL file edits.",
            _schema({
                "input": {
                    "type": "string",
                    "description": "Full patch envelope including Begin/End markers",
                },
            }, ["input"]),
            self._apply_patch,
            read_only=False,
            concurrency_safe=False,
            destructive=True,
        ))
        self._add(ToolSpec(
            "list_dir",
            "List the entries of a directory (directories suffixed with '/').",
            _schema({
                "path": {"type": "string", "description": "Directory (default: current working dir)"},
            }, []),
            self._list_dir, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "glob",
            "Find files matching a glob pattern, newest first. Example: '**/*.py'.",
            _schema({
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Base directory (default '.')"},
            }, ["pattern"]),
            self._glob, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "grep",
            "Search file contents for a regex. Uses ripgrep when available, else a Python walk.",
            _schema({
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "File or directory (default '.')"},
                "glob": {"type": "string", "description": "Only search files matching this glob"},
                "case_insensitive": {"type": "boolean"},
                "context": {"type": "integer", "description": "Lines of context"},
            }, ["pattern"]),
            self._grep, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "bash",
            "Run a shell command in the working directory and return its combined output.",
            _schema({
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (default 120)"},
            }, ["command"]),
            self._bash, read_only=False, concurrency_safe=False, destructive=True,
        ))
        self._add(ToolSpec(
            "python",
            "Execute a Python snippet with the current interpreter and return its output.",
            _schema({
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout": {"type": "integer"},
            }, ["code"]),
            self._python, read_only=False, concurrency_safe=False, destructive=True,
        ))
        self._add(ToolSpec(
            "web_fetch",
            "Fetch a URL and return its readable text (scripts/styles stripped).",
            _schema({
                "url": {"type": "string"},
                "selector": {"type": "string", "description": "Optional CSS selector"},
                "max_chars": {"type": "integer", "description": "Truncate output (default 8192)"},
            }, ["url"]),
            self._web_fetch, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "update_plan",
            "Create or replace the current structured plan for a multi-step task. "
            "Use one item per step and keep its status current.",
            _schema({
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "description": "Concrete action to perform",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["step", "status"],
                    },
                    "description": "The complete plan, in execution order",
                },
            }, ["plan"]),
            self._update_plan, read_only=False, concurrency_safe=False,
        ))
        self._add(ToolSpec(
            "todo_read",
            "Read the current structured plan.",
            _schema({}, []),
            self._todo_read, read_only=True, concurrency_safe=True,
        ))
        self._add(ToolSpec(
            "task_complete",
            "Call this ONLY when the user's task is fully done. "
            "Provide a one-paragraph summary of what was accomplished. "
            "Do not call it if you still have work to do.",
            _schema({
                "summary": {"type": "string", "description": "What was accomplished"},
            }, ["summary"]),
            self._task_complete,
            read_only=True,
            concurrency_safe=True,
        ))

    def _add(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    # -- dispatch ---------------------------------------------------------

    def call(self, name: str, args: dict) -> str:
        spec = self.specs.get(name)
        if spec is None:
            return f"error: unknown tool '{name}'"
        if not isinstance(args, dict):
            args = {}
        try:
            return spec.fn(args, self)
        except Exception as exc:  # never let a tool crash the loop
            return f"error: {type(exc).__name__}: {exc}"

    def openai_tools(self) -> list:
        return [s.openai_tool() for s in self.specs.values()]

    # -- implementations --------------------------------------------------

    def _resolve(self, path: str) -> str:
        path = os.path.expanduser(str(path))
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return os.path.normpath(path)

    def _read_file(self, args: dict, ctx: "Toolbox") -> str:
        path = self._resolve(args["path"])
        if not os.path.exists(path):
            return f"error: file not found: {path}"
        if not os.path.isfile(path):
            return f"error: not a file: {path}"
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
        offset = int(args.get("offset", 1) or 1)
        limit = args.get("limit")
        start = max(0, offset - 1)
        end = start + int(limit) if limit else len(lines)
        selected = lines[start:end]
        if not selected:
            return "(empty)"
        return "".join(f"{start + i + 1:>5}\t{ln}" for i, ln in enumerate(selected)).rstrip("\n")

    def _apply_patch(self, args: dict, ctx: "Toolbox") -> str:
        patch_text = args.get("input")
        if not isinstance(patch_text, str):
            return "error: apply_patch requires a string 'input'"
        try:
            operations = self._parse_patch(patch_text)
            return self._execute_patch(operations)
        except ValueError as exc:
            return f"error: {exc}"

    @staticmethod
    def _parse_patch(patch_text: str) -> list[dict]:
        """Parse the complete Codex patch envelope without touching disk."""
        lines = patch_text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        if not nonempty:
            raise ValueError("patch is empty")
        if lines[nonempty[0]].strip() != "*** Begin Patch":
            raise ValueError("patch must start with '*** Begin Patch'")
        if lines[nonempty[-1]].strip() != "*** End Patch":
            raise ValueError("patch must end with '*** End Patch'")

        start, end = nonempty[0] + 1, nonempty[-1]
        operations: list[dict] = []
        i = start

        def is_file_op(line: str) -> bool:
            return line.startswith((
                "*** Add File: ",
                "*** Delete File: ",
                "*** Update File: ",
            ))

        while i < end:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if line.startswith("*** Add File: "):
                path = line[len("*** Add File: "):].strip()
                if not path:
                    raise ValueError("Add File requires a path")
                i += 1
                body: list[str] = []
                while i < end and not is_file_op(lines[i]):
                    body_line = lines[i]
                    if body_line.startswith("*** "):
                        raise ValueError(f"unexpected patch directive: {body_line}")
                    if not body_line.startswith("+"):
                        raise ValueError(
                            f"{path}: Add File body lines must start with '+'"
                        )
                    body.append(body_line[1:])
                    i += 1
                content = "".join(f"{body_line}\n" for body_line in body)
                if not content.endswith("\n"):
                    content += "\n"
                operations.append({"kind": "add", "path": path, "content": content})
                continue

            if line.startswith("*** Delete File: "):
                path = line[len("*** Delete File: "):].strip()
                if not path:
                    raise ValueError("Delete File requires a path")
                operations.append({"kind": "delete", "path": path})
                i += 1
                continue

            if line.startswith("*** Update File: "):
                path = line[len("*** Update File: "):].strip()
                if not path:
                    raise ValueError("Update File requires a path")
                i += 1
                move_to = None
                if i < end and lines[i].startswith("*** Move to: "):
                    move_to = lines[i][len("*** Move to: "):].strip()
                    if not move_to:
                        raise ValueError("Move to requires a path")
                    i += 1
                hunks: list[dict] = []
                while (i < end and not is_file_op(lines[i])
                       and not lines[i].startswith("*** Move to: ")):
                    if not lines[i].startswith("@@"):
                        if not lines[i].strip():
                            i += 1
                            continue
                        raise ValueError(
                            f"{path}: expected '@@' before update hunk; got {lines[i]!r}"
                        )
                    i += 1
                    entries: list[tuple[str, str]] = []
                    after_eof = False
                    while (i < end and not lines[i].startswith("@@")
                           and not is_file_op(lines[i])
                           and not lines[i].startswith("*** Move to: ")):
                        hunk_line = lines[i]
                        if hunk_line.strip() == "*** End of File":
                            entries.append(("eof", ""))
                            after_eof = True
                            i += 1
                            continue
                        if hunk_line.startswith("*** "):
                            raise ValueError(
                                f"{path}: unexpected directive in hunk: {hunk_line}"
                            )
                        if after_eof:
                            if not hunk_line.startswith("+"):
                                raise ValueError(
                                    f"{path}: only '+' lines may follow"
                                    " '*** End of File'"
                                )
                            entries.append(("add", hunk_line[1:]))
                        elif hunk_line == "" or hunk_line.startswith(" "):
                            entries.append((
                                "context",
                                hunk_line[1:] if hunk_line.startswith(" ") else "",
                            ))
                        elif hunk_line.startswith("-"):
                            entries.append(("remove", hunk_line[1:]))
                        elif hunk_line.startswith("+"):
                            entries.append(("add", hunk_line[1:]))
                        else:
                            raise ValueError(
                                f"{path}: hunk lines must start with ' ', '-', or '+'"
                            )
                        i += 1
                    hunks.append({"entries": entries})
                operations.append({
                    "kind": "update",
                    "path": path,
                    "move_to": move_to,
                    "hunks": hunks,
                })
                continue

            if line.startswith("*** Move to: "):
                raise ValueError("'*** Move to:' must immediately follow '*** Update File:'")
            raise ValueError(f"unexpected patch directive: {line}")

        return operations

    def _execute_patch(self, operations: list[dict]) -> str:
        """Validate all operations against a staged view, then write once."""
        staged: dict[str, dict] = {}
        deleted: set[str] = set()
        summaries: list[str] = []
        changes: list[dict] = []

        def load(path: str) -> dict:
            if path in deleted:
                raise ValueError(f"file not found: {path}")
            if path in staged:
                return staged[path]
            if not os.path.exists(path):
                raise ValueError(f"file not found: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"not a file: {path}")
            raw = Path(path).read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: not UTF-8 text") from exc
            newline = "\r\n" if "\r\n" in text else "\n"
            return {
                "lines": text.splitlines(),
                "newline": newline,
                "trailing": text.endswith(("\n", "\r")),
            }

        for op in operations:
            display = op["path"]
            path = self._resolve(display)
            kind = op["kind"]
            if kind == "add":
                content = op["content"]
                old_lines: list[str] = []
                if path in staged:
                    old_lines = list(staged[path]["lines"])
                elif path not in deleted and os.path.exists(path):
                    old_lines = list(load(path)["lines"])
                new_data = {
                    "lines": content.splitlines(),
                    "newline": "\n",
                    "trailing": True,
                }
                staged[path] = new_data
                deleted.discard(path)
                summaries.append(f"added {display} ({len(new_data['lines'])} lines)")
                changes.append({
                    "display": display,
                    "old": old_lines,
                    "new": list(new_data["lines"]),
                    "move_to": None,
                })
                continue

            if kind == "delete":
                data = load(path)
                staged.pop(path, None)
                deleted.add(path)
                summaries.append(f"deleted {display} ({len(data['lines'])} lines)")
                changes.append({
                    "display": display,
                    "old": list(data["lines"]),
                    "new": [],
                    "move_to": None,
                })
                continue

            data = load(path)
            lines = list(data["lines"])
            original_lines = list(lines)
            cursor = 0
            for hunk in op["hunks"]:
                entries = hunk["entries"]
                eof_at = next(
                    (idx for idx, entry in enumerate(entries) if entry[0] == "eof"),
                    None,
                )
                if eof_at is None:
                    before_eof = entries
                    after_eof: list[tuple[str, str]] = []
                else:
                    before_eof = entries[:eof_at]
                    after_eof = entries[eof_at + 1:]

                old_lines = [text for action, text in before_eof
                             if action in ("context", "remove")]
                if old_lines:
                    match_at = None
                    search_end = len(lines) - len(old_lines) + 1
                    for candidate in range(cursor, max(cursor, search_end)):
                        if lines[candidate:candidate + len(old_lines)] == old_lines:
                            match_at = candidate
                            break
                    if match_at is None:
                        raise ValueError(
                            f"{display}: context mismatch; expected line "
                            f"{old_lines[0]!r}"
                        )
                    if eof_at is not None and match_at + len(old_lines) != len(lines):
                        raise ValueError(
                            f"{display}: context mismatch; expected line "
                            f"{old_lines[-1]!r} at end of file"
                        )
                    replacement: list[str] = []
                    old_offset = 0
                    for action, text in before_eof:
                        if action == "context":
                            replacement.append(old_lines[old_offset])
                            old_offset += 1
                        elif action == "remove":
                            old_offset += 1
                        elif action == "add":
                            replacement.append(text)
                    lines[match_at:match_at + len(old_lines)] = replacement
                    cursor = match_at + len(replacement)
                else:
                    additions = [text for action, text in before_eof if action == "add"]
                    additions += [text for action, text in after_eof if action == "add"]
                    if additions:
                        if eof_at is not None:
                            lines.extend(additions)
                            cursor = len(lines)
                        else:
                            lines[cursor:cursor] = additions
                            cursor += len(additions)

                if eof_at is not None and old_lines:
                    eof_additions = [text for action, text in after_eof if action == "add"]
                    if eof_additions:
                        lines.extend(eof_additions)
                        cursor = len(lines)

            data = dict(data)
            data["lines"] = lines
            staged[path] = data

            hunk_count = len(op["hunks"])
            change = {
                "display": display,
                "old": original_lines,
                "new": list(lines),
                "move_to": op.get("move_to"),
            }
            changes.append(change)
            if hunk_count or not op.get("move_to"):
                added, removed = _diff_counts(original_lines, lines)
                summaries.append(
                    f"updated {display} ({hunk_count} hunks) +{added} -{removed}"
                )
            if op.get("move_to"):
                move_display = op["move_to"]
                move_path = self._resolve(move_display)
                if move_path != path:
                    staged.pop(path, None)
                    staged[move_path] = data
                    deleted.discard(move_path)
                    deleted.add(path)
                    summaries.append(f"moved {display} -> {move_display}")

        prepared: list[tuple[str, str]] = []
        try:
            for path, data in staged.items():
                content = data["newline"].join(data["lines"])
                if data["trailing"] and data["lines"]:
                    content += data["newline"]
                parent = os.path.dirname(path) or "."
                os.makedirs(parent, exist_ok=True)
                tmp = f"{path}.mike-patch-{os.getpid()}"
                with open(tmp, "w", encoding="utf-8", newline="") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                prepared.append((path, tmp))
            for path, tmp in prepared:
                os.replace(tmp, path)
            for path in deleted:
                if os.path.exists(path):
                    os.remove(path)
        finally:
            for _path, tmp in prepared:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

        self._render_patch_diffs(changes)
        return "\n".join(summaries)

    def _render_patch_diffs(self, changes: list[dict]) -> None:
        """Display validated patch changes as syntax-highlighted unified diffs."""
        for change in changes:
            display = change["display"]
            target = change.get("move_to") or display
            diff_text = "\n".join(difflib.unified_diff(
                change["old"],
                change["new"],
                fromfile=f"a/{display}",
                tofile=f"b/{target}",
                lineterm="",
            ))
            added, removed = _diff_counts(change["old"], change["new"])
            if not diff_text:
                if change.get("move_to"):
                    self.console.print(
                        Text(f"• moved {display} → {target}", style="bold cyan")
                    )
                continue
            if _is_rich_console(self.console):
                header = Text("• ", style="dim")
                header.append(target, style="bold")
                header.append(f"  +{added}", style="green")
                header.append(f"  -{removed}", style="red")
                self.console.print(header)
                self.console.print(Syntax(
                    diff_text,
                    "diff",
                    theme="monokai",
                    line_numbers=True,
                ))
            else:
                self.console.print(diff_text)

    @staticmethod
    def _task_complete(args: dict, ctx: "Toolbox") -> str:
        return "task marked complete"

    def _list_dir(self, args: dict, ctx: "Toolbox") -> str:
        path = self._resolve(args.get("path", "."))
        if not os.path.isdir(path):
            return f"error: not a directory: {path}"
        entries = []
        with os.scandir(path) as it:
            for e in sorted(it, key=lambda x: (not x.is_dir(), x.name.lower())):
                entries.append(e.name + ("/" if e.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty directory)"

    def _glob(self, args: dict, ctx: "Toolbox") -> str:
        pattern = str(args["pattern"])
        base = self._resolve(args.get("path", "."))
        matches = _globlib.glob(os.path.join(base, pattern), recursive=True)
        matches = [m for m in matches if os.path.isfile(m)]
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if not matches:
            return "no files matched"
        return "\n".join(matches[:200])

    def _grep(self, args: dict, ctx: "Toolbox") -> str:
        pattern = str(args["pattern"])
        path = self._resolve(args.get("path", "."))
        glob_filter = args.get("glob")
        ci = bool(args.get("case_insensitive"))
        context_lines = int(args.get("context", 0) or 0)

        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
            if ci:
                cmd.append("-i")
            if context_lines:
                cmd += ["-C", str(context_lines)]
            if glob_filter:
                cmd += ["-g", str(glob_filter)]
            cmd += [pattern, path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode == 1:
                return "no matches"
            out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
            lines = out.splitlines()
            if len(lines) > 200:
                out = "\n".join(lines[:200]) + f"\n... ({len(lines)} matches total)"
            return out or "no matches"

        try:
            rx = re.compile(pattern, re.IGNORECASE if ci else 0)
        except re.error as exc:
            return f"error: bad regex: {exc}"
        hits: list[str] = []
        for dirpath, _dirs, files in os.walk(path):
            for fn in files:
                if glob_filter and not _globlib.fnmatch.fnmatch(fn, glob_filter):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, "r", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                hits.append(f"{fp}:{i}:{line.rstrip()}")
                                if len(hits) >= 200:
                                    break
                except Exception:
                    continue
                if len(hits) >= 200:
                    break
        return "\n".join(hits) if hits else "no matches"

    def _bash(self, args: dict, ctx: "Toolbox") -> str:
        cmd = str(args["command"])
        timeout = int(args.get("timeout", 120) or 120)
        for blocked in self.BLOCKED_BASH:
            if blocked in cmd:
                return f"error: refused - blocked dangerous pattern {blocked!r}"
        try:
            proc = subprocess.run(cmd, shell=True, cwd=self.cwd,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            return f"(exit {proc.returncode})\n{out}" if out else f"(exit {proc.returncode})"
        return out or "(no output)"

    def _python(self, args: dict, ctx: "Toolbox") -> str:
        code = str(args["code"])
        timeout = int(args.get("timeout", 120) or 120)
        try:
            proc = subprocess.run([sys.executable, "-c", code], cwd=self.cwd,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"error: timed out after {timeout}s"
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            return f"(exit {proc.returncode})\n{out}" if out else f"(exit {proc.returncode})"
        return out or "(no output)"

    def _web_fetch(self, args: dict, ctx: "Toolbox") -> str:
        url = str(args["url"])
        selector = args.get("selector")
        max_chars = int(args.get("max_chars", 8192) or 8192)
        try:
            import httpx
        except Exception:
            return "error: httpx is not installed"
        try:
            resp = httpx.get(url, timeout=20, follow_redirects=True, headers={
                "User-Agent": "mike/1.0 (+local coding harness)",
            })
            resp.raise_for_status()
            body = resp.text
        except Exception as exc:
            return f"error: fetch failed: {exc}"
        text = _html_to_text(body, selector)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated at {max_chars} chars)"
        return text or "(no readable text)"

    def load_plan(self) -> list[dict]:
        """Load the current plan, accepting v0.4's old string-only todo format."""
        try:
            with open(self.todos_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        plan: list[dict] = []
        for item in data:
            if isinstance(item, str):
                step, status = item, "pending"
            elif isinstance(item, dict):
                step = str(item.get("step", "")).strip()
                status = str(item.get("status", "pending")).strip().lower()
            else:
                continue
            if not step:
                continue
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"
            plan.append({"step": step, "status": status})
        return plan

    def _save_plan(self, plan: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.todos_path), exist_ok=True)
        atomic_write_json(Path(self.todos_path), plan)

    def _update_plan(self, args: dict, ctx: "Toolbox") -> str:
        raw_plan = args.get("plan", [])
        if not isinstance(raw_plan, list):
            return "error: 'plan' must be a list of objects"
        plan: list[dict] = []
        for index, item in enumerate(raw_plan, 1):
            if not isinstance(item, dict):
                return f"error: plan item {index} must be an object"
            step = str(item.get("step", "")).strip()
            status = str(item.get("status", "")).strip().lower()
            if not step:
                return f"error: plan item {index} requires a non-empty 'step'"
            if status not in ("pending", "in_progress", "completed"):
                return (
                    f"error: plan item {index} has invalid status {status!r}; "
                    "use pending, in_progress, or completed"
                )
            plan.append({"step": step, "status": status})
        self._save_plan(plan)
        self._render_plan(plan)
        return self._format_plan(plan)

    def _todo_read(self, args: dict, ctx: "Toolbox") -> str:
        return self._format_plan(self.load_plan())

    def _render_plan(self, plan: list[dict]) -> None:
        completed = sum(1 for item in plan if item["status"] == "completed")
        title = f"Plan · {completed}/{len(plan)} complete"
        if _is_rich_console(self.console):
            self.console.print(Text(title, style="bold cyan"))
            table = Table.grid(padding=(0, 1))
            table.add_column(width=1)
            table.add_column()
            for item in plan:
                marker, style = {
                    "completed": ("✓", "green"),
                    "in_progress": ("⠋", "yellow"),
                    "pending": ("○", "grey62"),
                }[item["status"]]
                table.add_row(Text(marker, style=f"bold {style}"), Text(item["step"]))
            self.console.print(table)
        else:
            self.console.print(self._format_plan(plan))

    @staticmethod
    def _format_plan(plan: list[dict]) -> str:
        if not plan:
            return "Plan · 0/0 complete\n○ (no steps)"
        completed = sum(1 for item in plan if item["status"] == "completed")
        markers = {
            "completed": "✓",
            "in_progress": "⠋",
            "pending": "○",
        }
        lines = [f"Plan · {completed}/{len(plan)} complete"]
        lines.extend(f"{markers[item['status']]} {item['step']}" for item in plan)
        return "\n".join(lines)

    @staticmethod
    def _format_todos(todos: list) -> str:
        """Compatibility helper for callers that only have string todos."""
        plan = []
        for item in todos:
            if isinstance(item, dict):
                plan.append({
                    "step": str(item.get("step", "")),
                    "status": str(item.get("status", "pending")),
                })
            else:
                plan.append({"step": str(item), "status": "pending"})
        return Toolbox._format_plan(plan)


def _html_to_text(body: str, selector: Optional[str] = None) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        if selector:
            els = soup.select(selector)
            if els:
                return re.sub(r"\n{3,}", "\n\n",
                              "\n\n".join(e.get_text("\n", strip=True) for e in els)).strip()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True)).strip()
    except Exception:
        pass
    stripped = re.sub(r"(?is)<(script|style|noscript|header|footer|nav).*?</\1>", " ", body)
    stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
    text = _html.unescape(stripped)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 7. Compactor (logic ported from cipher-harness/compact.py)                   #
# --------------------------------------------------------------------------- #

class Compactor:
    """Summarise older turns once the context grows past the threshold."""

    def __init__(self, counter: TokenCounter,
                 max_context: Optional[int] = None,
                 threshold: Optional[float] = None,
                 keep_recent: Optional[int] = None,
                 summarizer: Optional[Callable[[str], str]] = None):
        self.counter = counter
        self.max_context = DEFAULTS["max_context"] if max_context is None else max_context
        self.threshold = DEFAULTS["compact_threshold"] if threshold is None else threshold
        self.keep_recent = DEFAULTS["keep_recent_turns"] if keep_recent is None else keep_recent
        self.summarizer = summarizer

    # -- estimation -------------------------------------------------------

    @property
    def trigger_tokens(self) -> int:
        return int(self.max_context * self.threshold)

    def estimate(self, messages: list) -> int:
        return self.counter.count_messages(messages)

    def should_compact(self, messages: list) -> bool:
        return self.estimate(messages) > self.trigger_tokens

    # -- summarisation ----------------------------------------------------

    def build_summary(self, messages: list) -> str:
        """Return a summary that ALWAYS contains the seven required sections."""
        sections = self._extract_sections(messages)
        if self.summarizer is not None:
            try:
                requested = self.summarizer(self._summary_prompt(messages))
                if requested:
                    sections.update(self._parse_sections(requested, sections))
            except Exception:
                pass  # deterministic summary is the safety net
        parts = []
        for header in SUMMARY_SECTIONS:
            body = (sections.get(header) or "").strip() or "-"
            parts.append(f"{header}\n{body}")
        return "\n\n".join(parts)

    def compact(self, messages: list) -> list:
        """Return a new history: one summary message + the recent turns verbatim."""
        if len(messages) <= self.keep_recent:
            summary = self.build_summary(messages)
            return [{"role": "user", "content": f"{COMPACTION_HEADER}\n\n{summary}"}]
        older = messages[:-self.keep_recent]
        tail = messages[-self.keep_recent:]
        summary = self.build_summary(older)
        head = {"role": "user", "content": f"{COMPACTION_HEADER}\n\n{summary}"}
        return [head] + list(tail)

    # -- deterministic extraction ----------------------------------------

    def _extract_sections(self, messages: list) -> dict:
        users = [m for m in messages if m.get("role") == "user"]
        touched = _files_touched(messages)

        goal = ""
        for m in users:
            text = strip_thinking(_content_text(m.get("content")))
            if (text and not text.startswith(COMPACTION_PREFIX)
                    and not text.startswith("[CONTEXT SUMMARY")):
                goal = text
                break

        last_user = ""
        for m in reversed(users):
            text = strip_thinking(_content_text(m.get("content")))
            if (text and not text.startswith(COMPACTION_PREFIX)
                    and not text.startswith("[CONTEXT SUMMARY")
                    and "<tool_result>" not in text):
                last_user = text
                break

        constraints = _find_lines(messages, ("must ", "never ", "do not ", "don't ", "constraint"))
        decisions = _find_lines(messages, ("decid", "decision", "chose", "approach", "instead of"))
        todos = _latest_todos(messages)
        next_steps = todos or [
            "Continue the current task from the last user intent.",
            "Verify changes (run the relevant tests/commands).",
        ]

        return {
            "## Goal": _clip(goal, 600),
            "## Constraints": _clip("\n".join(f"- {c}" for c in constraints[:8]), 600),
            "## Files Touched": _clip("\n".join(f"- {p}" for p in sorted(touched)), 600),
            "## Decisions Made": _clip("\n".join(f"- {d}" for d in decisions[:8]), 600),
            "## Open Todos": _clip("\n".join(f"- {t}" for t in todos[:12]), 600),
            "## Last User Intent": _clip(last_user, 400),
            "## Next Steps": _clip("\n".join(f"- {s}" for s in next_steps[:8]), 600),
        }

    @staticmethod
    def _summary_prompt(messages: list) -> str:
        return (
            "Summarise this coding-agent conversation. Reply with markdown that has "
            "EXACTLY these seven sections, in this order, each starting with '## ':\n"
            "## Goal\n## Constraints\n## Files Touched\n## Decisions Made\n"
            "## Open Todos\n## Last User Intent\n## Next Steps\n\n"
            "Be concise and concrete. Conversation:\n\n"
            + _serialize_messages(messages)
        )

    @staticmethod
    def _parse_sections(text: str, fallback: dict) -> dict:
        """Split a model summary by '## ' headers into the known sections."""
        found: dict = {}
        current = None
        buf: list = []
        for line in text.splitlines():
            if line.strip().startswith("## "):
                if current is not None:
                    found[current] = "\n".join(buf).strip()
                current = line.strip()
                buf = []
            elif current is not None:
                buf.append(line)
        if current is not None:
            found[current] = "\n".join(buf).strip()
        merged = dict(fallback)
        for header, body in found.items():
            for known in SUMMARY_SECTIONS:
                if header.lower().startswith(known.lower()) and body:
                    merged[known] = body
        return merged


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _serialize_messages(messages: list, limit: int = 500) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "?").upper()
        text = strip_thinking(_content_text(m.get("content")))
        for call in m.get("tool_calls", []) or []:
            fn = (call.get("function") or {})
            text += f"\n[tool: {fn.get('name')} {str(fn.get('arguments', ''))[:120]}]"
        parts.append(f"{role}: {text[:limit]}")
    return "\n\n".join(parts)


def _files_touched(messages: list) -> set:
    files: set = set()
    for m in messages:
        text = _content_text(m.get("content"))
        for match in re.finditer(r'"(?:path|file_path)"\s*:\s*"([^"]+)"', text):
            files.add(match.group(1))
        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
            parsed = _loads_tool_obj(match.group(1))
            if parsed and "path" in parsed["arguments"]:
                files.add(str(parsed["arguments"]["path"]))
        for call in m.get("tool_calls", []) or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if isinstance(args, dict) and args.get("path"):
                files.add(str(args["path"]))
        # Tool results mention paths in prose ("wrote N line(s) to /path",
        # "edited /path", "File not found: /path"); capture those too.
        for match in re.finditer(r"(?:to|edited|from|not found:)\s+(\S+)", text):
            candidate = match.group(1).strip("`'\".,)")
            if "/" in candidate and not candidate.startswith("http"):
                files.add(candidate)
    return files


def _find_lines(messages: list, keywords: tuple, max_lines: int = 8) -> list:
    hits: list = []
    for m in messages:
        text = _content_text(m.get("content"))
        if text.startswith((COMPACTION_PREFIX, "[CONTEXT SUMMARY")):
            continue
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in keywords) and 8 < len(line.strip()) < 200:
                cleaned = line.strip().lstrip("-*# ").strip()
                if cleaned and cleaned not in hits:
                    hits.append(cleaned)
    return hits[:max_lines]


def _latest_todos(messages: list) -> list:
    for m in reversed(messages):
        text = _content_text(m.get("content"))
        match = re.search(r"<tool_result>\s*(.*?)\s*</tool_result>", text, re.DOTALL)
        if match and "[ ]" in match.group(1):
            todos = []
            for line in match.group(1).splitlines():
                mm = re.match(r"\s*\d+\.\s*\[( |x)\]\s*(.+)", line)
                if mm:
                    todos.append(mm.group(2).strip())
            if todos:
                return todos
    return []


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


# --------------------------------------------------------------------------- #
# 8. Session store (JSON on disk, atomic, resumable, rollover)                 #
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_words: int = 6) -> str:
    text = (text or "").strip().lower()
    words = [w for w in _SLUG_RE.split(text) if w]
    return "-".join(words[:max_words]) or "session"


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


class Session:
    def __init__(self, data: dict, path: Optional[Path] = None):
        self.id: str = data.get("id") or str(uuid.uuid4())
        self.name: str = data.get("name") or "session"
        self.created_at: str = data.get("created_at") or _now_iso()
        self.updated_at: str = data.get("updated_at") or self.created_at
        self.model: str = data.get("model") or MODEL_LABEL
        self.messages: list = data.get("messages") or []
        self.compactions: list = data.get("compactions") or []
        # Upstream DeepSeek thread id. Kept so every turn — including tool
        # result feedback — continues the SAME server-side chat instead of
        # starting a new one on each request.
        self.conversation_id: Optional[str] = data.get("conversation_id")
        self.path: Optional[Path] = Path(path) if path else None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "messages": self.messages,
            "compactions": self.compactions,
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def load(cls, path: Path) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, path=path)

    def save(self) -> Path:
        self.updated_at = _now_iso()
        if self.path is None:
            raise RuntimeError("session has no path; created in memory only")
        atomic_write_json(self.path, self.to_dict())
        return self.path

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size if self.path else len(json.dumps(self.to_dict()))
        except Exception:
            return 0

    def last_user_preview(self, limit: int = 60) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                text = strip_thinking(_content_text(m.get("content")))
                text = re.sub(r"<tool_result>.*?</tool_result>", "", text, flags=re.DOTALL)
                text = re.sub(r"This is a continuation of session.*?follows:", "", text, flags=re.DOTALL)
                text = " ".join(text.split())
                if text:
                    return text[:limit]
        return "(empty)"


class SessionStore:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else (ROOT / "sessions")
        self.directory.mkdir(parents=True, exist_ok=True)

    def new_session(self, name: Optional[str] = None, first_message: str = "") -> Session:
        slug = slugify(name or first_message or "session")
        session = Session({
            "id": str(uuid.uuid4()),
            "name": slug,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "model": MODEL_LABEL,
            "messages": [],
            "compactions": [],
        })
        session.path = self._path_for(slug)
        session.save()
        return session

    def _path_for(self, slug: str) -> Path:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = self.directory / f"{stamp}__{slug}.json"
        counter = 1
        while candidate.exists():
            candidate = self.directory / f"{stamp}__{slug}-{counter}.json"
            counter += 1
        return candidate

    def unique_name(self, base: str) -> str:
        existing = {s.name for s in self.list_sessions()}
        if base not in existing:
            return base
        i = 2
        while f"{base}-{i}" in existing:
            i += 1
        return f"{base}-{i}"

    def rollover(self, old: Session, summary: str) -> Session:
        """Create ``<slug>-contN`` seeded with the continuation header + summary."""
        n = 1
        existing = {s.name for s in self.list_sessions()}
        while f"{old.name}-cont{n}" in existing:
            n += 1
        new_name = f"{old.name}-cont{n}"
        seeded = {
            "role": "user",
            "content": (
                f"This is a continuation of session {old.name}. Summary follows:\n\n{summary}"
            ),
        }
        new_session = Session({
            "id": str(uuid.uuid4()),
            "name": new_name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "model": MODEL_LABEL,
            "messages": [seeded],
            "compactions": [],
        })
        new_session.path = self._path_for(slugify(new_name))
        old.compactions.append({
            "at_tokens": None,
            "summary": summary,
            "new_session": new_session.name,
        })
        old.save()
        new_session.save()
        return new_session

    def list_sessions(self) -> list:
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                sessions.append(Session.load(path))
            except Exception:
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def find(self, key: str) -> Optional[Session]:
        sessions = self.list_sessions()
        if key.isdigit():
            idx = int(key)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]
        for s in sessions:
            if s.name == key:
                return s
        for s in sessions:
            if s.id == key:
                return s
        for s in sessions:
            if s.name.startswith(key):
                return s
        for s in sessions:
            if s.id.startswith(key):
                return s
        return None


# --------------------------------------------------------------------------- #
# 8b. Markdown transcript export                                               #
# --------------------------------------------------------------------------- #

_TOOL_RESULT_BODY_RE = re.compile(r"\s*<tool_result>\s*(.*?)\s*</tool_result>\s*$", re.DOTALL)
_CONT_RE = re.compile(r"^(.*)-cont\d+$")


def _is_tool_result(text: str) -> bool:
    return bool(text) and text.lstrip().startswith("<tool_result>")


def _tool_result_body(text: str) -> str:
    m = _TOOL_RESULT_BODY_RE.match(text or "")
    return m.group(1) if m else (text or "")


def _escape_user_text(text: str) -> str:
    """Preserve user text byte-for-byte; only guard a leading ``---`` line."""
    if text.startswith("---"):
        return "\\" + text
    return text


def _truncate(text: str, limit: int = 500) -> str:
    text = (text or "").replace("\r\n", "\n")
    return text if len(text) <= limit else text[:limit] + "\u2026"


def _rollover_parent(name: str) -> str:
    m = _CONT_RE.match(name or "")
    return m.group(1) if m else "\u2014"


def _call_args_json(call: dict) -> str:
    fn = call.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
    elif raw is None:
        obj = {}
    else:
        obj = raw
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _call_name(call: dict) -> str:
    return (call.get("function") or {}).get("name") or call.get("name") or "?"


def render_transcript(session: Session) -> str:
    """Render a session as a Markdown transcript (see README for the structure)."""
    header = [
        f"# Session: {session.name}",
        "",
        f"- **id:** {session.id}",
        f"- **created:** {session.created_at}",
        f"- **updated:** {session.updated_at}",
        f"- **model:** {session.model}",
        f"- **messages:** {len(session.messages)}",
        f"- **compactions:** {len(session.compactions)}",
        f"- **rollover-of:** {_rollover_parent(session.name)}",
    ]

    # Split the transcript into turns at each genuine user message (tool-result
    # wrappers, which are role:user in prompted mode, do NOT start a turn).
    turns: list = []
    current: Optional[list] = None
    for msg in session.messages:
        role = msg.get("role")
        text = _content_text(msg.get("content"))
        if role == "user" and not _is_tool_result(text):
            if current is not None:
                turns.append(current)
            current = [msg]
        else:
            if current is None:
                current = []
            current.append(msg)
    if current:
        turns.append(current)

    blocks = ["\n".join(header).rstrip("\n")]
    for number, turn in enumerate(turns, 1):
        blocks.append(_render_turn(number, turn))

    return "\n\n---\n\n".join(b.strip("\n") for b in blocks) + "\n"


def _render_turn(number: int, messages: list) -> str:
    out: list = []

    def add(line: str = "") -> None:
        out.append(line)

    first = messages[0]
    add(f"## Turn {number} \u2014 user")
    add("")
    add(_escape_user_text(_content_text(first.get("content"))))
    add("")

    i = 1
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "assistant":
            clean = strip_thinking(_content_text(msg.get("content")))
            reasoning = msg.get("reasoning") or ""
            calls = msg.get("tool_calls") or []

            # Collect the tool results that immediately follow this message.
            results: list = []
            j = i + 1
            while j < len(messages):
                nm = messages[j]
                ntext = _content_text(nm.get("content"))
                if nm.get("role") == "tool":
                    results.append(ntext)
                    j += 1
                elif nm.get("role") == "user" and _is_tool_result(ntext):
                    results.append(_tool_result_body(ntext))
                    j += 1
                else:
                    break

            add(f"## Turn {number} \u2014 assistant")
            add("")
            if clean:
                add(clean)
                add("")
            if reasoning:
                add("<details><summary>reasoning</summary>")
                add("")
                add(reasoning)
                add("")
                add("</details>")
                add("")
            if calls:
                add("**Tool calls:**")
                add("")
                for k, call in enumerate(calls):
                    add(f"- `{_call_name(call)}({_call_args_json(call)})`")
                    if k < len(results):
                        add(f"  - result: {_truncate(results[k])}")
                add("")
            i = j
            continue

        if role == "tool":
            add(f"## Turn {number} \u2014 tool")
            add("")
            add(_content_text(msg.get("content")))
            add("")
            i += 1
            continue

        # A stray user message inside a turn that is not a tool result.
        if role == "user" and not _is_tool_result(_content_text(msg.get("content"))):
            add(f"## Turn {number} \u2014 user")
            add("")
            add(_escape_user_text(_content_text(msg.get("content"))))
            add("")
        i += 1

    return "\n".join(out).rstrip("\n")


class ExportCancelled(RuntimeError):
    pass


def default_export_name(session: Session) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{session.name}__{stamp}.md"


def resolve_export_path(path_arg: Optional[str], session: Session, cfg: dict,
                        cwd: Optional[str] = None) -> Path:
    """Resolve the destination path for an export (dir vs file semantics)."""
    if not path_arg:
        return resolved_export_dir(cfg) / default_export_name(session)
    p = Path(os.path.expanduser(str(path_arg)))
    if not p.is_absolute():
        p = Path(cwd or os.getcwd()) / p
    if str(path_arg).endswith(("/", os.sep)) or p.is_dir():
        return p / default_export_name(session)
    return p


def export_session(session: Session, path: Path, force: bool = False,
                   console=None, interactive: bool = True) -> Path:
    """Write the session transcript to *path*. Never touches the session file."""
    console = console or _make_console()
    path = Path(path)
    if path.exists() and not force:
        console.print(f"[yellow]warning: {path} already exists[/yellow]")
        proceed = False
        if interactive and sys.stdin.isatty():
            try:
                answer = input("Overwrite? [y/N] ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                answer = ""
            proceed = answer.startswith("y")
        if not proceed:
            raise ExportCancelled(f"not overwriting {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_transcript(session)
    path.write_text(markdown, encoding="utf-8")
    return path


def do_export(session: Session, cfg: dict, path_arg: Optional[str],
              force: bool, console, interactive: bool = True) -> Optional[Path]:
    try:
        path = resolve_export_path(path_arg, session, cfg)
        written = export_session(session, path, force=force, console=console,
                                 interactive=interactive)
    except ExportCancelled as exc:
        console.print(f"[yellow]export cancelled: {exc}[/yellow]")
        return None
    except Exception as exc:
        console.print(f"[red]export failed: {exc}[/red]")
        return None
    size = written.stat().st_size
    console.print(f"[green]Exported \u2192 {written.resolve()} ({size} bytes)[/green]")
    return written


# --------------------------------------------------------------------------- #
# 9. Server supervisor (auto-launch / attach, idempotent)                      #
# --------------------------------------------------------------------------- #

class ServerStartError(RuntimeError):
    pass


class ServerManager:
    """Start the local DeepSeek-API server, or attach to one already running."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 root: Path = ROOT, console=None, no_server: bool = False,
                 server_cmd: str = ""):
        self.host = host if host is not None else DEFAULTS["server_host"]
        self.port = int(port) if port is not None else DEFAULTS["server_port"]
        self.root = Path(root)
        self.console = console or _make_console()
        self.no_server = no_server
        self.server_cmd = (server_cmd or "").strip()
        self.process: Optional[subprocess.Popen] = None
        self.owned = False
        self._log: deque = deque(maxlen=200)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def port_open(self, timeout: float = 0.3) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def _models_endpoint_ready(self) -> bool:
        try:
            import httpx

            resp = httpx.get(f"{self.base_url}/models", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> None:
        if self.port_open():
            # Someone already serves this port - reuse it, never claim ownership.
            self.owned = False
            return
        if self.no_server:
            raise ServerStartError(
                f"No server is listening on {self.host}:{self.port} and --no-server was passed."
            )
        env = os.environ.copy()
        env.setdefault("HOST", self.host)
        env.setdefault("PORT", str(self.port))
        if self.server_cmd:
            argv = shlex.split(self.server_cmd)
        else:
            argv = [sys.executable, "app.py"]
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=str(self.root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            raise ServerStartError(f"Failed to spawn the server: {exc}") from exc
        self.owned = True
        threading.Thread(target=self._drain_log, daemon=True).start()
        self._install_cleanup()
        self._wait_ready()

    def _drain_log(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log.append(line.rstrip("\n"))

    def _wait_ready(self, timeout: float = STARTUP_TIMEOUT) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                break
            if self._models_endpoint_ready():
                return
            time.sleep(0.25)
        tail = "\n".join(list(self._log)[-50:])
        self.stop()
        raise ServerStartError(
            "The DeepSeek-API server did not become ready within "
            f"{timeout:.0f}s.\nLast server output:\n{tail}"
        )

    def _install_cleanup(self) -> None:
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, self._wrap_signal(previous))
            except Exception:
                pass

    def _wrap_signal(self, previous):
        def handler(signum, frame):
            self.stop()
            if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                previous(signum, frame)
            else:
                raise KeyboardInterrupt
        return handler

    def restart_once(self) -> bool:
        """Restart the server if *we* own it. Returns True on success."""
        if not self.owned:
            return False
        self.stop()
        try:
            self.ensure_running()
            return True
        except ServerStartError:
            return False

    def stop(self) -> None:
        if not self.owned or self.process is None:
            return
        proc = self.process
        self.process = None
        self.owned = False
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# 10. OpenAI-compatible LLM client (openai SDK, else httpx)                    #
# --------------------------------------------------------------------------- #

class StreamEvent(dict):
    """Dict subclass for readability; keys: type, text/calls/finish_reason."""


def _is_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "connection" in name or "timeout" in name or "connect" in name:
        return True
    text = str(exc).lower()
    return any(k in text for k in ("connection refused", "connection reset",
                                   "remote end closed", "timed out", "broken pipe"))


def _is_rate_limited(exc: BaseException) -> bool:
    if "ratelimit" in type(exc).__name__.lower():
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()


class DeepSeekLLM:
    """Thin OpenAI-compatible client pointed at the local server.

    Uses the ``openai`` SDK when installed, otherwise falls back to ``httpx``
    with a hand-rolled SSE reader. Only ever talks to ``base_url`` (127.0.0.1).
    """

    def __init__(self, base_url: str, api_key: str = API_KEY, model: str = MODEL_ID,
                 timeout: float = 300.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._openai = None
        try:
            from openai import OpenAI  # type: ignore

            self._openai = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
            self.backend = "openai-sdk"
        except Exception:
            self.backend = "httpx"

    def chat_stream(self, messages: list, tools: Optional[list] = None,
                    thinking: bool = False, search: bool = False,
                    conversation_id: Optional[str] = None,
                    max_retries: int = 3) -> Iterator[StreamEvent]:
        """Yield StreamEvents: content / reasoning / tool_calls / conversation_id / done.

        Pass ``conversation_id`` to continue an existing upstream DeepSeek thread
        (so tool results and follow-up prompts land in the *same* chat instead of
        a fresh one). The server echoes the thread id back on the final chunk;
        the caller stores it and passes it on subsequent requests.
        """
        for attempt in range(max_retries):
            try:
                yield from self._chat_stream_once(
                    messages, tools, thinking, search, conversation_id)
                return
            except Exception as exc:
                if _is_rate_limited(exc) and attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

    def complete(self, messages: list, thinking: bool = False,
                 max_tokens: int = 2048) -> str:
        parts = []
        for event in self.chat_stream(messages, tools=None, thinking=thinking):
            if event["type"] == "content":
                parts.append(event["text"])
        return "".join(parts)

    # -- internals --------------------------------------------------------

    def _extra_body(self, thinking: bool, search: bool,
                    conversation_id: Optional[str] = None) -> dict:
        body = {"thinking": bool(thinking), "search": bool(search)}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return body

    def _chat_stream_once(self, messages, tools, thinking, search,
                          conversation_id=None) -> Iterator[StreamEvent]:
        if self._openai is not None:
            yield from self._stream_openai(messages, tools, thinking, search, conversation_id)
        else:
            yield from self._stream_httpx(messages, tools, thinking, search, conversation_id)

    @staticmethod
    def _finalize_tool_calls(tool_calls) -> list:
        calls = []
        for idx in sorted(tool_calls):
            entry = tool_calls[idx]
            try:
                args = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({
                "id": entry["id"] or f"call_{uuid.uuid4().hex[:8]}",
                "name": entry["name"],
                "arguments": args,
            })
        return calls

    def _stream_openai(self, messages, tools, thinking, search,
                       conversation_id=None) -> Iterator[StreamEvent]:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "extra_body": self._extra_body(thinking, search, conversation_id),
        }
        if tools:
            kwargs["tools"] = tools
        stream = self._openai.chat.completions.create(**kwargs)
        tool_calls: "OrderedDict[int, dict]" = OrderedDict()
        returned_cid = None
        for chunk in stream:
            cid = getattr(chunk, "conversation_id", None) or \
                (getattr(chunk, "model_extra", None) or {}).get("conversation_id")
            if cid:
                returned_cid = cid
            extra = getattr(chunk, "model_extra", None) or {}
            err = extra.get("error") or getattr(chunk, "error", None)
            if isinstance(err, dict) and err.get("message"):
                # Server-side upstream failure (see server/openai_format.error_frame).
                yield StreamEvent(type="error", message=str(err["message"]))
                continue
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                reasoning = (getattr(delta, "reasoning_content", None)
                             or getattr(delta, "reasoning", None))
                if reasoning:
                    yield StreamEvent(type="reasoning", text=str(reasoning))
                content = getattr(delta, "content", None)
                if content:
                    yield StreamEvent(type="content", text=str(content))
                for tc in (getattr(delta, "tool_calls", None) or []):
                    idx = getattr(tc, "index", 0) or 0
                    entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += fn.arguments
            if getattr(choice, "finish_reason", None):
                yield StreamEvent(type="finish", finish_reason=choice.finish_reason)
        if tool_calls:
            yield StreamEvent(type="tool_calls", calls=self._finalize_tool_calls(tool_calls))
        if returned_cid:
            yield StreamEvent(type="conversation_id", id=returned_cid)
        yield StreamEvent(type="done")

    def _stream_httpx(self, messages, tools, thinking, search,
                      conversation_id=None) -> Iterator[StreamEvent]:
        import httpx

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self._extra_body(thinking, search, conversation_id),
        }
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"}
        tool_calls: "OrderedDict[int, dict]" = OrderedDict()
        returned_cid = None
        with httpx.stream("POST", f"{self.base_url}/chat/completions",
                          json=body, headers=headers, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("conversation_id"):
                    returned_cid = obj["conversation_id"]
                err = obj.get("error")
                if isinstance(err, dict) and err.get("message"):
                    # Server-side upstream failure (see server/openai_format.error_frame).
                    yield StreamEvent(type="error", message=str(err["message"]))
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield StreamEvent(type="reasoning", text=delta["reasoning_content"])
                    if delta.get("content"):
                        yield StreamEvent(type="content", text=delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        yield StreamEvent(type="finish", finish_reason=choice["finish_reason"])
        if tool_calls:
            yield StreamEvent(type="tool_calls", calls=self._finalize_tool_calls(tool_calls))
        if returned_cid:
            yield StreamEvent(type="conversation_id", id=returned_cid)
        yield StreamEvent(type="done")


# --------------------------------------------------------------------------- #
# 11. System prompts                                                           #
# --------------------------------------------------------------------------- #

BASE_SYSTEM_PROMPT = f"""You are mike, an autonomous coding agent running on the user's machine.
You DO things - you use your tools instead of explaining how. Be concise and concrete.

Rules:
1. Read a file before editing it. Use `apply_patch` for every file edit.
2. For multi-step tasks, call `update_plan` before starting and update statuses as work advances.
3. Keep calling tools until the task is genuinely complete; don't stop after one step.
4. Never print raw JSON tool schemas or fake tool output to the user.
5. Use absolute or repo-relative paths. Never guess - use glob/grep/list_dir to find things.
6. After you finish, give a one-line summary of what changed and how to run/test it.

Before calling tools, write a short 1-2 sentence progress update as standard
text. Do not use `[commentary]` or `[final]` tags. When the task is fully
complete, call `task_complete`.

## Compaction awareness
The conversation may be summarized when it grows long. If you see a message
beginning “The conversation was compacted…”, treat it as authoritative context,
not as a new user request. Assume the last user intent is current. Do not
repeat completed work or re-send progress updates that were already delivered.

You always run as the single model {MODEL_LABEL}. Working directory: {{cwd}}.
Context budget: {{max_context:,}} tokens (auto-compaction at {{compact_at:,}}).
"""


def build_system_prompt(cwd: str, max_context: int, compact_at_tokens: int) -> str:
    return BASE_SYSTEM_PROMPT.format(cwd=cwd, max_context=max_context,
                                     compact_at=compact_at_tokens)


def build_prompted_tool_prompt(toolbox: Toolbox) -> str:
    lines = [
        "## Tools",
        "",
        "You can call tools. To call one, emit one or more blocks in EXACTLY this format:",
        "",
        "<tool_call>",
        '{"name": "<tool_name>", "arguments": { ... }}',
        "</tool_call>",
        "",
        "Rules for tool calls:",
        "- Emit ONLY the envelope for the call; put any message to the user OUTSIDE the blocks.",
        "- You may emit several <tool_call> blocks in one turn.",
        "- Arguments must be a JSON object matching the tool's parameters.",
        "- After the tools run you will receive a <tool_result> message; keep going until done.",
        "- For multi-step tasks, use `update_plan` first and keep each step's status current.",
        "- Use `todo_read` to inspect the current plan.",
        "- For file edits, ALWAYS use `apply_patch`; do not use shell redirection, `cat`, or `sed -i`.",
        "- Use ONLY this envelope for tools. Do NOT use any other tool syntax"
        " (no DSML tags, no bare function names, no fenced code blocks).",
        "",
        "Available tools:",
    ]
    for spec in toolbox.specs.values():
        lines.append(f"- {spec.name}: {spec.description}")
        lines.append(f"  parameters: {json.dumps(spec.parameters, ensure_ascii=False)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 12. The agent                                                                #
# --------------------------------------------------------------------------- #

class MikeAgent:
    """The agentic loop: stream, parse tools, execute, repeat."""

    def __init__(self, llm: DeepSeekLLM, session: Session, store: SessionStore,
                 compactor: Compactor, toolbox: Toolbox, console,
                 tool_mode: str = "prompted", thinking_mode: str = "spinner",
                 server: Optional[ServerManager] = None,
                 cwd: Optional[str] = None, cfg: Optional[dict] = None):
        self.llm = llm
        self.session = session
        self.store = store
        self.compactor = compactor
        self.toolbox = toolbox
        self.console = console
        self.tool_mode = tool_mode
        self.thinking_mode = thinking_mode  # spinner | on | off
        self.server = server
        self.cwd = cwd or os.getcwd()
        self.cfg = cfg or dict(DEFAULTS)
        self.last_reasoning = self._last_reasoning_from_history()
        self._system = build_system_prompt(
            self.cwd, self.compactor.max_context, self.compactor.trigger_tokens
        )
        # `session.messages` is the FULL, append-only history kept on disk.
        # `self.context` is what we actually send upstream; compaction shrinks
        # only this, so a crash or a compact never loses the transcript.
        self.context: list = list(self.session.messages)
        # Messages the upstream thread has not seen yet (new user turns and tool
        # results). When a conversation_id exists we send only these, so a
        # single DeepSeek chat grows turn by turn instead of many throwaway ones.
        self._unsent: list = []
        self._has_called_tools_in_turn = False

    def _last_reasoning_from_history(self) -> str:
        for msg in reversed(self.session.messages):
            if msg.get("reasoning"):
                return msg["reasoning"]
        return ""

    # -- helpers ----------------------------------------------------------

    def _save(self) -> None:
        try:
            self.session.save()
        except Exception as exc:  # never lose data silently
            self.console.print(f"[red]warning: could not save session: {exc}[/red]")

    def _api_messages(self) -> list:
        """Build the OpenAI message list sent upstream."""
        out = [{"role": "system", "content": self._system}]
        if self.tool_mode == "prompted":
            out.append({"role": "system", "content": build_prompted_tool_prompt(self.toolbox)})
        for msg in self.context:
            role = msg.get("role", "user")
            if role == "assistant":
                entry = self._render_outgoing(msg)
                if (not entry["content"] and not entry.get("tool_calls")
                        and not entry.get("reasoning_content")):
                    continue
                out.append(entry)
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
            else:
                out.append({"role": role, "content": msg.get("content", "")})
        return out

    def _render_outgoing(self, msg: dict) -> dict:
        """Convert one stored message to the shape sent upstream."""
        role = msg.get("role", "user")
        if role == "assistant":
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning")
            if self.tool_mode == "native":
                entry: dict = {
                    "role": "assistant",
                    "content": content or None,
                }
                if reasoning:
                    entry["reasoning_content"] = str(reasoning)
            else:
                if reasoning:
                    content = (
                        f"<reasoning>\n{reasoning}\n</reasoning>\n{content}"
                    )
                entry = {"role": "assistant", "content": content}
            if msg.get("tool_calls") and self.tool_mode == "native":
                entry["tool_calls"] = msg["tool_calls"]
            return entry
        if role == "tool":
            return {"role": "tool", "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "")}
        return {"role": role, "content": msg.get("content", "")}

    def _request_messages(self) -> list:
        """Choose what to send upstream for the next request.

        * No upstream thread yet -> the full context (system + tools + history),
          which starts a new DeepSeek chat and returns its ``conversation_id``.
        * Thread exists -> only the messages added since the last request (the
          new user turn and/or tool results), so everything stays in that one
          chat instead of spawning a new one per request.
        """
        if self.session.conversation_id:
            pending = list(self._unsent)
            if not pending:
                pending = self.context[-1:] or [{"role": "user", "content": "(continue)"}]
            return [self._render_outgoing(m) for m in pending]
        return self._api_messages()

    # -- main turn --------------------------------------------------------

    def run(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""
        self._has_called_tools_in_turn = False

        if not self.session.messages:
            self._name_from(user_input)

        self._append({"role": "user", "content": user_input}, unsent=True)
        self._save()
        self._maybe_compact()

        final_text = ""
        nudge_count = 0
        dsml_retry = False
        guard = 0
        while True:
            guard += 1
            if guard > 40:
                self.console.print("[yellow]stopping: too many tool iterations[/yellow]")
                break
            text, reasoning, native_calls, finish, stream_error = self._stream_once()
            # The request reached the server (and the upstream thread): the
            # pending messages are now inside the conversation.
            self._unsent = []
            if reasoning:
                self.last_reasoning = reasoning

            if self.tool_mode == "native":
                calls = native_calls
                clean_text = strip_thinking(text)
            else:
                calls, clean_text = parse_tool_calls(text)
                clean_text = strip_thinking(clean_text)
                if not reasoning:
                    maybe_reasoning, _ = extract_thinking(text)
                    if maybe_reasoning:
                        self.last_reasoning = maybe_reasoning

            # Strip any echoed <tool_result> wrapper the model leaked into its reply
            # (both tool modes). Idempotent; the stored assistant content stays clean
            # so the next turn's history has no stray fragments.
            clean_text = strip_channel_tags(strip_tool_results(clean_text))

            if stream_error:
                # The server sent a terminal error frame (Fix 1). Show the exact
                # upstream reason instead of silently returning to the prompt, and
                # stop the turn without appending an empty assistant message.
                self.console.print(f"[red]error: {stream_error}[/red]")
                self.console.print(
                    "[dim]This usually means the session was invalidated. Run /auth "
                    "to re-authenticate, or check the server log for the exact error.[/dim]")
                return stream_error

            if clean_text.strip() or (reasoning and self.thinking_mode == "on"):
                self._render_assistant_response(clean_text, reasoning)

            assistant_msg: dict = {"role": "assistant", "content": clean_text}
            if reasoning:
                assistant_msg["reasoning"] = reasoning

            complete_call = next(
                (call for call in calls if call.get("name") == "task_complete"),
                None,
            )
            if complete_call is not None:
                complete_args = complete_call.get("arguments") or {}
                summary = complete_args.get("summary", "") if isinstance(
                    complete_args, dict) else ""
                if not isinstance(summary, str):
                    summary = str(summary)
                summary = strip_channel_tags(summary)
                # The signalling call is harness metadata, not a tool result. Do
                # not store or replay it in the assistant's tool_calls.
                self._append(assistant_msg)
                self._save()
                if clean_text.strip():
                    final_text = clean_text
                else:
                    final_text = summary
                    self._render_markdown(summary)
                break

            if calls:
                # Normalise a stable id onto every call dict FIRST, so the
                # assistant tool-call message and the tool result in _run_tools
                # reference the SAME id (the model pairs them by id). parse_*_calls
                # already sets ids in prompted mode; native calls may not.
                for c in calls:
                    if not c.get("id"):
                        c["id"] = f"call_{uuid.uuid4().hex[:8]}"
                # Always record the calls in the session (rich history + Files
                # Touched). They are only *sent* upstream in native mode.
                assistant_msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                        },
                    }
                    for c in calls
                ]
                self._append(assistant_msg)
                self._save()
                self._run_tools(calls)
                self._has_called_tools_in_turn = True
                self._save()
                self._maybe_compact()
                continue

            raw = text or ""
            looks_like_tool = (
                "DSML" in raw
                or "<tool_call>" in raw
                or "invoke name=" in raw
            )
            if finish == "stop" and looks_like_tool and not dsml_retry:
                dsml_retry = True
                log.warning("dsml_unparsed_retry")
                self.console.print(
                    "[yellow]warning: tool-call syntax detected in content but not parsed; "
                    "retrying with a stricter nudge[/yellow]")
                self._append({
                    "role": "user",
                    "content": (
                        "Your last message contained tool syntax but no executable tool call. "
                        "Emit a single valid <tool_call> envelope or a DSML invoke block now."
                    ),
                }, unsent=True)
                self._save()
                continue

            if not clean_text.strip() and not calls and not reasoning:
                # Truly empty stream with no error frame: same silent-blank failure.
                # Report it and stop; do NOT append an empty assistant message.
                self.console.print(
                    "[red]error: no response from the model — the upstream stream was empty.[/red]")
                self.console.print(
                    "[dim]This usually means the session was invalidated. Run /auth "
                    "to re-authenticate, or check the server log for the exact error.[/dim]")
                return ""

            self._append(assistant_msg)
            self._save()
            if self._has_called_tools_in_turn and nudge_count == 0:
                nudge_count += 1
                self._append({
                    "role": "user",
                    "content": (
                        "You have not called `task_complete`. Continue the task or "
                        "call `task_complete` with a summary."
                    ),
                }, unsent=True)
                self._save()
                continue
            final_text = clean_text
            break
        return final_text

    def _append(self, message: dict, unsent: bool = False) -> None:
        """Append to the full history, the in-context view, and the send queue."""
        self.session.messages.append(message)
        self.context.append(message)
        if unsent:
            self._unsent.append(message)

    def _name_from(self, first_message: str) -> None:
        """Name/label the session from its first user message (kept on disk)."""
        new_name = self.store.unique_name(slugify(first_message))
        old_path = self.session.path
        self.session.name = new_name
        self.session.path = self.store._path_for(new_name)
        try:
            if old_path and old_path.exists() and old_path != self.session.path:
                old_path.unlink()
        except Exception:
            pass

    # -- streaming --------------------------------------------------------

    def _stream_once(self):
        messages = self._request_messages()
        tools = self.toolbox.openai_tools() if self.tool_mode == "native" else None
        thinking = self.thinking_mode != THINKING_OFF

        for attempt in range(3):
            parser = StreamParser()
            native_calls: list = []
            finish = None
            stream_error: Optional[str] = None

            live = None
            if (self.thinking_mode in ("on", "spinner")
                    and _RICH and hasattr(self.console, "set_live")):
                live = Live(self._live_status(), console=self.console,
                            refresh_per_second=12, transient=True, auto_refresh=True)
                live.start()

            def stop_live():
                nonlocal live
                if live is not None:
                    live.stop()
                    live = None

            try:
                for event in self.llm.chat_stream(
                        messages, tools=tools, thinking=thinking,
                        conversation_id=self.session.conversation_id):
                    etype = event.get("type")
                    if etype == "conversation_id":
                        cid = event.get("id")
                        if cid and cid != self.session.conversation_id:
                            self.session.conversation_id = cid
                            self._save()
                    elif etype == "reasoning":
                        if live is not None:
                            live.update(self._live_status(
                                parser.reasoning + event["text"]
                            ))
                        parser.reasoning += event["text"]
                    elif etype == "content":
                        parsed_chunks = parser.feed(event["text"])
                        for kind, chunk in parsed_chunks:
                            if kind == "reasoning":
                                if live is not None:
                                    live.update(self._live_status(parser.reasoning))
                        if any(kind != "reasoning" for kind, _chunk in parsed_chunks):
                            stop_live()
                    elif etype == "tool_calls":
                        native_calls.extend(event.get("calls", []))
                    elif etype == "error":
                        stream_error = event.get("message") or "upstream error"
                        log.warning("upstream SSE error: %s", stream_error)
                    elif etype == "finish":
                        finish = event.get("finish_reason")
                for kind, chunk in parser.finish():
                    if kind == "reasoning":
                        parser.reasoning += chunk
            except Exception as exc:
                # The OpenAI SDK raises (rather than yields) when the server sends
                # an SSE error frame, and httpx/other backends may raise too.
                if not stream_error:
                    stream_error = str(exc) or exc.__class__.__name__
                log.warning("stream raised: %s", stream_error)
            finally:
                stop_live()

            is_empty = not parser.raw and not parser.reasoning and not native_calls
            if stream_error or not is_empty:
                return parser.raw, parser.reasoning, native_calls, finish, stream_error

            if attempt < 2:
                retry_number = attempt + 1
                log.warning("empty upstream stream; retrying...")
                self.console.print(
                    "[yellow]warning: empty response from upstream, retrying "
                    f"(attempt {retry_number}/2)...[/yellow]"
                )
                time.sleep(2.0)
                continue

            # All three attempts were empty. Return the final empty result and let
            # run() use the existing graceful "no response" error path.
            return parser.raw, parser.reasoning, native_calls, finish, stream_error

        return "", "", [], None, None

    def _live_status(self, reasoning: str = ""):
        estimate = self.compactor.estimate(self.context)
        context = f"Ctx: {estimate:,} / {self.compactor.max_context:,}"
        if self.thinking_mode == "on" and _RICH:
            return Panel(
                Text(reasoning or "Thinking...", style="dim"),
                title=f"Reasoning · {context}",
                border_style="dim",
                expand=False,
            )
        text = (
            f" Thinking... ({context})"
        )
        tail = reasoning[-80:].replace("\n", " ")
        if tail:
            text += f" · {tail}"
        if _RICH:
            return Spinner("dots", text=Text(text, style="dim"), style="cyan")
        return text

    def _spinner_text(self, reasoning: str):
        """Compatibility wrapper retained for tests and embedders."""
        return self._live_status(reasoning)

    def _render_markdown(self, text: str) -> None:
        if not text.strip():
            return
        if _is_rich_console(self.console):
            self.console.print(Markdown(text, code_theme="monokai"))
        else:
            self.console.print(text)

    def _render_assistant_response(self, text: str, reasoning: str = "") -> None:
        # Reasoning is shown only in the transient Live view so it collapses
        # when visible content starts. The stored copy remains available to
        # /thinking last and reasoning replay.
        self._render_markdown(text)

    # -- tools ------------------------------------------------------------

    def _run_tools(self, calls: list) -> None:
        if self.tool_mode == "native":
            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments", {}) or {}
                self.console.print(Text(f"* {name}({_tool_preview(name, args)})", style="cyan"))
                output = self.toolbox.call(name, args)
                self.console.print(Text(_result_preview(output), style="dim"))
                self._append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": output,
                }, unsent=True)
                log.info("tool result queued: name=%s id=%s bytes=%d",
                         name, call.get("id", ""), len(output or ""))
            return

        # Prompted mode. The reference encoder (encoding.py) has no standalone
        # "tool" role: tool results are wrapped in <tool_result> blocks and
        # MERGED into a single following user message, ordered to match the
        # assistant's tool_calls (merge_tool_messages + sort_tool_results_by_call_order).
        # We mirror that: one user message, one <tool_result> per call in order,
        # each tagged with the assistant's call id so the model can pair them and
        # continue the task instead of stopping.
        blocks: list[str] = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {}) or {}
            call_id = call.get("id", "")
            self.console.print(Text(f"* {name}({_tool_preview(name, args)})", style="cyan"))
            output = self.toolbox.call(name, args)
            self.console.print(Text(_result_preview(output), style="dim"))
            blocks.append(f'<tool_result name="{name}" id="{call_id}">\n{output}\n</tool_result>')
            log.info("tool result queued: name=%s id=%s bytes=%d",
                     name, call_id, len(output or ""))
        self._append({"role": "user", "content": "\n\n".join(blocks)}, unsent=True)

    # -- compaction / rollover -------------------------------------------

    def _maybe_compact(self) -> None:
        if self.compactor.should_compact(self.context):
            self.compact(force=False)

    def compact(self, force: bool = True) -> str:
        before = self.compactor.estimate(self.context)
        if not force and before <= self.compactor.trigger_tokens:
            return ""
        summary = self.compactor.build_summary(self.context)
        self.session.compactions.append({
            "at_tokens": before,
            "summary": summary,
            "new_session": None,
        })
        # Only the in-context view shrinks; session.messages keeps the full
        # transcript on disk (never lose user data).
        pending = list(self._unsent)
        self.context = self.compactor.compact(self.context)
        if self.session.conversation_id:
            # The upstream thread keeps the old turns, so send the fresh summary
            # plus anything still pending (e.g. the user turn we just added).
            pending = [
                m for m in pending
                if not str(m.get("content", "")).startswith(
                    (COMPACTION_PREFIX, "[CONTEXT SUMMARY")
                )
            ]
            self._unsent = [self.context[0]] + pending
        else:
            self._unsent = []
        self._save()
        after = self.compactor.estimate(self.context)
        if after > self.compactor.trigger_tokens:
            self.rollover(summary)
        return summary

    def rollover(self, summary: Optional[str] = None) -> Session:
        if summary is None:
            summary = self.compactor.build_summary(self.context)
        old = self.session
        new = self.store.rollover(old, summary)
        self.session = new
        self.context = list(new.messages)
        self._unsent = []
        self.console.print(Text(f"-> rolled over to session {new.name}", style="yellow"))
        return new


def _tool_preview(name: str, args: dict) -> str:
    if not isinstance(args, dict):
        return ""
    if name == "apply_patch" and "input" in args:
        val = str(args["input"]).replace("\n", " ")
        return val[:60] + ("..." if len(val) > 60 else "")
    for key in ("path", "command", "pattern", "url", "code"):
        if key in args:
            val = str(args[key]).replace("\n", " ")
            return val[:60] + ("..." if len(val) > 60 else "")
    return ", ".join(f"{k}={str(v)[:20]}" for k, v in list(args.items())[:2])


def _result_preview(output: str) -> str:
    lines = (output or "").splitlines()
    if not lines:
        return "  | (no output)"
    first = lines[0][:70]
    if len(lines) > 1:
        return f"  | {first} ... +{len(lines) - 1} lines"
    if len(lines[0]) > 70:
        return f"  | {first}..."
    return f"  | {first}"


# --------------------------------------------------------------------------- #
# 13. Tool-mode probe (native vs prompted), cached in .mike_state.json         #
# --------------------------------------------------------------------------- #

PING_TOOL = [{
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Health probe. Call this to confirm you can use tools.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
        },
    },
}]


def detect_tool_mode(llm: DeepSeekLLM) -> str:
    """Return 'native' if the backend returns structured tool_calls, else 'prompted'."""
    messages = [
        {"role": "system", "content": "You must use the provided tool. Reply only by calling it."},
        {"role": "user", "content": "Call the ping tool now."},
    ]
    try:
        saw_native = False
        for event in llm.chat_stream(messages, tools=PING_TOOL, thinking=False):
            if event.get("type") == "tool_calls" and event.get("calls"):
                saw_native = True
        return "native" if saw_native else "prompted"
    except Exception:
        # Can't probe (offline / hiccup) -> prompted works against any
        # text-completion backend, so it is the safe default.
        return "prompted"


# --------------------------------------------------------------------------- #
# 14. REPL + slash commands                                                    #
# --------------------------------------------------------------------------- #

SLASH_COMMANDS = OrderedDict([
    ("/help", "Show available commands"),
    ("/quit", "Exit and save the session"),
    ("/exit", "Exit and save the session"),
    ("/clear", "Clear the terminal"),
    ("/model", "Show the single available model"),
    ("/thinking", "Set reasoning display: on, off, or last"),
    ("/sessions", "List saved sessions"),
    ("/resume", "Resume a saved session"),
    ("/rename", "Rename the current session"),
    ("/export", "Export this session as Markdown"),
    ("/compact", "Compact the current context"),
    ("/tokens", "Show context usage"),
    ("/tools", "List available tools"),
    ("/rollover", "Start a continuation session"),
])


class MikeCompleter(Completer):
    """Slash-command and argument completion for the mike composer."""

    def __init__(self, agent: Optional["MikeAgent"] = None):
        self.agent = agent

    def get_completions(self, document, complete_event):
        if Completion is None:
            return
        before = document.text_before_cursor
        current_line = before.splitlines()[-1] if before.splitlines() else ""
        if not current_line.startswith("/"):
            return

        if " " not in current_line:
            prefix = current_line
            for command, description in SLASH_COMMANDS.items():
                if command.startswith(prefix):
                    yield Completion(
                        command,
                        start_position=-len(prefix),
                        display_meta=description,
                    )
            return

        command, arg_prefix = current_line.split(" ", 1)
        for candidate, description in self._argument_options(command):
            if candidate.startswith(arg_prefix):
                yield Completion(
                    candidate,
                    start_position=-len(arg_prefix),
                    display_meta=description,
                )

    def _argument_options(self, command: str) -> list[tuple[str, str]]:
        if command == "/thinking":
            return [
                ("on", "stream full reasoning"),
                ("off", "hide reasoning"),
                ("last", "show the latest reasoning block"),
                ("spinner", "animate a reasoning preview"),
            ]
        if command == "/model":
            return [(alias, "routes to DeepSeek V4.1 Flash") for alias in MODEL_ALIASES]
        if command == "/export":
            return [("--force", "overwrite an existing export")]
        if command == "/resume" and self.agent is not None:
            options: list[tuple[str, str]] = []
            try:
                for index, session in enumerate(
                        self.agent.store.list_sessions(), 1):
                    options.append((session.name, f"{index}. {session.last_user_preview(36)}"))
            except Exception:
                pass
            return options
        return []


def _display_cwd(cwd: str) -> str:
    path = os.path.abspath(os.path.expanduser(cwd))
    home = os.path.abspath(os.path.expanduser("~"))
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def render_status_bar(agent: MikeAgent) -> Any:
    """Render a compact Codex-like state row for the inline REPL."""
    estimate = agent.compactor.estimate(agent.context)
    plan = agent.toolbox.load_plan()
    completed = sum(1 for item in plan if item.get("status") == "completed")
    if not _RICH:
        return (
            f"{MODEL_LABEL} | Ctx {estimate:,} / "
            f"{agent.compactor.max_context:,} | Plan {completed}/{len(plan)} | "
            f"{_display_cwd(agent.cwd)}"
        )
    status = Text()
    status.append(MODEL_LABEL, style="bold cyan")
    status.append(" | Ctx ", style="dim")
    status.append(
        f"{estimate:,} / {agent.compactor.max_context:,}",
        style="bright_white",
    )
    status.append(" | Plan ", style="dim")
    status.append(f"{completed}/{len(plan)}", style="yellow")
    status.append(" | ", style="dim")
    status.append(_display_cwd(agent.cwd), style="green")
    return status


def _prompt_key_bindings():
    if not _PROMPT_TOOLKIT:
        return None
    bindings = KeyBindings()

    @bindings.add("c-m", eager=True)
    def _submit(event):
        buffer = event.current_buffer
        completion = (
            buffer.complete_state.current_completion
            if buffer.complete_state is not None else None
        )
        if completion is not None:
            buffer.apply_completion(completion)
        else:
            buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline_after_escape(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")
    def _newline_ctrl_enter(event):
        # Most terminals encode Ctrl+Enter as LINE FEED (Ctrl+J).
        event.current_buffer.insert_text("\n")

    return bindings


def _build_prompt_session(agent: MikeAgent):
    if not _PROMPT_TOOLKIT:
        return None
    style = Style.from_dict({
        "prompt": "bold ansicyan",
        "completion-menu.completion": "bg:#1f2937 #d1d5db",
        "completion-menu.completion.current": "bg:#2563eb #ffffff bold",
        "completion-menu.meta.completion": "bg:#111827 #9ca3af",
        "completion-menu.meta.completion.current": "bg:#1d4ed8 #dbeafe",
    })
    return PromptSession(
        history=FileHistory(str(ROOT / ".mike_history")),
        completer=MikeCompleter(agent),
        complete_while_typing=True,
        multiline=True,
        key_bindings=_prompt_key_bindings(),
        style=style,
        reserve_space_for_menu=10,
    )


HELP_TEXT = """\
Commands:
  /help                 show this help
  /quit, /exit          quit (saves the session)
  /clear                clear the screen
  /model [name]         show the model (there is exactly one)
  /thinking on|off|last thinking display: on / off / show last block
  /sessions             list saved sessions
  /resume <n|name>      switch to a saved session
  /rename <name>        rename the current session
  /export [--force] [path]  write this session to a Markdown transcript
  /compact              force compaction now
  /tokens               show the current token estimate + threshold
  /tools                list the available tools
  /rollover             force a session rollover

Input:
  Enter                 submit
  Esc, Enter / Ctrl+J   insert a newline
  Tab                   complete slash commands and arguments
  Up / Down             navigate input history
"""


def run_repl(agent: MikeAgent, console, cfg: Optional[dict] = None) -> None:
    cfg = cfg or getattr(agent, "cfg", None) or dict(DEFAULTS)
    _print_banner(agent, console)
    prompt_session = _build_prompt_session(agent)
    prompt = FormattedText([("class:prompt", "> ")]) if _PROMPT_TOOLKIT else "> "
    while True:
        try:
            console.print()
            console.print(render_status_bar(agent), no_wrap=True, overflow="ellipsis")
            if prompt_session is not None and sys.stdin.isatty():
                user_input = prompt_session.prompt(prompt).strip()
            else:
                user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nbye.")
            break
        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            console.print("bye.")
            break
        if user_input == "/help":
            console.print(HELP_TEXT)
            continue
        if user_input == "/clear":
            os.system("clear")
            continue
        if user_input == "/tools":
            for spec in agent.toolbox.specs.values():
                mode = "read-only" if spec.read_only else "read-write"
                console.print(f"  {spec.name} [{mode}] - {spec.description}")
            continue
        if user_input == "/tokens":
            est = agent.compactor.estimate(agent.context)
            console.print(
                f"estimated {est:,} tokens / threshold {agent.compactor.trigger_tokens:,} "
                f"(max {agent.compactor.max_context:,}) via {agent.compactor.counter.name}"
            )
            continue
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                resolve_model(parts[1].strip())  # accepted; single model either way
            console.print(f"{MODEL_LABEL} ({MODEL_ID}) - the only model")
            continue
        if user_input.startswith("/thinking"):
            parts = user_input.split()
            arg = parts[1] if len(parts) > 1 else ""
            if arg == "on":
                agent.thinking_mode = "on"
                console.print("thinking: full reasoning shown (dim)")
            elif arg == THINKING_OFF:
                agent.thinking_mode = THINKING_OFF
                console.print("thinking: hidden")
            elif arg == "last":
                if agent.last_reasoning:
                    console.print(Panel(Text(agent.last_reasoning, style="dim"),
                                        title="last reasoning", border_style="dim"))
                else:
                    console.print("(no reasoning captured yet)")
            else:
                console.print(f"thinking mode: {agent.thinking_mode} (use on|off|last)")
            continue
        if user_input == "/sessions":
            _print_sessions_table(agent.store, console)
            continue
        if user_input.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("usage: /resume <n|name>")
                continue
            chosen = agent.store.find(parts[1].strip())
            if chosen is None:
                console.print("no such session")
            else:
                agent.session = chosen
                agent.context = list(chosen.messages)
                agent._unsent = []
                console.print(f"resumed session {chosen.name} ({len(chosen.messages)} messages)")
            continue
        if user_input.startswith("/rename"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("usage: /rename <name>")
                continue
            new_name = slugify(parts[1])
            old_path = agent.session.path
            agent.session.name = new_name
            agent.session.path = agent.store._path_for(new_name)
            try:
                if old_path and old_path.exists() and old_path != agent.session.path:
                    old_path.unlink()
            except Exception:
                pass
            agent.session.save()
            console.print(f"renamed -> {new_name}")
            continue
        if user_input.startswith("/export"):
            rest = user_input[len("/export"):].strip()
            force = False
            if rest.startswith("--force"):
                force = True
                rest = rest[len("--force"):].strip()
            elif rest.split(" ", 1)[0] == "--force":
                rest = rest.split(" ", 1)[1].strip() if " " in rest else ""
            agent._save()
            do_export(agent.session, cfg, rest or None, force, console,
                      interactive=True)
            continue
        if user_input == "/compact":
            console.print("compacting...")
            summary = agent.compact(force=True)
            console.print(Panel(Text(summary, style="dim"), title="summary", border_style="dim"))
            continue
        if user_input == "/rollover":
            old_name = agent.session.name
            new = agent.rollover()
            console.print(f"-> rolled over to session {new.name} (from {old_name})")
            continue
        if user_input.startswith("/"):
            console.print(f"unknown command: {user_input}. Try /help.")
            continue

        try:
            agent.run(user_input)
        except KeyboardInterrupt:
            console.print("\n[interrupted]")
        except Exception as exc:
            _handle_crash(agent, console, exc)


def _print_banner(agent: MikeAgent, console) -> None:
    console.print(f"[bold]mike[/bold] v{__version__} - {MODEL_LABEL} "
                  f"[dim]({agent.tool_mode} tools, thinking:{agent.thinking_mode}, "
                  f"ctx {agent.compactor.max_context:,})[/dim]")
    console.print(f"[dim]server {agent.llm.base_url} | session {agent.session.name} | "
                  f"cwd {agent.cwd}[/dim]")
    console.print("[dim]type /help for commands[/dim]")


def _print_sessions_table(store: SessionStore, console) -> None:
    sessions = store.list_sessions()
    if not sessions:
        console.print("No sessions yet.")
        return
    if _RICH:
        table = Table(title="Sessions", show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("name", style="cyan")
        table.add_column("updated", style="dim")
        table.add_column("messages")
        table.add_column("size", style="dim")
        table.add_column("preview")
        for i, s in enumerate(sessions, 1):
            table.add_row(str(i), s.name, s.updated_at, str(len(s.messages)),
                          _human_size(s.size_bytes), s.last_user_preview())
        console.print(table)
    else:
        for i, s in enumerate(sessions, 1):
            console.print(f"{i}. {s.name} | {s.updated_at} | {len(s.messages)} msgs | "
                          f"{_human_size(s.size_bytes)} | {s.last_user_preview()}")


def _prompt_resume(store: SessionStore, console) -> Optional[Session]:
    sessions = store.list_sessions()
    if not sessions:
        console.print("No sessions yet.")
        return None
    _print_sessions_table(store, console)
    try:
        choice = input(f"Select a session to resume [1-{len(sessions)}, or q]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not choice or choice.lower() == "q":
        return None
    chosen = store.find(choice)
    if chosen is None:
        console.print("no such session; starting a new one.")
    return chosen


# --------------------------------------------------------------------------- #
# 15. Crash handling                                                           #
# --------------------------------------------------------------------------- #

def _handle_crash(agent: MikeAgent, console, exc: Exception) -> None:
    try:
        agent._save()
    except Exception:
        pass
    tb = traceback.format_exc()
    if _RICH:
        console.print(Panel(tb, title=f"{type(exc).__name__}: {exc}", border_style="red"))
    else:
        print(tb)
    console.print("[dim]session saved. Send this traceback to the model for a "
                  "self-diagnosis pass? (y/n, default y in 5s)[/dim]")
    if _ask_yes_no(timeout=5.0, default=True):
        try:
            diagnosis = agent.llm.complete([
                {"role": "system", "content": "You are debugging a Python coding-agent harness."},
                {"role": "user", "content": "Explain the likely cause and a fix:\n\n" + tb},
            ])
            console.print(Panel(diagnosis, title="self-diagnosis", border_style="yellow"))
        except Exception as secondary:
            console.print(f"[red]diagnosis failed too: {secondary}[/red]")


def _ask_yes_no(timeout: float = 5.0, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    try:
        ready, _, _ = _select.select([sys.stdin], [], [], timeout)
    except Exception:
        return default
    if not ready:
        return default
    line = sys.stdin.readline().strip().lower()
    if not line:
        return default
    return line.startswith("y")


# --------------------------------------------------------------------------- #
# 16. CLI                                                                      #
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mike",
        description=(
            "mike - a self-contained, zero-cost DeepSeek coding harness. "
            "Launches the local DeepSeek-API server, talks to its OpenAI-compatible "
            "endpoint, and runs an interactive agent with tools, compaction and "
            f"resumable sessions. Exactly one model is surfaced: {MODEL_LABEL}. "
            "Config is layered: defaults < ~/.config/mike/config.toml < ./mike.toml "
            "< MIKE_* env vars < CLI flags."
        ),
        epilog=("Examples:\n"
                "  python mike.py\n"
                "  python mike.py resume\n"
                "  python mike.py config\n"
                "  python mike.py --max-context 200000 config\n"
                "  python mike.py export my-session ./out.md"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=["resume", "config", "export"],
                        help="'resume' opens the session picker; 'config' prints the "
                             "resolved config; 'export' writes a session to Markdown")
    parser.add_argument("rest", nargs="*", help="arguments for the 'export' command")
    parser.add_argument("--resume", action="store_true",
                        help="same as the 'resume' command")
    parser.add_argument("--force", action="store_true",
                        help="allow /export or the export subcommand to overwrite")
    parser.add_argument("--no-server", action="store_true",
                        help="assume a server is already running; never launch one")
    parser.add_argument("--version", action="version", version=f"mike {__version__}")

    # Config-mirroring flags. default=SUPPRESS so an unset flag is simply absent
    # from the namespace — that is how `load_config` tells "cli" from "default".
    S = argparse.SUPPRESS
    parser.add_argument("--max-context", dest="max_context", type=int, default=S,
                        metavar="N", help="total token budget (compaction threshold × max)")
    parser.add_argument("--compact-threshold", dest="compact_threshold", type=float, default=S,
                        metavar="F", help="fraction of the budget that triggers compaction")
    parser.add_argument("--keep-recent-turns", dest="keep_recent_turns", type=int, default=S,
                        metavar="N", help="turns preserved verbatim across a compaction")
    parser.add_argument("--host", dest="server_host", default=S, help="server host")
    parser.add_argument("--port", dest="server_port", type=int, default=S, help="server port")
    parser.add_argument("--server-cmd", dest="server_cmd", default=S,
                        help="command to launch the server (empty => auto-detect)")
    parser.add_argument("--model", dest="model", default=S,
                        help="model name/alias (all aliases route to "
                             f"{MODEL_ID})")
    parser.add_argument("--thinking", dest="thinking_default",
                        choices=THINKING_CHOICES, default=S,
                        help="default thinking display")
    parser.add_argument("--tool-call-mode", dest="tool_call_mode",
                        choices=["auto", "native", "prompted"], default=S)
    parser.add_argument("--session-dir", dest="session_dir", default=S)
    parser.add_argument("--export-dir", dest="export_dir", default=S)
    parser.add_argument("--log-level", dest="log_level",
                        choices=["debug", "info", "warn", "warning", "error"], default=S)
    return parser


def _fmt_cfg_value(value) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def print_config(cfg: dict, sources: dict, console) -> None:
    """Print the resolved config as a deterministic key | value | source table."""
    if _RICH:
        table = Table(title="mike config", show_header=True, header_style="bold")
        table.add_column("key", style="cyan")
        table.add_column("value")
        table.add_column("source", style="dim")
        for key in DEFAULTS:
            table.add_row(key, _fmt_cfg_value(cfg[key]), sources.get(key, "default"))
        console.print(table)
        console.print(f"[dim]user config: {USER_CONFIG} "
                      f"({'found' if USER_CONFIG.exists() else 'missing'})[/dim]")
        console.print(f"[dim]repo config: {REPO_CONFIG} "
                      f"({'found' if REPO_CONFIG.exists() else 'missing'})[/dim]")
    else:
        for key in DEFAULTS:
            console.print(f"{key} | {_fmt_cfg_value(cfg[key])} | {sources.get(key, 'default')}")


def _cli_overrides(args) -> dict:
    ns = vars(args)
    return {k: v for k, v in ns.items() if k in DEFAULTS and v is not None}


def cmd_export(args, cfg: dict, console) -> int:
    """Non-interactive ``mike.py export <session-name-or-id> [path]``."""
    rest = list(getattr(args, "rest", []) or [])
    if not rest:
        console.print("usage: mike.py export <session-name-or-id> [path]")
        return 2
    name = rest[0]
    path_arg = rest[1] if len(rest) > 1 else None
    store = SessionStore(resolved_session_dir(cfg))
    session = store.find(name)
    if session is None:
        console.print(f"[red]no session matching {name!r}[/red]")
        return 1
    written = do_export(session, cfg, path_arg, force=args.force,
                        console=console, interactive=False)
    return 0 if written else 1


def main(argv: Optional[list] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    cfg, sources = load_config(_cli_overrides(args))
    configure_logging(cfg["log_level"])
    console = _make_console()

    # --- pure/local subcommands that do not need the server ----------------------
    if args.command == "config":
        print_config(cfg, sources, console)
        return 0
    if args.command == "export":
        return cmd_export(args, cfg, console)

    state = load_state()
    counter = TokenCounter.create()
    if state.get("estimator") != counter.name:
        state["estimator"] = counter.name
        save_state(state)

    # --- start / attach the server (argparse already handled --help) -------------
    server = ServerManager(cfg["server_host"], cfg["server_port"], ROOT, console=console,
                           no_server=args.no_server, server_cmd=cfg["server_cmd"])
    try:
        server.ensure_running()
    except ServerStartError as exc:
        console.print(f"[red]could not start the DeepSeek-API server:[/red]\n{exc}")
        return 1

    llm = DeepSeekLLM(server.base_url, API_KEY, resolve_model(cfg["model_id"]))

    # --- tool mode: cached, else probe once --------------------------------------
    forced = cfg["tool_call_mode"]
    if forced in ("native", "prompted"):
        tool_mode = forced
    else:
        tool_mode = state.get("tool_mode")
    if tool_mode not in ("native", "prompted"):
        tool_mode = detect_tool_mode(llm)
    state["tool_mode"] = tool_mode
    state["port"] = cfg["server_port"]
    save_state(state)

    console.print(
        f"[dim]info: single model {MODEL_LABEL} ({MODEL_ID}); "
        f"{len(MODEL_ALIASES)} legacy aliases route to it; tools: {tool_mode}[/dim]"
    )

    store = SessionStore(resolved_session_dir(cfg))
    compactor = Compactor(
        counter,
        max_context=cfg["max_context"],
        threshold=cfg["compact_threshold"],
        keep_recent=cfg["keep_recent_turns"],
        summarizer=None,  # deterministic, guaranteed 7-section summaries
    )
    toolbox = Toolbox(os.getcwd(), console=console)

    want_resume = args.resume or args.command == "resume"
    session = None
    if want_resume:
        session = _prompt_resume(store, console)
    if session is None:
        session = store.new_session()

    agent = MikeAgent(
        llm=llm, session=session, store=store, compactor=compactor,
        toolbox=toolbox, console=console, tool_mode=tool_mode,
        thinking_mode=(THINKING_OFF if cfg["thinking_default"] in ("last", THINKING_OFF)
                       else cfg["thinking_default"]),
        server=server, cwd=os.getcwd(), cfg=cfg,
    )

    try:
        run_repl(agent, console, cfg)
    finally:
        agent._save()
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
