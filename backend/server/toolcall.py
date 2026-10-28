"""Prompted tool-call and stream parsing for the DeepSeek web backend.

The DeepSeek web chat has no native tool channel.  Codex speaks the Responses
API, so the bridge teaches the model a strict text envelope, then translates
it back into function/custom-tool calls.  The parsers here are adapted from the
proven implementation in ``mike.py`` (including the DeepSeek V4/V4.1 DSML
syntax, which the web model sometimes emits despite being prompted for JSON).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

log = logging.getLogger("uvicorn.error")

# --------------------------------------------------------------------------- #
# Thinking / tool-call stream parsing                                          #
# --------------------------------------------------------------------------- #

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
    if re.match(r"<\s*/?\s*tool_result\b", low):
        return "tool_result", raw.startswith("</")
    if "dsml" in low:
        closing = raw.startswith("</")
        if closing:
            # Only the block close leaves tool-call mode; inner closes are noise.
            return ("tool_call", True) if "calls" in low else ("dsml_inner", True)
        # Any opening DSML tag starts hidden tool markup, as in the reference
        # parser, which enters its tool-call stage on the bare DSML token.
        return "tool_call", False
    if "▁of▁" in raw:
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
    """Split *text* into (reasoning, clean_content)."""
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
        elif name in ("tool_call", "tool_result"):
            mode = "content" if closing else "hidden"
    tail = text[pos:]
    if tail:
        if mode == "reasoning":
            reasoning_parts.append(tail)
        elif mode != "hidden":
            out.append(tail)
    return "".join(reasoning_parts).strip(), "".join(out).strip()


def strip_thinking(text: str) -> str:
    return extract_thinking(text)[1]


_TOOL_RESULT_RE = re.compile(
    r"<\s*tool_result\b[^>]*>.*?<\s*/\s*tool_result\s*>"
    r"|<\s*/\s*tool_result\s*>"
    r"|<\s*tool_result\b[^>]*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_tool_results(text: str) -> str:
    if not text:
        return text
    return _TOOL_RESULT_RE.sub("", text)


def strip_channel_tags(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\[commentary\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[final\]\s*", "", text, flags=re.IGNORECASE)
    return text


class StreamParser:
    """Incremental filter that never leaks reasoning/tool syntax to the client."""

    MAX_TAG_LEN = 32

    def __init__(self) -> None:
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
            elif name in ("tool_call", "tool_result"):
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


# --------------------------------------------------------------------------- #
# Orphaned DSML fragments                                                      #
# --------------------------------------------------------------------------- #

# The model occasionally emits a stray closing tag (often with doubled bars)
# inside an argument value, e.g. {"cmd": "pwd</｜｜DSML｜｜>"}. Such fragments
# are markup, never data, so they are removed from argument values.
_DSML_ORPHAN_TAG_RE = re.compile(
    r"</?\s*(?:\uff5c)+DSML(?:\uff5c)+[^<>]*>", re.IGNORECASE
)
_DSML_BARE_TOKEN_RE = re.compile(r"(?:\uff5c)+DSML(?:\uff5c)+", re.IGNORECASE)


def _strip_orphan_dsml(text: str) -> str:
    cleaned = _DSML_ORPHAN_TAG_RE.sub("", text or "")
    return _DSML_BARE_TOKEN_RE.sub("", cleaned)


def _clean_arg_string(value: str) -> str:
    """Remove leaked DSML markup from one argument string.

    Values the model left untouched are returned verbatim; when a fragment was
    removed, surrounding whitespace from the markup is trimmed as well.
    """
    cleaned = _strip_orphan_dsml(value)
    if cleaned == value:
        return value
    return cleaned.strip()


def _clean_arguments(value):
    if isinstance(value, str):
        return _clean_arg_string(value)
    if isinstance(value, list):
        return [_clean_arguments(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_arguments(item) for key, item in value.items()}
    return value


# --------------------------------------------------------------------------- #
# <tool_call> JSON envelopes                                                   #
# --------------------------------------------------------------------------- #

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


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
            args = {"input": args}
    if not isinstance(args, dict):
        args = {"input": args} if isinstance(args, str) else {}
    return {"name": _clean_arg_string(str(name)), "arguments": _clean_arguments(args)}


# --------------------------------------------------------------------------- #
# DeepSeek DSML (V4 and V4.1), modeled on deepseek-recipe output_parser.rs      #
# --------------------------------------------------------------------------- #
#
# The reference parser is a byte-level state machine: outside tool calls it
# looks for the DSML token; inside the call block it consumes markup and
# assembles JSON arguments from ``invoke name="..."`` / ``parameter name="..."``
# tags, with ``string="true"`` values treated as JSON strings and
# ``string="false"`` values spliced in as raw JSON. Unmatched markup is
# skipped, so stray spaces, newlines, orphaned tags, and trailing garbage never
# reach the arguments. The scanner below applies the same rules to a complete
# response string (the bridge parses after the upstream turn ends).

# The token is ``｜DSML｜`` in the reference; the web model sometimes doubles or
# even triples the full-width bars, so every occurrence accepts one or more.
_DSML_TOKEN = r"(?:\uff5c)+DSML(?:\uff5c)+"
_DSML_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*" + _DSML_TOKEN +
    r"\s*(?P<word>invoke|parameter|tool_calls|calls)?"
    r"(?P<attrs>[^<>]*?)>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_ATTR_RE = re.compile(
    r"""([A-Za-z_][\w.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s/>]+))"""
)
_DSML_BLOCK_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*(?:tool_calls|calls)\s*>.*?"
    r"</\s*" + _DSML_TOKEN + r"\s*(?:tool_calls|calls)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE_BLOCK_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*invoke\b[^<>]*>.*?"
    r"</\s*" + _DSML_TOKEN + r"\s*invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM_BLOCK_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*parameter\b[^<>]*>.*?"
    r"</\s*" + _DSML_TOKEN + r"\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_BLOCK_OPEN_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*(?:tool_calls|calls)\s*>", re.IGNORECASE
)
_DSML_INVOKE_OPEN_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*invoke\b", re.IGNORECASE
)
_DSML_PARAM_OPEN_RE = re.compile(
    r"<\s*" + _DSML_TOKEN + r"\s*parameter\b", re.IGNORECASE
)


def _parse_attrs(attrs: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in _DSML_ATTR_RE.finditer(attrs or ""):
        value = next(
            (group for group in match.groups()[1:] if group is not None), ""
        )
        values[match.group(1).lower()] = value
    return values


def _dsml_value(raw: str, is_string: bool):
    value = _clean_arg_string(raw).strip()
    if is_string:
        return value
    if not value:
        return ""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        # Reference splices raw JSON; malformed values degrade to text instead
        # of breaking the whole call.
        return value


def parse_dsml_calls(text: str) -> list[dict]:
    """Extract tools calls from native DSML markup (V4/V4.1, any bar width)."""
    text = text or ""
    calls: list[dict] = []
    current: Optional[dict] = None
    in_value = False
    value_name: Optional[str] = None
    value_is_string = True
    value_start = 0

    def flush_value(end: int) -> None:
        nonlocal in_value
        if in_value and current is not None and value_name:
            current["arguments"][value_name] = _dsml_value(
                text[value_start:end], value_is_string
            )
        in_value = False

    def flush_call() -> None:
        nonlocal current
        if current is not None:
            calls.append(current)
            current = None

    for match in _DSML_TAG_RE.finditer(text):
        closing = bool(match.group("closing"))
        word = (match.group("word") or "").lower()
        attrs = match.group("attrs") or ""
        self_closing = attrs.rstrip().endswith("/")
        if self_closing:
            attrs = attrs.rstrip()[:-1]

        # A value ends at the next DSML tag of any kind. This is what turns
        # "pwd</｜｜DSML｜｜>" into "pwd" even when the close tag is malformed.
        if in_value:
            flush_value(match.start())

        if word in ("tool_calls", "calls"):
            if closing:
                flush_call()
            continue
        if word == "invoke":
            if closing:
                flush_call()
                continue
            info = _parse_attrs(attrs)
            name = info.get("name")
            if not name:
                continue
            flush_call()
            current = {"name": name, "arguments": {}}
            if self_closing:
                flush_call()
            continue
        if word == "parameter":
            if closing or current is None:
                continue
            info = _parse_attrs(attrs)
            name = info.get("name")
            if not name:
                continue
            value_name = name
            # Missing string="true" is treated as a string, the common intent.
            value_is_string = info.get("string", "true").lower() != "false"
            value_start = match.end()
            in_value = True
            continue
        # Bare tag (e.g. "</｜｜DSML｜｜>"): markup with no meaning; the value
        # that ended at it has already been flushed, and the tag itself is
        # consumed by the scanner.

    if in_value:
        flush_value(len(text))
    flush_call()
    return calls


def strip_dsml(text: str) -> str:
    """Remove every DSML block or stray tag from visible content."""
    if not text:
        return text
    out = _DSML_BLOCK_RE.sub("", text)
    out = _DSML_INVOKE_BLOCK_RE.sub("", out)
    out = _DSML_PARAM_BLOCK_RE.sub("", out)
    # A block left open by truncation swallows the remainder as markup.
    for opener in (_DSML_BLOCK_OPEN_RE, _DSML_INVOKE_OPEN_RE, _DSML_PARAM_OPEN_RE):
        found = opener.search(out)
        if found is not None:
            out = out[: found.start()]
    out = _DSML_TAG_RE.sub("", out)
    return _DSML_BARE_TOKEN_RE.sub("", out)


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """Return (calls, clean_text) for both JSON envelopes and DSML calls."""
    if not text:
        return [], ""

    calls: list[dict] = []
    for raw in _TOOL_CALL_BLOCK_RE.findall(text):
        parsed = _loads_tool_obj(raw)
        if parsed is not None:
            calls.append(parsed)

    clean = _TOOL_CALL_BLOCK_RE.sub("", text)

    if not calls:
        for raw in _FENCED_JSON_RE.findall(clean):
            parsed = _loads_tool_obj(raw)
            if parsed is not None and parsed.get("name"):
                calls.append(parsed)
                clean = clean.replace(raw, "")
                clean = re.sub(r"```(?:json)?\s*```", "", clean)

    # DSML outside a JSON envelope is a native tool call. Parsing only the
    # remainder keeps markup leaked inside an envelope from becoming a second
    # call; those fragments are cleaned from the arguments instead.
    calls.extend(parse_dsml_calls(clean))
    clean = strip_dsml(clean)
    return calls, clean.strip()


def looks_like_tool_syntax(text: str) -> bool:
    low = (text or "").lower()
    return (
        "dsml" in low
        or "<tool_call>" in low
        or re.search(r"invoke\s+name\s*=", low) is not None
        or "```json" in low
    )
