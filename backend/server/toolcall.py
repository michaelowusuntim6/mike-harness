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
        if "calls" in low:
            return "tool_call", raw.startswith("</")
        return "dsml_inner", raw.startswith("</")
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
    return {"name": str(name), "arguments": args}


# --------------------------------------------------------------------------- #
# DeepSeek DSML (V4 and V4.1)                                                  #
# --------------------------------------------------------------------------- #

_DSML_TOKEN = "\uff5cDSML\uff5c"  # full-width bar U+FF5C
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
_DSML_BLOCK_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>.*?"
    r"</\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_CALLS_OPEN_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*(?:tool_calls|calls)\s*>", re.IGNORECASE
)
_DSML_INVOKE_OPEN_BARE_RE = re.compile(
    r"<\s*\uff5cDSML\uff5c\s*invoke\b", re.IGNORECASE
)
_DSML_PARAM_ANY_RE = re.compile(
    r"</?\s*\uff5cDSML\uff5c\s*parameter\b[^<>]*>", re.IGNORECASE
)


def _attr_value(attrs: str, key: str) -> Optional[str]:
    for name, value in _DSML_ATTR_RE.findall(attrs or ""):
        if name.lower() == key.lower():
            return value
    return None


def _attrs_to_args(attrs: str) -> dict:
    args: dict = {}
    for name, value in _DSML_ATTR_RE.findall(attrs or ""):
        if name.lower() in ("name", "string"):
            continue
        args[name] = value
    return args


def _parse_param_tags(body: str) -> dict:
    args: dict = {}
    for pm in _DSML_PARAM_RE.finditer(body or ""):
        attrs = pm.group(1)
        args[_attr_value(attrs, "name") or "value"] = _dsml_value(pm.group(2), attrs)
    return args


def parse_dsml_calls(text: str) -> list[dict]:
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
            body = text[m.end():]
            calls.append({"name": name, "arguments": _parse_param_tags(body)})
            log.warning("tool_call_truncated name=%s", name)
    return calls


def _dsml_value(raw: str, attrs: str):
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
    if not text:
        return text
    out = _DSML_BLOCK_RE.sub("", text)
    out = re.sub(
        r"<\s*\uff5cDSML\uff5c\s*invoke\b[^<>]*>.*?"
        r"</\s*\uff5cDSML\uff5c\s*invoke\s*>",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(
        r"<\s*\uff5cDSML\uff5c\s*invoke\b[^<>]*/>",
        "",
        out,
        flags=re.IGNORECASE,
    )
    m = _DSML_CALLS_OPEN_RE.search(out)
    if m is not None:
        out = out[:m.start()]
    m = _DSML_INVOKE_OPEN_BARE_RE.search(out)
    if m is not None:
        out = out[:m.start()]
    out = _DSML_PARAM_ANY_RE.sub("", out)
    return out


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

    calls.extend(parse_dsml_calls(text))
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
