"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import LoginRequired, Session, get_session
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# The [auth] refresh log lines must be visible in the server console. uvicorn only
# attaches handlers to its own loggers (not the root), so we borrow uvicorn.error;
# this code path only runs inside the server process.
log = logging.getLogger("uvicorn.error")

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    reasoning: str = ""

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- auth refresh (401 handling) ---------------------------------------

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        """True only for DeepSeek's 401-class auth failures, not any RuntimeError.

        Matches the message the web API returns via `_biz`:
        "DeepSeek API error: Authorization Failed (invalid token)".
        """
        text = str(exc).lower()
        return "authorization failed" in text or "invalid token" in text

    def _apply_session(self, new: Session) -> None:
        """Swap in a refreshed session and update the live httpx client.

        httpx keeps the old `authorization` header on the client, so it must be
        overwritten explicitly; the cookies and user-agent change with the session.
        """
        self.session = new
        self._http.headers["authorization"] = f"Bearer {new.token}"
        self._http.headers["user-agent"] = new.user_agent
        self._http.cookies.clear()
        self._http.cookies.update(new.cookies)

    def _refresh_after_401(self) -> None:
        """Force exactly one headless token refresh from the browser profile.

        `max_age=0` bypasses the cached session.json and re-captures from the
        persistent profile. If the profile can no longer recover headlessly
        (LoginRequired), there is nothing we can do without a human — we log and
        re-raise so the caller propagates a clear error rather than retrying.
        """
        try:
            new = get_session(max_age=0, allow_interactive=False)
        except LoginRequired:
            log.warning("[auth] token refresh failed; session cannot be recovered headlessly")
            raise
        self._apply_session(new)
        log.info("[auth] token refreshed after 401, retrying")

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        try:
            r = self._http.post("/api/v0/chat_session/create", json={})
            r.raise_for_status()
            return _biz(r.json())["chat_session"]["id"]
        except Exception as exc:
            if not self._is_auth_error(exc):
                raise
        # Auth failure: refresh once and retry exactly once. A second failure
        # propagates (no loop).
        self._refresh_after_401()
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]
        with self._pow_lock:
            return self._pow.make_header(challenge)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        text = "".join(s)
        return Reply(text=text, conversation_id=s.conversation_id, reasoning=s.reasoning)

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of reply-text chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation."""

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: str,
                 thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None
        self._reasoning = ""

    def _request_headers(self) -> dict:
        # PoW challenges are short-lived, so solve right before the request.
        return {"x-ds-pow-response": self._client._pow_header()}

    def _body(self) -> dict:
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        return body

    def iter_events(self) -> Iterator[tuple]:
        """Yield ``("response"|"reasoning", text)`` tuples for the whole stream.

        If the completion POST (or its short-lived PoW header) returns a 401-class
        auth error before anything has been emitted, refresh the token once and
        reissue the request exactly once. We never retry after yielding, so a
        partial stream is never duplicated.
        """
        meta: dict = {}
        attempt = 0
        emitted = False
        while True:
            try:
                headers = self._request_headers()
                with self._client._http.stream(
                    "POST", COMPLETION_PATH, json=self._body(), headers=headers
                ) as resp:
                    resp.raise_for_status()
                    for kind, text in _iter_sse_events(resp.iter_lines(), meta):
                        emitted = True
                        if kind == "reasoning":
                            self._reasoning += text
                        yield kind, text
                break
            except Exception as exc:
                if (attempt == 0 and not emitted
                        and self._client._is_auth_error(exc)):
                    attempt = 1
                    self._client._refresh_after_401()
                    continue
                raise
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]

    def __iter__(self) -> Iterator[str]:
        for kind, text in self.iter_events():
            if kind == "response":
                yield text

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


def _is_reasoning_type(ftype) -> bool:
    """DeepSeek labels DeepThink fragments 'THINK' (older builds: 'THINKING')."""
    if not ftype:
        return False
    ftype = str(ftype).upper()
    return "THINK" in ftype or "REASON" in ftype


def _idx_of(path: str, active_idx: Optional[int], count: int) -> int:
    """Resolve a `.../fragments/<i>/...` path to a concrete fragment index."""
    m = re.search(r"fragments/(-?\d+)/", path)
    if m:
        i = int(m.group(1))
        if i >= 0:
            return i
    if active_idx is not None:
        return active_idx
    return max(0, count - 1)


def _iter_sse_events(lines, meta: Optional[dict] = None) -> Iterator[tuple]:
    """Turn DeepSeek's SSE completion stream into typed events.

    Yields ``("response", text)`` for the answer and ``("reasoning", text)`` for
    DeepThink reasoning fragments. DeepSeek streams a snapshot frame (the full
    response object with typed `fragments`), then path-based APPEND frames for
    each fragment's content, and finally appends any NEW fragments — including
    the answer — as a list under `response/fragments`.

    We keep per-fragment accumulated content and emit only the newly-arrived
    suffix, so reasoning and the answer are separated without duplication.

    If `meta` is given, the assistant's `message_id` is recorded into it.
    """
    frag_types: dict = {}
    frag_content: dict = {}
    emitted: dict = {}
    active_idx: Optional[int] = None

    def emit(idx: int) -> Iterator[tuple]:
        full = frag_content.get(idx, "")
        prev = emitted.get(idx, 0)
        if len(full) <= prev:
            return
        emitted[idx] = len(full)
        kind = "reasoning" if _is_reasoning_type(frag_types.get(idx)) else "response"
        yield (kind, full[prev:])

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        v = obj.get("v")

        # Snapshot frame: the full response object with typed fragments.
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)
            frag_types.clear()
            frag_content.clear()
            emitted.clear()
            frags = v["response"].get("fragments", [])
            for i, frag in enumerate(frags):
                frag_types[i] = frag.get("type")
                frag_content[i] = frag.get("content") or ""
            active_idx = len(frags) - 1 if frags else None
            for i in range(len(frags)):
                yield from emit(i)
            continue

        if "p" not in obj:
            # Bare append to the currently active fragment's content.
            if isinstance(v, str) and active_idx is not None:
                frag_content[active_idx] = frag_content.get(active_idx, "") + v
                yield from emit(active_idx)
            continue

        path = obj["p"]
        if meta is not None and path.endswith("message_id") and isinstance(v, int):
            meta["message_id"] = v

        if path.endswith("content") and isinstance(v, str):
            idx = _idx_of(path, active_idx, len(frag_types))
            if obj.get("o") == "APPEND":
                frag_content[idx] = frag_content.get(idx, "") + v
            else:
                frag_content[idx] = v
            active_idx = idx
            frag_types.setdefault(idx, "RESPONSE")
            yield from emit(idx)
        elif path.endswith("fragments") and isinstance(v, list):
            # A list of newly-appended fragments (the answer arrives this way).
            for frag in v:
                idx = len(frag_types)
                frag_types[idx] = frag.get("type")
                frag_content[idx] = frag.get("content") or ""
                emitted[idx] = 0
                active_idx = idx
                yield from emit(idx)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[str]:
    """Backwards-compatible view: yield only the response-text deltas.

    (Reasoning fragments are dropped here; use `_iter_sse_events` for typed
    events. `meta` is still populated with the message id.)
    """
    for kind, text in _iter_sse_events(lines, meta):
        if kind == "response":
            yield text


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame.

    DeepSeek nests the assistant message under `response`; we check there first,
    then the snapshot root, accepting `message_id` or `id`.
    """
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
