"""Who owns which upstream DeepSeek chat session.

A `chat_session_id` belongs to exactly one account: replay a conversation id
that someone else created and the web API rejects the turn. Mike's core keeps
only the conversation id the bridge handed back on the previous turn, so the
bridge is the one place that can notice the signed-in account changed, move the
thread onto a session the current account owns, and give the new id back.

This module is that memory:

* `owners`  — chat_session_id -> account fingerprint (who created it).
* `threads` — mike thread -> {account fingerprint: chat_session_id}. One logical
  thread therefore has at most one DeepSeek chat per account, and signing back
  in as an earlier account finds the session it already used instead of opening
  yet another chat. The thread key is the core's `thread_id` when the request
  carries one: the core holds the conversation id in memory only, so a resumed
  mike process forgets it while the ledger still connects the new session to the
  old account's. When a client sends no thread id, the first session id stands in
  for it.

State lives in `session/accounts.json` (git-ignored, next to the captured
session). Fingerprints are one-way hashes of the account id: the file never
holds a token, an email, or a phone number.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from deepseek.client import _decode_cid

# Same directory the captured session lives in; see deepseek.auth.
SESSION_DIR = Path(__file__).resolve().parent.parent / "session"
DEFAULT_LEDGER_PATH = SESSION_DIR / "accounts.json"

# Bump only if the on-disk shape changes incompatibly.
LEDGER_VERSION = 1

log = logging.getLogger("uvicorn.error")


def current_account(client) -> Optional[str]:
    """Fingerprint of the account `client` is signed in as, or None.

    None means "cannot tell", and every caller then behaves exactly as it did
    before multi-account support: a client without an identity lookup (the test
    fakes) or a failing `users/current` call must never fail a turn.
    """
    lookup = getattr(client, "account_id", None)
    if not callable(lookup):
        return None
    try:
        return lookup()
    except Exception:  # pragma: no cover - defensive; account_id() swallows
        log.exception("accounts: identity lookup raised")
        return None


class AccountLedger:
    """Persisted map from upstream chat session to the account that owns it."""

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._owners: dict[str, str] = {}
        self._threads: dict[str, dict[str, str]] = {}
        self._alias: dict[str, str] = {}
        self._loaded = False
        self._dirty = False

    # --- persistence ----------------------------------------------------------

    def _load(self) -> None:
        """Read the ledger once. A missing or damaged file starts us clean."""
        self._loaded = True
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.warning("accounts: ignoring unreadable %s (%s)", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        owners = data.get("owners")
        threads = data.get("threads")
        alias = data.get("alias")
        if isinstance(owners, dict):
            self._owners = {str(k): str(v) for k, v in owners.items() if v}
        if isinstance(threads, dict):
            self._threads = {
                str(key): {str(o): str(s) for o, s in value.items() if s}
                for key, value in threads.items()
                if isinstance(value, dict)
            }
        if isinstance(alias, dict):
            self._alias = {str(k): str(v) for k, v in alias.items()}

    def _save(self) -> None:
        """Write atomically: a crash mid-turn must not leave a truncated ledger."""
        payload = {
            "version": LEDGER_VERSION,
            "owners": self._owners,
            "threads": self._threads,
            "alias": self._alias,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._path)
            self._dirty = False
        except OSError as exc:
            # Losing the ledger costs us re-homing memory, not the turn itself.
            log.warning("accounts: could not persist %s (%s)", self._path, exc)

    # --- bookkeeping ----------------------------------------------------------

    def _group_for(self, thread_id: Optional[str], session_id: Optional[str]) -> str:
        """The key one logical mike thread is remembered under."""
        if thread_id:
            return f"mike:{thread_id}"
        if session_id:
            return self._alias.get(session_id) or session_id
        return ""

    def _record(self, session_id: str, account: str, group: str) -> bool:
        """Attach a session to its owner and thread; True if anything changed."""
        changed = self._owners.get(session_id) != account
        self._owners[session_id] = account
        if group:
            if self._alias.get(session_id) != group:
                self._alias[session_id] = group
                changed = True
            aliases = self._threads.setdefault(group, {})
            if aliases.get(account) != session_id:
                aliases[account] = session_id
                changed = True
        if changed:
            self._dirty = True
        return changed

    def owner_of(self, session_id: str) -> Optional[str]:
        with self._lock:
            if not self._loaded:
                self._load()
            return self._owners.get(session_id)

    def session_for(self, thread_id: str, account: str) -> Optional[str]:
        """The session this account already uses for `thread_id`, if any."""
        with self._lock:
            if not self._loaded:
                self._load()
            return self._threads.get(f"mike:{thread_id}", {}).get(account)

    def resolve(
        self,
        conversation_id: Optional[str],
        account: Optional[str],
        thread_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Translate an incoming conversation id for the currently signed-in account.

        Returns ``(conversation_id_to_use, thread_key)``, where the thread key is
        what the caller passes back to `stamp` for the session this turn ends up
        on. The chosen id is either:

          * the incoming one, when this account owns that session (or we have
            never seen it, in which case we claim it for this account);
          * a bare session id — no message part — when this account has another
            session for the same thread. Bare is deliberate: the previous
            account's branch cannot be continued, and re-rooting makes the
            bridge replay the whole transcript instead of a stale tail;
          * None, meaning open a new thread in this account and link it to the
            same mike thread.

        A request with no conversation id always starts a new upstream thread,
        even when this account already has one for the thread: clearing the id is
        how the core says it reset its history (`mike resume` and history resets
        both do), and guessing from the transcript would undo that.
        """
        if not account:
            return conversation_id, None

        session_id, _parent = _decode_cid(conversation_id) if conversation_id else (None, None)
        with self._lock:
            if not self._loaded:
                self._load()
            group = self._group_for(thread_id, session_id)
            if session_id is None:
                return None, group or None

            recorded = self._owners.get(session_id)
            if recorded is None or recorded == account:
                if recorded is None:
                    log.info("accounts: session %s belongs to %s", session_id[:8], account[:8])
                self._record(session_id, account, group)
                if self._dirty:
                    self._save()
                return conversation_id, group or None

            mine = self._threads.get(group, {}).get(account) if group else None
            if mine:
                if mine != session_id:
                    log.info(
                        "accounts: %s signed in, thread %s resumes in its own session %s "
                        "(not %s)",
                        account[:8], group[:24], mine[:8], session_id[:8],
                    )
                return mine, group or None

            log.warning(
                "accounts: signed in as %s, but thread %s was created by %s; "
                "opening a session for this account",
                account[:8], group[:24], recorded[:8],
            )
            return None, group or None

    def stamp(
        self,
        conversation_id: Optional[str],
        account: Optional[str],
        thread_key: Optional[str] = None,
    ) -> None:
        """Record the owner of the session behind `conversation_id`.

        `thread_key` is the group `resolve` answered for this turn: passing it
        back is what turns a freshly opened session into the account's alias of
        an existing mike thread.
        """
        if not conversation_id or not account:
            return
        session_id, _parent = _decode_cid(conversation_id)
        if session_id is None:
            return

        with self._lock:
            if not self._loaded:
                self._load()
            group = thread_key or self._group_for(None, session_id)
            if self._record(session_id, account, group) and self._dirty:
                self._save()


# The bridge is a single process, so one shared ledger is enough; it loads
# lazily on first use and serialises writers with a lock.
ledger = AccountLedger()
