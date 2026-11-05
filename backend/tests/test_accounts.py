"""Multi-account re-homing: the bridge moves a thread to the signed-in account.

A DeepSeek chat session belongs to the account that created it, so a mike thread
resumed after `python -m deepseek.auth` re-captured a different sign-in has to be
re-homed rather than replayed. These cover the ledger that remembers who owns
what, and the two endpoints' behaviour across a switch and back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepseek.client import _account_fingerprint
from server import responses
from server.accounts import AccountLedger, current_account
from server.responses import _response_events

SID_A = "aaaaaaaa-0000-4000-8000-000000000001"
SID_B = "bbbbbbbb-0000-4000-8000-000000000002"
ACCT_A = "account-a-fingerprint"
ACCT_B = "account-b-fingerprint"


class FakeStream:
    def __init__(self, events, conversation_id=None):
        self.events = events
        self.conversation_id = conversation_id

    def iter_events(self):
        yield from self.events


class AccountedClient:
    """A DeepSeekClient stand-in whose signed-in account can change per turn."""

    def __init__(self, account=None):
        self.account = account
        self.scripts: list[dict] = []
        self.prompts: list[str] = []
        self.conversation_ids: list[str | None] = []
        self.models: list[str | None] = []

    def account_id(self):
        return self.account

    def queue(self, text="Done.", conversation_id=None):
        self.scripts.append({"events": [("response", text)],
                             "conversation_id": conversation_id})

    def stream(self, prompt, model=None, thinking=False, conversation_id=None):
        self.prompts.append(prompt)
        self.models.append(model)
        self.conversation_ids.append(conversation_id)
        script = self.scripts.pop(0)
        return FakeStream(script["events"], script.get("conversation_id"))


def history(first="GLIMMER-42 is the codeword. Reply ACK.", second="What was it?"):
    """A two-turn transcript: the shape mike sends when resuming a thread."""
    return [
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": "You are mike."}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": first}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "ACK."}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": second}]},
    ]


def body_with(input_items=None, conversation_id=None, thread_id=None):
    body = {
        "model": "deepseek-v4.1-flash",
        "input": input_items if input_items is not None else history(),
        "tools": [],
    }
    if conversation_id:
        body["conversation_id"] = conversation_id
    if thread_id:
        body["client_metadata"] = {"thread_id": thread_id}
    return body


def run(client, body):
    events = []
    for raw in _response_events(client, body):
        for block in raw.split("\n\n"):
            if block.startswith("data: "):
                events.append(json.loads(block[6:]))
    return events


def completed(events):
    return [e for e in events if e["type"] == "response.completed"][0]["response"]


@pytest.fixture
def ledger(tmp_path):
    led = AccountLedger(path=tmp_path / "accounts.json")
    responses.ledger = led
    yield led
    responses.ledger = AccountLedger()  # restore the process singleton


# --- account fingerprint ------------------------------------------------------

def test_fingerprint_prefers_the_user_id_and_ignores_the_token():
    assert _account_fingerprint({"id": 9001, "email": "a@b.c"}) == \
        _account_fingerprint({"id": 9001, "email": "changed@b.c"})


def test_fingerprint_falls_back_to_a_contact_handle():
    assert _account_fingerprint({"email": "a@b.c"}) is not None
    assert _account_fingerprint({"phone_number": "+1555"}) is not None
    assert _account_fingerprint({}) is None
    assert _account_fingerprint({"id": "   "}) is None
    assert _account_fingerprint(None) is None


def test_fingerprint_is_not_the_raw_account_id():
    digest = _account_fingerprint({"id": 9001})
    assert "9001" not in digest


def test_client_without_an_identity_lookup_reports_no_account():
    class NoIdentity:
        pass

    assert current_account(NoIdentity()) is None


def _identity_client(biz):
    """A DeepSeekClient whose only network call is `users/current`."""
    import threading
    from deepseek.client import DeepSeekClient

    calls = {"n": 0}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            if isinstance(biz, Exception):
                raise biz
            return {"code": 0, "data": {"biz_data": biz}}

    class Jar:
        def clear(self):
            pass

        def update(self, cookies):
            pass

    class HTTP:
        def __init__(self):
            self.headers = {}
            self.cookies = Jar()

        def get(self, path):
            return Response()

    client = DeepSeekClient.__new__(DeepSeekClient)
    client._http = HTTP()
    client.session = None
    client._account = None
    client._account_failed_at = 0.0
    client._account_lock = threading.RLock()
    client.calls = calls
    return client


def test_account_id_caches_one_lookup_per_session():
    client = _identity_client({"id": 9001})
    first = client.account_id()
    assert first is not None
    assert client.account_id() == first
    assert client.calls["n"] == 1


def test_a_payload_without_an_identity_field_reports_no_account():
    assert _identity_client({"avatar_url": "x"}).account_id() is None
    assert _identity_client({}).account_id() is None


def test_a_broken_identity_lookup_never_fails_a_turn():
    client = _identity_client(RuntimeError("DeepSeek API error: boom"))
    assert client.account_id() is None          # swallowed, not raised
    assert current_account(client) is None


def test_a_refreshed_token_rechecks_who_is_signed_in():
    import time as _time
    from deepseek.client import Session

    client = _identity_client({"id": 9001})
    assert client.account_id() is not None
    assert client.calls["n"] == 1

    # Re-capturing the session may mean a different human signed in.
    client._apply_session(Session(token="t", cookies={}, user_agent="UA",
                                  captured_at=_time.time()))
    client.account_id()
    assert client.calls["n"] == 2


# --- ledger -------------------------------------------------------------------

def test_an_unseen_session_is_claimed_for_the_signed_in_account(ledger):
    assert ledger.resolve(f"{SID_A}:5", ACCT_A) == (f"{SID_A}:5", SID_A)
    assert ledger.owner_of(SID_A) == ACCT_A


def test_the_same_account_resumes_its_own_session_unchanged(ledger):
    ledger.resolve(f"{SID_A}:5", ACCT_A)
    assert ledger.resolve(f"{SID_A}:6", ACCT_A) == (f"{SID_A}:6", SID_A)


def test_no_conversation_id_or_no_identity_changes_nothing(ledger):
    assert ledger.resolve(None, ACCT_A) == (None, None)
    assert ledger.resolve(f"{SID_A}:5", None) == (f"{SID_A}:5", None)


def test_a_new_account_gets_a_fresh_thread_and_keeps_the_link(ledger):
    ledger.resolve(f"{SID_A}:5", ACCT_A)
    cid, thread_key = ledger.resolve(f"{SID_A}:6", ACCT_B)
    assert cid is None                      # nothing of B's to resume yet
    assert thread_key == SID_A              # ... but remember whose thread it was
    ledger.stamp(f"{SID_B}:2", ACCT_B, thread_key)
    assert ledger.owner_of(SID_B) == ACCT_B


def test_switching_back_reuses_the_original_session_id(ledger):
    ledger.resolve(f"{SID_A}:5", ACCT_A)
    _cid, thread_key = ledger.resolve(f"{SID_A}:6", ACCT_B)
    ledger.stamp(f"{SID_B}:2", ACCT_B, thread_key)

    # Back to A: the message part is B's, the session part goes back to A's.
    cid, _key = ledger.resolve(f"{SID_B}:3", ACCT_A)
    assert cid == SID_A
    assert ledger.resolve(f"{SID_B}:4", ACCT_A)[0] == SID_A

    # And B keeps its own session, so a third switch is not a third new chat.
    assert ledger.resolve(f"{SID_A}:7", ACCT_B)[0] == SID_B


def test_a_re_homed_session_is_bare_so_the_whole_transcript_is_replayed(ledger):
    from deepseek.client import _decode_cid

    ledger.resolve(f"{SID_A}:5", ACCT_A)
    _cid, key = ledger.resolve(f"{SID_A}:6", ACCT_B)
    ledger.stamp(f"{SID_B}:2", ACCT_B, key)
    target, _ = ledger.resolve(f"{SID_B}:3", ACCT_A)
    assert _decode_cid(target) == (SID_A, None)


def test_the_thread_id_groups_the_sessions_of_one_mike_thread(ledger):
    thread = "11111111-2222-4333-8444-555555555555"
    cid, key = ledger.resolve(None, ACCT_A, thread)
    assert (cid, key) == (None, f"mike:{thread}")   # a new thread is always new
    ledger.stamp(f"{SID_A}:5", ACCT_A, key)
    assert ledger.session_for(thread, ACCT_A) == SID_A

    # B resumes A's id: its own session, opened fresh, linked to the same thread.
    cid, key = ledger.resolve(f"{SID_A}:6", ACCT_B, thread)
    assert (cid, key) == (None, f"mike:{thread}")
    ledger.stamp(f"{SID_B}:2", ACCT_B, key)

    # Back to A. The core has moved on to B's session; A's original one comes back.
    assert ledger.resolve(f"{SID_B}:3", ACCT_A, thread)[0] == SID_A
    # A request with no conversation id never resumes: that is how the core says
    # it reset its history (`mike resume`, /new), account switch or not.
    assert ledger.resolve(None, ACCT_A, thread) == (None, f"mike:{thread}")


def test_the_thread_id_outlives_a_restarted_bridge(tmp_path):
    thread = "11111111-2222-4333-8444-555555555555"
    first = AccountLedger(path=tmp_path / "accounts.json")
    _cid, key = first.resolve(None, ACCT_A, thread)
    first.stamp(f"{SID_A}:5", ACCT_A, key)

    second = AccountLedger(path=tmp_path / "accounts.json")
    assert second.session_for(thread, ACCT_A) == SID_A
    # A restarted core resumes B's id and still finds A's original session.
    second.stamp(f"{SID_B}:2", ACCT_B, key)
    assert second.resolve(f"{SID_B}:3", ACCT_A, thread)[0] == SID_A


def test_ledger_survives_a_restart(tmp_path):
    path = tmp_path / "accounts.json"
    first = AccountLedger(path=path)
    first.resolve(f"{SID_A}:5", ACCT_A)
    _cid, key = first.resolve(f"{SID_A}:6", ACCT_B)
    first.stamp(f"{SID_B}:2", ACCT_B, key)

    second = AccountLedger(path=path)
    assert second.resolve(f"{SID_B}:3", ACCT_A)[0] == SID_A
    assert second.owner_of(SID_A) == ACCT_A


def test_a_damaged_ledger_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "accounts.json"
    path.write_text("{ this is not json", encoding="utf-8")
    led = AccountLedger(path=path)
    assert led.resolve(f"{SID_A}:5", ACCT_A)[0] == f"{SID_A}:5"


def test_stamping_an_unchanged_owner_does_not_rewrite_the_file(ledger, tmp_path):
    ledger.resolve(f"{SID_A}:5", ACCT_A)
    written = ledger._path.stat().st_mtime_ns
    ledger.stamp(f"{SID_A}:6", ACCT_A)
    assert ledger._path.stat().st_mtime_ns == written


# --- /v1/responses ------------------------------------------------------------

def test_same_account_resumes_with_only_the_new_turn(ledger):
    client = AccountedClient(ACCT_A)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2]))
    assert client.conversation_ids == [None]

    client.queue("It was GLIMMER-42.", f"{SID_A}:6")
    events = run(client, body_with(conversation_id=f"{SID_A}:5"))
    assert client.conversation_ids == [None, f"{SID_A}:5"]
    # Resuming sends the tail, not the transcript.
    assert "GLIMMER-42 is the codeword" not in client.prompts[-1]
    assert "What was it?" in client.prompts[-1]
    assert completed(events)["conversation_id"] == f"{SID_A}:6"


def test_resuming_after_a_different_account_signed_in_re_homes_the_thread(ledger):
    client = AccountedClient(ACCT_A)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2]))

    # Someone else is now signed in; B has never seen this thread.
    client.account = ACCT_B
    client.queue("It was GLIMMER-42.", f"{SID_B}:9")
    events = run(client, body_with(conversation_id=f"{SID_A}:5"))

    assert client.conversation_ids == [None, None]   # a fresh thread for B
    assert client.models[-1] == "default"            # ... which selects a model
    # B's empty thread gets the whole transcript, so the answer still works and
    # the token usage the bridge reports covers everything the model sees.
    assert "GLIMMER-42 is the codeword" in client.prompts[-1]
    assert "What was it?" in client.prompts[-1]
    assert completed(events)["conversation_id"] == f"{SID_B}:9"

    # The original account's session is untouched and still owned by A.
    assert ledger.owner_of(SID_A) == ACCT_A
    assert ledger.owner_of(SID_B) == ACCT_B


def test_switching_back_to_the_original_account_reuses_its_session(ledger):
    client = AccountedClient(ACCT_A)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2]))

    client.account = ACCT_B
    client.queue("For B.", f"{SID_B}:9")
    run(client, body_with(conversation_id=f"{SID_A}:5"))

    client.account = ACCT_A
    client.queue("Back to A.", f"{SID_A}:12")
    events = run(client, body_with(conversation_id=f"{SID_B}:9"))

    # A's own chat session, re-rooted: no message part, so no stale branch.
    assert client.conversation_ids[-1] == SID_A
    assert "GLIMMER-42 is the codeword" in client.prompts[-1]
    assert completed(events)["conversation_id"] == f"{SID_A}:12"
    # A third switch finds B's session waiting, not a new chat each time.
    client.account = ACCT_B
    client.queue("And B again.", f"{SID_B}:15")
    run(client, body_with(conversation_id=f"{SID_A}:12"))
    assert client.conversation_ids[-1] == SID_B


def test_a_resumed_mike_thread_switches_accounts_and_back(ledger):
    """A restarted core resumes with no conversation id, then an account switch.

    The thread id is the only thing that still says which mike thread this is,
    and it is what lets the original account find the session it created.
    """
    thread = "11111111-2222-4333-8444-555555555555"
    client = AccountedClient(ACCT_A)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2], thread_id=thread))
    assert client.conversation_ids == [None]          # a fresh chat for a new thread

    # Same thread, second run: no id, so still a fresh chat — a reset is a reset.
    client.queue("Hello again.", f"{SID_A}:6")
    run(client, body_with(input_items=history(), thread_id=thread))
    assert client.conversation_ids[-1] is None

    # Now somebody else is signed in and replays the id this account made.
    client.account = ACCT_B
    client.queue("B never saw this.", f"{SID_B}:7")
    run(client, body_with(conversation_id=f"{SID_A}:6", thread_id=thread))
    assert client.conversation_ids[-1] is None
    assert ledger.session_for(thread, ACCT_B) == SID_B

    # And back to A: its own session, re-rooted rather than continued.
    client.account = ACCT_A
    client.queue("Back on A.", f"{SID_A}:9")
    run(client, body_with(conversation_id=f"{SID_B}:7", thread_id=thread))
    assert client.conversation_ids[-1] == SID_A
    assert ledger.session_for(thread, ACCT_A) == SID_A


def test_re_homing_does_not_distort_the_reported_context_usage(ledger):
    client = AccountedClient(ACCT_A)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2]))

    # Resumed in its own account: only the tail goes over the wire.
    client.queue("It was GLIMMER-42.", f"{SID_A}:6")
    resumed = completed(run(client, body_with(conversation_id=f"{SID_A}:5")))
    assert "GLIMMER-42 is the codeword" not in client.prompts[-1]

    # Re-homed onto a fresh account: the whole transcript goes over the wire.
    client.account = ACCT_B
    client.queue("It was GLIMMER-42.", f"{SID_B}:9")
    re_homed = completed(run(client, body_with(conversation_id=f"{SID_A}:5")))
    assert "GLIMMER-42 is the codeword" in client.prompts[-1]

    # Usage is billed against mike's transcript either way, so the footer's
    # `Ctx: n / 128,000` does not jump because somebody else signed in.
    assert re_homed["usage"]["input_tokens"] == resumed["usage"]["input_tokens"]
    for usage in (resumed["usage"], re_homed["usage"]):
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


def test_without_an_identity_the_bridge_behaves_as_it_always_did(ledger):
    client = AccountedClient(None)
    client.queue("GLIMMER-42.", f"{SID_A}:5")
    run(client, body_with(input_items=history()[:2]))

    client.queue("It was GLIMMER-42.", f"{SID_A}:6")
    events = run(client, body_with(conversation_id=f"{SID_A}:5"))
    assert client.conversation_ids == [None, f"{SID_A}:5"]
    assert "GLIMMER-42 is the codeword" not in client.prompts[-1]
    assert ledger.owner_of(SID_A) is None


def test_a_bare_rewound_id_re_homes_like_any_other_id(ledger):
    """After compaction the core sends a session with no message part; an
    account switch must still move that, not replay into the old account."""
    client = AccountedClient(ACCT_A)
    client.queue("compact ack", SID_A)
    run(client, body_with(input_items=history()[:2]))

    client.account = ACCT_B
    client.queue("re-homed summary", f"{SID_B}:3")
    run(client, body_with(conversation_id=SID_A))
    assert client.conversation_ids[-1] is None
    assert client.models[-1] == "default"

    client.account = ACCT_A
    client.queue("back on A", f"{SID_A}:8")
    run(client, body_with(conversation_id=f"{SID_B}:3"))
    assert client.conversation_ids[-1] == SID_A
    assert client.models[-1] is None
