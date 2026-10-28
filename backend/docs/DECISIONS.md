# DECISIONS — building `mike.py` on Deepseek-API

Canonical location: `docs/DECISIONS.md`.

Every non-obvious choice made while executing the brief, in the form
**Decision — Rationale — Alternatives considered**. Also records the test
summary and the end-to-end transcripts required by the brief.

---

## A. Architecture

### A1. Single file, no sibling imports
**Decision.** `mike.py` is one file; the agent loop, tool registry, compactor,
session manager, thinking parser, tool-call parser and server supervisor all live
inside it. It imports only the stdlib and packages already in the venv
(`openai`/`httpx`, `rich`, `tiktoken`, `python-dotenv`).
**Rationale.** The brief requires zero runtime dependency on `~/nanocode` or
`~/cipher-harness`; verified by renaming both directories and running mike.
**Alternatives.** A `mike/` package — rejected: the deliverable is explicitly a
single file.

### A2. Provider: OpenAI-compatible only
**Decision.** The only client is `DeepSeekLLM`, pointed at
`http://127.0.0.1:<port>/v1` with `api_key="sk-local"`. The `openai` SDK is used
when present, with a hand-rolled `httpx` SSE fallback otherwise.
**Rationale.** The brief forbids OpenRouter/Anthropic paths. The server ignores
keys, but we still send one for SDK compatibility.
**Alternatives.** Pure httpx always — rejected: the SDK is already installed and
handles retries/backoff.

### A3. Keep the full history on disk; compact only the in-context view
**Decision.** `session.messages` is append-only and holds the complete
transcript; `MikeAgent.context` is what is sent upstream, and compaction shrinks
only `context`.
**Rationale.** The brief requires both *"the summary REPLACES the older turns
in-context"* and *"the full untruncated history stays on disk"* / *"never lose
user data"*. Keeping one list that does both is impossible.
**Alternatives.** Store the pre-compaction history inside the compaction record —
rejected: bloats the file and duplicates the transcript.

---

## B. Server integration

### B1. Auto-launch is attach-first and idempotent
**Decision.** TCP-probe the port first; if it answers, reuse it and **never**
claim ownership. Only spawn `sys.executable app.py` (cwd = repo root) when the
port is free, capture its combined output into a 200-line ring buffer, and poll
`/v1/models` for up to 30 s at 250 ms backoff. On failure print the last 50 log
lines and raise `ServerStartError` (exit 1).
**Rationale.** Running mike twice must not spawn two servers; and we must not
kill a server we merely attached to.
**Alternatives.** Kill-by-port / pidfile — rejected: can kill an unrelated
process the user started deliberately.

### B2. Tool-calling mode is probed once and cached
**Decision.** At startup (only if not cached) send a probe asking the model to
call a dummy `ping` tool; if structured `tool_calls` come back use **native**,
otherwise **prompted**. Cache the result in `.mike_state.json`.
**Rationale.** The web-backed bridge drops `tools=` silently, so a one-time probe
is the only reliable signal; caching avoids paying for it every launch.
**Alternatives.** Always prompted — rejected as less future-proof. Probe on every
launch — rejected as slower.

### B3. Support DeepSeek's native DSML syntax *and* the required envelope
**Decision.** In prompted mode, instruct the model to use the exact
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` envelope, **and** also
parse DeepSeek's native DSML `invoke`/`parameter` blocks that the real model
actually emits. DSML/`<tool_call>` blocks are hidden from scrollback.
**Rationale.** Empirically the free web model ignores the requested envelope and
emits its own DSML syntax; without DSML support no tool ever fires. The brief
mandates the envelope, so both are supported.
**Alternatives.** Envelope only — rejected: tools never run in practice.
DSML only — rejected: violates the brief.

### B4. Server-side single model + alias rewrite shim
**Decision.** `server/config.py` defines one canonical id
(`deepseek-v4.1-flash`, display name *DeepSeek V4.1 Flash*) with a
compatibility alias list. `/v1/models` returns exactly one entry with an
`aliases: [...]` extension. Any accepted alias is rewritten to the canonical id
and logged at INFO exactly once per alias per process. Unknown ids still 404.
**Rationale.** The brief requires one surfaced model plus transparent aliasing
without log spam.
**Alternatives.** Return the aliases as separate model entries — rejected: the
user must see exactly one model. Silent no-404 fallback for unknown ids —
rejected: hides typos.

### B5. Reasoning is surfaced as `reasoning_content`
**Decision.** The upstream stream is parsed into typed events by fragment type
(`THINK*`/`REASON*` → reasoning, everything else → answer) and the bridge emits
`delta.reasoning_content`, matching DeepSeek's own non-standard field. Reply
objects gained a `reasoning` attribute.
**Rationale.** The brief wants thinking on/off to work end-to-end; the original
`_parse_sse` discarded non-RESPONSE fragments and the answer fragment arrives as
an appended list, which had to be handled to avoid dropping the answer.
**Alternatives.** Leave reasoning inlined in `content` — rejected: it pollutes
answers and can't be toggled.

---

## C. Harness behaviour

### C1. Compaction summary is deterministic (no LLM call)
**Decision.** `Compactor.build_summary` always emits the seven required sections
(`Goal`, `Constraints`, `Files Touched`, `Decisions Made`, `Open Todos`,
`Last User Intent`, `Next Steps`) by extracting them locally; the optional
summarizer hook is left `None` by default.
**Rationale.** Guarantees `/compact` and rollover always produce a valid,
seven-section summary even offline or on a bad day, and keeps the behaviour
deterministic for tests.
**Alternatives.** Always call the model to summarise (as `compact.py` did) —
rejected: slow, costs a request, and can omit sections. A model call is
available via the `summarizer` hook if wanted.

### C2. Token counter: tiktoken with persisted choice
**Decision.** Use `tiktoken` (`cl100k_base`) if importable, else `len(text)//4`.
Persist the chosen estimator name in `.mike_state.json`.
**Rationale.** Brief requires it and asks the choice be persisted.
**Alternatives.** Always heuristic — rejected: less accurate where tiktoken is
available.

### C3. Thresholds
**Decision.** `MAX_CONTEXT_TOKENS = 500_000` (overridable via `MIKE_MAX_CONTEXT`),
`COMPACT_THRESHOLD = 0.80` → trigger at `400_000`, `KEEP_RECENT_TURNS = 6`.
**Rationale.** Exactly what the brief specifies; `/tokens` reports the derived
400,000 threshold.
**Alternatives.** None sensible.

### C4. Session files are atomic and human-readable
**Decision.** Write JSON pretty-printed with sorted keys via a temp file +
`fsync` + `os.replace`. Sessions live in `sessions/` as
`YYYYMMDD-HHMMSS__<slug>.json`; rollover creates `<slug>-contN` seeded with a
single message beginning *"This is a continuation of session <old>. Summary
follows:"*.
**Rationale.** "Never lose user data" and "human-readable JSON" requirements; a
crash mid-write must not corrupt a session.
**Alternatives.** Plain `json.dump` in place — rejected: a crash can truncate the
file.

### C5. Safety of the `bash` tool
**Decision.** Hard-block a small set of catastrophic patterns (`rm -rf /`,
fork bombs, `mkfs`, …). Destructive-but-normal commands are allowed without an
interactive prompt.
**Rationale.** An interactive `input()` prompt inside a tool would steal the
user's next console line and stall automation; the brief favours robustness.
**Alternatives.** Port cipher's confirm-on-danger flow — rejected for the reason
above; it can be re-added behind a flag.

### C6. Crash handling default
**Decision.** On an unhandled error: save the session, print a rich traceback,
and offer a self-diagnosis pass (y/n, default **yes** after 5 s when a TTY is
attached; default yes when non-interactive).
**Rationale.** Brief specifies y/n with default y after 5 s.
**Alternatives.** Always auto-diagnose — rejected: could burn rate limit
unattended.

### C7. Thinking display defaults
**Decision.** Default mode is `spinner`: reasoning is requested from the model
and the last 80 characters are shown in a dimmed single-line status area. `/thinking
on` streams full dim reasoning; `/thinking off` hides it entirely; `/thinking
last` prints the most recent block once. Reasoning is stored under `reasoning`
and is replayed through the v0.4 outgoing protocol without changing the stored
assistant content.
**Rationale.** Satisfies both "render thinking dimmed by default / don't dump to
scrollback" and "/thinking off is the hidden mode".
**Alternatives.** Reasoning off at the API level by default — rejected: there
would be nothing to show or inspect.

### C8. Repo hygiene
**Decision.** `sessions/`, `.mike/` and `.mike_state.json` are git-ignored.
`mike.py` has a shebang and is chmod +x.
**Rationale.** Session data is user data and must not be committed; the file is
meant to be run directly.
**Alternatives.** Commit sessions — rejected.

---

## D. Test summary (Phase 6)

Run inside the project venv, fully hermetic (the OpenAI client is stubbed):

```
$ venv/bin/python -m pytest tests/test_mike.py
tests/test_mike.py ...................................                   [100%]
============================== 35 passed in 1.16s ==============================
```

Covered: prompted parser (single / multiple / malformed JSON / text around the
block / JSON-string arguments / DeepSeek DSML), thinking stripper
(`<think>`, `<reasoning>`, `reasoning_content` on a mock delta, incremental
stream parser), deterministic token estimator, compactor trigger at 400k and
non-trigger at 300k (mock counter), seven-section summary, session rollover to
`-cont1` seeded with the continuation header, pretty/sorted JSON, alias routing
for every alias, and an end-to-end agent turn with a stubbed LLM that fires a
tool and feeds the result back.

---

## E. End-to-end manual check (Phase 7)

All runs below used `venv/bin/python mike.py` with the server auto-launched (no
server was running beforehand). ANSI codes stripped for readability.

### E1. Startup, streamed answer, thinking, file creation, rename

```
$ python mike.py
info: single model DeepSeek V4.1 Flash (deepseek-v4.1-flash); 8 legacy aliases route to it; tools: prompted
mike v1.0.0 - DeepSeek V4.1 Flash (prompted tools, thinking:spinner, ctx 500,000)
server http://127.0.0.1:8000/v1 | session session | cwd /home/mike/Deepseek-API
type /help for commands

> What is 6*7? Answer in one short line.
42

> /thinking last
╭────────────── last reasoning ──────────────╮
│ The user asks a simple math question. No   │
│ tools needed.                              │
╰────────────────────────────────────────────╯

> Create a file named mike_e2e_check.txt containing exactly the line: hello from mike
* write_file(/home/mike/Deepseek-API/mike_e2e_check.txt)
  | wrote 1 line(s) to /home/mike/Deepseek-API/mike_e2e_check.txt
* read_file(/home/mike/Deepseek-API/mike_e2e_check.txt)
  |     1       hello from mike
... verified by read-back.

> /rename hello-world
renamed -> hello-world

> /quit
bye.
```

Confirmed on disk: `mike_e2e_check.txt` contains `hello from mike`. The
model's native DSML tool-call syntax was hidden from scrollback (only the clean
tool header/result lines appear).

### E2. Resume, tokens, compaction, rollover

```
$ python mike.py resume
                                    Sessions
 #   name         updated           messages  size   preview
 1   hello-world  2026-10-08T00:43… 8         2.8KB  Create a file named mike_e2e_check.t…
Select a session to resume [1-1, or q]: 1
server http://127.0.0.1:8000/v1 | session hello-world | cwd /home/mike/Deepseek-API

> /tokens
estimated 502 tokens / threshold 400,000 (max 500,000) via tiktoken

> /compact
╭──────── summary ────────╮
│ ## Goal                 │
│ What is 6*7? ...        │
│ ## Constraints          │
│ ## Files Touched        │
│ - /home/mike/Deepseek-API/mike_e2e_check.txt
│ ## Decisions Made       │
│ ## Open Todos           │
│ ## Last User Intent     │
│ Create a file named mike_e2e_check.txt ...
│ ## Next Steps           │
╰─────────────────────────╯

> /rollover
-> rolled over to session hello-world-cont1 (from hello-world)

> /sessions
 1   hello-world        8 messages
 2   hello-world-cont1  1 message
```

Confirmed on disk: `20261008-004336__hello-world-cont1.json` exists, its first
message is `This is a continuation of session hello-world. Summary follows:` +
the seven-section summary, and the original `hello-world` file still contains the
full 8-message transcript (compaction did not truncate it).

---

## F. Acceptance checklist

| | Criterion | Result |
| --- | --- | --- |
| A | `python mike.py --help` prints usage, exits 0 | ✅ |
| B | starts the server, connects, one model `DeepSeek V4.1 Flash` | ✅ |
| C | sibling dirs renamed → mike still works | ✅ (restored) |
| D | no `openrouter`/`anthropic` in mike.py | ✅ |
| E | `/tokens` threshold = 400,000 | ✅ |
| F | `/compact` summary has all seven sections | ✅ |
| G | `resume` lists sessions (name/timestamp/preview) and restores | ✅ |
| H | forced rollover → `-cont1` file seeded with the summary header | ✅ |
| I | `pytest tests/test_mike.py` passes | ✅ 35 passed |
| J | README.md and DECISIONS.md updated | ✅ |
| K | all work committed, final message `feat(mike): self-contained free-DeepSeek harness with 500k ctx + resume.` | ✅ |

---

# v0.2 — config, /export, hardened tool-call round-trip

## v0.2 Recon

Read-only reconnaissance of the v0.1 codebase before any v0.2 change.

**1. Current precedence for `MIKE_MAX_CONTEXT` and `COMPACT_THRESHOLD`.**
There is no config file layer yet. `MIKE_MAX_CONTEXT` is read exactly once, at
module import time (`mike.py` ~lines 122-126):
`DEFAULT_MAX_CONTEXT_TOKENS = 500_000`, then
`MAX_CONTEXT_TOKENS = int(os.getenv("MIKE_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT_TOKENS)))`
(wrapped in try/except, falling back to the default on a bad value). So today the
precedence is simply **env > hard-coded default**. `COMPACT_THRESHOLD` is a bare
literal `0.80` with **no** env/CLI override, and `COMPACT_AT_TOKENS` is derived
as `int(MAX_CONTEXT_TOKENS * COMPACT_THRESHOLD)`. (`MIKE_PORT` / `PORT` and
`HOST` similarly feed `DEFAULT_PORT` / `DEFAULT_HOST` at import time.)

**2. Functions that parse DSML and `<tool_call>` envelopes.**
One entry point, `parse_tool_calls(text) -> (calls, clean_text)` (mike.py:456). It
extracts `<tool_call>…</tool_call>` blocks with `_TOOL_CALL_BLOCK_RE`, falls back
to fenced ```json objects (`_FENCED_JSON_RE` + `_loads_tool_obj`), then calls
`parse_dsml_calls(text)` and strips the DSML blocks with `strip_dsml(text)`.
DSML itself is parsed by `parse_dsml_calls(text)` (mike.py:427) using
`_DSML_INVOKE_RE` and `_DSML_PARAM_RE`; each parameter value goes through
`_dsml_value(raw, attrs)`. Display-side hiding of both syntaxes is done by
`StreamParser` via `_TAG_RE` / `_tag_name`.

**3. Functions that dispatch a parsed call and format the result back.**
Dispatch is `Toolbox.call(name, args) -> str` (mike.py:683), which looks the tool
up in `Toolbox.specs` and wraps exceptions into an `error: …` string. Feeding the
result back into the conversation is `MikeAgent._run_tools(calls)` (mike.py:1918):
in **prompted** mode it appends `{"role": "user", "content": "<tool_result>…</tool_result>"}`;
in **native** mode it appends `{"role": "tool", "tool_call_id": …}`. Both go
through `MikeAgent._append(msg)` (mike.py:1822) so they land in the on-disk
`session.messages` *and* the in-context `self.context`.

**4. Does the DSML parser handle self-closing / unclosed tags?**
**No.** `_DSML_INVOKE_RE` is
`<[^<>]*?invoke\s+name="…"[^<>]*>(.*?)</[^<>]*?invoke\s*>` — it *requires* a
matching close tag. A self-closing `<…invoke name="x" />`, attribute-style args
(`<…invoke name="read_file" path="x.py" />`), and a truncated/unclosed invoke all
match nothing and are silently dropped today. `_dsml_value` also silently
converts malformed "JSON" parameter values to the raw string only when the tag
lacks `string="true"` (it does keep them as raw strings — so that part is fine).

**5. Session JSON location and schema.**
Sessions live in `SESSIONS_DIR = ROOT / "sessions"` (`~/Deepseek-API/sessions/`)
as `YYYYMMDD-HHMMSS__<slug>.json`. `SessionStore.list_sessions()` / `.find()` read
them via the `Session.load(path)` classmethod (`json.loads`); `Session.save()`
writes via `atomic_write_json` (temp file + `fsync` + `os.replace`, pretty-printed
with sorted keys). Schema:
`{"id", "name", "created_at", "updated_at", "model", "messages": [...], "compactions": [...]}`.
Message dicts are OpenAI-shaped and may additionally carry `reasoning` (stripped
from anything sent upstream) and `tool_calls`; compaction entries are
`{"at_tokens", "summary", "new_session"}`.

---

## v0.2 config precedence

Five layers, resolved once at startup; later wins. Merge is a shallow per-key
`dict.update` (not a deep merge). `mike.py config` prints the winning source.

| # | Layer | Source label |
| - | ----- | ------------ |
| 1 | `DEFAULTS` in `mike.py` | `default` |
| 2 | `~/.config/mike/config.toml` | `user` |
| 3 | `./mike.toml` | `repo` |
| 4 | `MIKE_*` environment variables | `env` |
| 5 | CLI flags | `cli` |

Worked example — `max_context`:

* no config, no env → `500000` (`default`)
* `./mike.toml` says `max_context = 300000` → `300000` (`repo`)
* …and `MIKE_MAX_CONTEXT=250000` is exported → `250000` (`env`)
* …and the command is `mike.py --max-context 200000 config` → `200000` (`cli`)

`MIKE_MAX_CONTEXT` keeps working exactly as documented in v0.1 (env beats
default/repo); it is now simply one layer of the same ladder.

### v0.2 decisions

**D1. `model_id` is an extra config key.** *Decision:* the wire model id lives in
`DEFAULTS["model_id"]` alongside `model` (the display name). *Rationale:* §3.6
wants the model-id literal to occur exactly once; putting it in the defaults
block satisfies both §3.6 and acceptance I, and `MODEL_ID`/`MODEL_ALIASES` derive
from it. *Alternatives:* a bare module constant — rejected (duplicate literal).

**D2. `config` and `export` do not start the server.** *Decision:* these two
subcommands resolve config and either print it or write a file, then exit.
*Rationale:* neither needs the model; launching the server would be slow and
could pop a login window. *Alternatives:* start the server anyway — rejected as
wasteful.

**D3. Default `thinking_default` changed to `off`.** *Decision:* the v0.2 schema
specifies `off`, so the interactive default is now hidden reasoning (v0.1
defaulted to the dimmed spinner). `/thinking on|spinner|last` still work.
*Rationale:* matches §3.2 exactly. *Alternatives:* keep `spinner` — rejected
(contradicts the specified schema default).

**D4. `session_dir` / `export_dir` default to empty string = repo dirs.**
*Decision:* empty means `<repo>/sessions/` and `<repo>/exports/`; a set value is
`expanduser`'d. *Rationale:* keeps the v0.1 locations while allowing overrides.

**D5. Overwrite guard defaults to NO.** *Decision:* `/export` and the export
subcommand refuse to overwrite unless `--force` is given (or the user answers
`y` at an interactive prompt). *Rationale:* never destroy data by default.

## v0.2 export format

One sentence on the reasoning block: reasoning is wrapped in a GitHub
`<details><summary>reasoning</summary>` block so it collapses by default and
stays out of the way when reading a transcript, while still being present
verbatim for audit.

## v0.2 DSML hardening

Parsing is done by `parse_dsml_calls()` (one pass, emission order) and hidden
from user output by `strip_dsml()` / `StreamParser`. Edge cases:

* **Standard open/close invoke** — `_DSML_OPEN_RE` finds the invoke tag,
  `_DSML_CLOSE_RE` finds its close, `_DSML_PARAM_RE` collects the
  `parameter` children. Dispatched with the parsed dict.
* **Self-closing invoke** (`invoke … /`) — detected via the trailing `/` in the
  open tag; dispatched with `{}` as arguments.
* **Self-closing with attributes** — `_attrs_to_args()` turns non-`name`
  attributes into arguments (e.g. `path="x.py"` → `{"path": "x.py"}`).
* **Truncated / unclosed invoke** — when no close tag follows, the rest of the
  message is treated as the body, the partial call is still dispatched, and a
  WARN `tool_call_truncated name=<name>` is logged (test asserts the record).
* **Malformed parameter JSON** — `_dsml_value()` keeps the raw string when
  `json.loads` fails, so the call is dispatched rather than dropped.
* **Multiple invokes** — the scanner iterates every invoke tag in order, so all
  are dispatched, in emission order.
* **Mixed envelope** — `parse_tool_calls()` runs the `<tool_call>` JSON parser
  and the DSML parser over the same message and concatenates the results, so a
  message containing both is fully handled.

`edit_file` also accepts the short `{path, old, new}` form in addition to v0.1's
`{path, old_string, new_string}`, matching §5.2.

## v0.2 e2e — tool-call round-trip

Manual verification of §5.4 against the real model (server auto-launched).
Scratch file `e2e_edit_scratch.txt` started as:

```
line one
OLD_VALUE
line three
```

### Steps 1–3 — edit an existing file, check the disk diff, check scrollback

```
> Use the edit_file tool to edit the file e2e_edit_scratch.txt: replace the exact line OLD_VALUE with NEW_VALUE.
I'll read the file first, then make the edit.
* read_file(e2e_edit_scratch.txt)
  |     1       line one ... +2 lines
* edit_file(e2e_edit_scratch.txt)
  | edited /home/mike/Deepseek-API/e2e_edit_scratch.txt (1 replacement)
* read_file(e2e_edit_scratch.txt)
  |     1       line one ... +2 lines
The edit is done — line 2 of `e2e_edit_scratch.txt` now reads `NEW_VALUE`.
```

On-disk diff confirmed the mutation:

```
 line one
-OLD_VALUE
+NEW_VALUE
 line three
```

Scrollback leak check for real tool syntax markers
(`｜DSML｜`, `<tool_call>`, `invoke name=`, `parameter name=`): **0 matches**.

### Step 4 — repeat with `/thinking on`

```
> /thinking on
thinking: full reasoning shown (dim)
> Use the edit_file tool to edit e2e_edit_scratch.txt: replace the exact line OLD_VALUE with NEW_VALUE.
Let me first read the file, then edit it.            <- dim reasoning (visible)
* read_file(/home/mike/Deepseek-API/e2e_edit_scratch.txt)
  |     1       line one ... +2 lines
The user wants me to edit the file, replacing OLD_VALUE with NEW_VALUE…  <- dim reasoning
* edit_file(e2e_edit_scratch.txt)
  | edited /home/mike/Deepseek-API/e2e_edit_scratch.txt (1 replacement)
…
Done — `e2e_edit_scratch.txt` line 2 now reads `NEW_VALUE`.
```

Reasoning is visible (dim), the file was edited (diff identical to above), and
the tool-syntax leak check was again **0 matches**.

---

## v0.2 conversation threading (bug fix)

**Symptom (reported):** every request — the initial prompt, each tool result,
and every follow-up — showed up as a *separate* DeepSeek chat upstream, even
though no compaction had happened.

**Cause:** `DeepSeekLLM` never sent the server's non-standard `conversation_id`
field. The server creates a new upstream chat whenever `conversation_id` is
absent, so each of mike's requests (one per agent step) opened a fresh chat.

**Fix:** mike now threads the upstream thread id end to end.

* `DeepSeekLLM.chat_stream(..., conversation_id=...)` sends it in the request
  body (`extra_body` for the SDK, JSON for httpx) and yields a
  `conversation_id` event when the server echoes it on the final chunk.
* `MikeAgent` stores it on the session (`Session.conversation_id`, persisted in
  the session JSON, so resume keeps the same chat).
* A send queue (`self._unsent`) tracks messages the thread has not seen. While a
  `conversation_id` exists, mike sends **only the new messages** (the user turn
  and/or tool results) — the thread already holds the earlier turns, so nothing
  is duplicated. The very first request of a thread sends the full context
  (system + tools + history) to establish it.
* Compaction keeps the thread: after compacting, the summary message is queued
  ahead of anything still pending. Rollover/resume reset the queue (a new
  session opens its own thread).

**Verified:** unit test `test_threads_one_conversation` asserts the first request
has no id, the follow-up reuses it, and the follow-up carries only the tool
result. Live check: asking "Remember BANANA", then in a *later* turn "What word
did I ask you to remember?" returned `BANANA` while only the new prompt was sent
— i.e. the same upstream chat was reused. (`sessions/…json` shows the stored
`conversation_id`.)

---

# v0.3 — reference DSML parser + tool-result pairing

## v0.3 motivation

Two problems were reported with the live harness:

1. **Tool calling was prompted-only.** `server/schemas.py` has no `tools` field,
   so OpenAI-style `tools=[...]` is silently dropped and `detect_tool_mode()`
   always returns `"prompted"`. The fix lives in mike's parse/feedback path, not
   the server (the brief forbids touching the server request/response shape and
   the transport).
2. **The model stopped mid-task.** After a tool call the harness appended the
   result as a generic `<tool_result>` **user** message with no link back to the
   call, so the model could not tell its own tool call from a new user turn.

## v0.3 D1. Replace the regex DSML parser with the reference parser

**Decision.** `parse_dsml_calls()` and its helpers (`_dsml_value`,
`_parse_param_tags`, `_attrs_to_args`, `strip_dsml`) now mirror the DeepSeek
reference `encoding.py` parser, anchored on the real `｜DSML｜` token (the bar is
U+FF5C, never ASCII `|`). The previous regex keyed only off the ASCII words
(`invoke`/`parameter`/`calls`) with `<[^<>]*?…>`, which matched *any* tag that
merely contained those substrings — over-permissive and not the reference shape.

**Reference sources fetched and compared** (scratch, in `/tmp/dsrefs`, not
vendored):
- Official Rust `deepseek-recipe` (cloned). Ground-truth tag constants:
  `DSML_SP_TOKEN = "｜DSML｜"`, V4 block name `tool_calls` / V4.1 block name
  `" calls"` (leading space), and the stream parser accepts **both** end labels
  `["</｜DSML｜tool_calls>", "</｜DSML｜ calls>"]`. Rust is the tie-breaker for
  edge cases (it flushes a truncated call on `finish()`; empty invoke ⇒ `{}`).
- HuggingFace `msuiche/DeepSeek-V4.1…/encoding/encoding.py` (pure Python). Has
  the actual **parse** side (`parse_message_from_completion_text`,
  `parse_tool_calls`, `_read_until_stop`, `decode_dsml_to_arguments`). This is
  the base we vendored mechanics from.
- vLLM `deepseek_v41_encoding.py` (pure Python). Confirmed **encode-only** (no
  parse function) but it documents the tool-result template.

**Base chosen: HF `encoding.py`** (only pure-Python *parse* implementation),
generalized to accept **both** V4 (no-space) and V4.1 (space) tag forms per the
Rust rules, with Rust-style tolerance instead of HF's strict `raise`.

**V4 vs V4.1 difference.** V4 writes `<｜DSML｜tool_calls>` / `<｜DSML｜invoke>`
(no space after the token). V4.1 writes `<｜DSML｜ calls>` / `<｜DSML｜ invoke>`
(leading space). The regexes allow optional whitespace and either keyword
(`(?:tool_calls|calls)`), so both are accepted. Verified by unit tests
`test_dsml_v41_space_form` and `test_dsml_v4_no_space_form`.

**Tolerated edge cases** (all asserted): standard open/close, self-closing with
`{}`, attribute-style args on a self-closing invoke, truncated invoke dispatched
with a WARN `tool_call_truncated name=…`, malformed parameter JSON kept as the
raw string, multiple invokes in emission order, and a block split across two SSE
frames (mike parses the reassembled `StreamParser.raw`, so frame boundaries never
break a tag).

## v0.3 D2. Tool results are paired to their call id

**Decision.** `parse_tool_calls()` returns the pure reference shape
(`{"name","arguments"}`); `MikeAgent.run` normalises a stable `id` onto every
call dict **before** building the assistant message and before `_run_tools`, so
both reference the SAME id. In prompted mode `_run_tools` merges **all** results
of a turn into a single following **user** message, each wrapped as
`<tool_result name="…" id="call_…">output</tool_result>` in call order — mirroring
the reference `merge_tool_messages` + `sort_tool_results_by_call_order`. An INFO
line `tool result queued: name=… id=… bytes=…` is emitted per result.

**Which candidate A/B was chosen — neither, deliberately.** The brief offered
candidate A (result as an *assistant* message) and candidate B (result as a
*system* message). The reference is unambiguous: DeepSeek-V4.1 has no standalone
`tool` role and encodes results as `<tool_result>` blocks inside a **user**
message (`tool_output_template = "<tool_result>{content}</tool_result>"`). That is
what mike already used, so the role was *not* the bug — the missing call-id
pairing and the un-merged multi-result shape were. We kept role=user, added the
id, and merged results into one user turn.

**Why "stops mid-task" is fixed.** Previously multiple tool results in one turn
were appended as several separate user messages; the server's
`messages_to_prompt` then rendered them as multiple `User:` turns, which the model
does not associate with the single assistant tool-call turn that preceded them.
The merged single user message, ordered and tagged by call id, restores the
assistant→tool-result pairing the model continues from.

**Live evidence (real model, server auto-launched).** Three runs:
- TEST 1 — one write_file call; `/tmp/dsml_v4_test.txt` = `hello-dsml`.
- TEST 2 — the multi-step case: the model issued write_file×3, continued after
  each merged result, then read_file×3, then summarised — **no early stop**; the
  log shows six `tool result queued: … id=call_…` lines.
- TEST 3 — write_file then bash in sequence; script created and `python3
  /tmp/hello.py` ran, output `hi`.

**Honest note on the DSML path.** `detect_tool_mode()` still returns `prompted`
(the web bridge exposes no native tool channel), and in these live runs the model
honoured the requested `<tool_call>{json}</tool_call>` envelope rather than raw
DSML. Both paths flow through the same id-threading and merged-result fix, and
both are unit-tested (`test_dsml_*` for DSML, `test_agent_runs_tool_then_answers`
/ `test_*_roundtrip` for the envelope and DSML round-trips). The DSML work is the
correctness fix the brief required; the pairing fix is what made the model
continue.

---

# v0.4 — Codex-style completion and `apply_patch`

## Completion is explicit; commentary is not final

**Decision.** The agent loop only terminates normally when the model emits a
structured `task_complete` call. A text-only reply is stored as commentary and
followed by exactly one nudge asking the model to continue or call
`task_complete`; a second text-only reply is the bounded fallback final answer.
The system prompt introduces `[commentary]` / `[final]` guidance, but those
labels are not required in prompted mode.

**Rationale.** DeepSeek often says “I’ll read the file next” and stops without
making the call. Breaking on the first empty `tool_calls` list treats that
promise as a final answer. An explicit structured signal matches Codex’s
completion model while retaining a finite failure mode when the web model
ignores the tool.

**Alternatives.** Infer completion from clean text or `finish_reason: stop` —
rejected because both are routinely emitted for progress messages. Loop until
`task_complete` with no fallback — rejected because a free web model that never
calls the tool would hang the turn.

## `apply_patch` replaces direct write/edit tools

**Decision.** `write_file` and `edit_file` are no longer registered. The sole
file-mutation tool is a Codex-style, line-oriented `apply_patch` with Add,
Update, Move, Delete, hunks, and EOF insertion support. The whole envelope is
parsed and all hunk contexts are validated against a staged view before any
disk write. Add File overwrites and ensures a trailing newline; bare blank hunk
lines are accepted as empty context.

**Rationale.** One constrained patch surface makes edits inspectable, supports
multi-file changes atomically from the model’s perspective, prevents partial
application on a context error, and gives a clear place to enforce “no shell
redirection / `cat` / `sed -i`” edits. Old sessions containing `write_file` or
`edit_file` calls remain valid JSON history; only the live registry changed.

**Alternatives.** Keep string-replacement editing beside the patch tool —
rejected because two mutation paths weaken the edit protocol and make prompts
ambiguous.

## Reasoning replay and server schema

**Decision.** Stored assistant `reasoning` is replayed on every later request.
In prompted mode mike prepends a `<reasoning>…</reasoning>` block to the
outgoing assistant content only; in native tools mode it sends a top-level
`reasoning_content` field. `server.schemas.ChatMessage` accepts
`reasoning_content`, and `messages_to_prompt()` serializes it as
`Assistant (reasoning): …`. Non-streaming completion responses preserve the
field when reasoning is returned.

**Rationale.** DeepSeek rejects thinking-mode requests with tools when prior
assistant reasoning is omitted. The flat-prompt bridge otherwise discards the
field. Tagged prompted replay keeps the flat architecture while the named field
keeps native OpenAI-shaped requests compatible.

**Alternatives.** Replay reasoning as ordinary visible assistant content —
rejected because it pollutes the transcript and answer semantics. Change the
bridge to native Chat Completions tool handling — outside this hardening pass
and unnecessary for the workaround.

## Compaction is explicit continuation context

**Decision.** Compactor summaries now start with “The conversation was
compacted…” and the system prompt contains a **Compaction awareness** section.
Section extraction skips both the new prefix and legacy `[CONTEXT SUMMARY]`
messages. When a `conversation_id` exists, the new summary is queued as the
first unsent message and prior queued summaries are filtered out.

**Rationale.** A bare `[CONTEXT SUMMARY]` user message can look like a fresh
request and cause the model to restart. Explicit continuation language keeps
the last intent current and avoids duplicate summaries in an existing thread.

**Alternatives.** Change the summary role to system — rejected because the web
bridge flattens roles and the existing user-message threading is known to work.

## DSML safety net after `finish_reason: "stop"`

**Decision.** If a stopped response parses to no calls but its raw content still
contains `DSML`, `<tool_call>`, or `invoke name=`, mike logs
`dsml_unparsed_retry`, warns the user, and sends one stricter continuation
nudge. The retry is allowed once per user turn; subsequent unparsed syntax falls
through to normal commentary/nudge handling.

**Rationale.** The free web model can return a complete-looking or malformed
tool block as plain content with `finish_reason: stop`. Silently treating that
as completion reintroduces the early-stop bug even with `task_complete`.

**Alternatives.** Retry indefinitely — rejected because malformed generations
can repeat forever. Parse arbitrary content heuristically — rejected because it
would execute unvalidated tool syntax.

## v0.4 verification

`venv/bin/python -m pytest tests/test_mike.py -q` covers the new patch grammar,
atomic context failure, missing tool registry entries, `task_complete`
termination, the one-shot commentary nudge, prompted/native reasoning replay,
compaction headers and awareness, and the DSML `stop` retry. Result:

```
97 passed in 3.02s
```

The live smoke test ran `venv/bin/python mike.py` against the local bridge with
the requested two prompts. The model called `apply_patch` (summary output
`added hello.txt`), read `hello.txt` back and received `hi`, then emitted its
final completion turn without a commentary nudge. The explicit
`task_complete` call ended the turn and is omitted from stored `tool_calls` as
designed. The smoke session and `hello.txt` were removed afterwards.

---

# v0.5 — Codex-style inline TUI

## Reference review

**Decision.** The UI pass was based on a shallow sparse checkout of the official
`openai/codex` repository. The reviewed TUI sources included
`codex-rs/tui/src/bottom_pane/chat_composer.rs`, its `slash_input.rs`,
`popup_state.rs`, and `footer_state.rs` modules,
`codex-rs/tui/src/status_indicator_widget.rs`,
`codex-rs/tui/src/history_cell/plans.rs`, and
`codex-rs/tui/src/diff_render.rs`. The installed `codex --help` was also checked.

**Observed patterns.** Codex keeps the composer at the bottom, opens a command
popup above it for slash input, preserves history, displays a single animated
working row with elapsed/context information, renders plans as compact
checklists, and gives diffs per-file headers, line counts, gutter signs, and
syntax-aware coloring.

**Scope decision.** Mike remains an inline scrollback REPL. Recreating Codex's
Ratatui alternate-screen layout and frame ownership would be a separate UI
rewrite; matching the interaction primitives and visual hierarchy provides the
polish requested without destabilizing the agent loop.

## Prompt composer

**Decision.** `prompt_toolkit` supplies the `PromptSession`, `FileHistory`
stored at `.mike_history`, slash-command and argument completion, styled prompt,
and multiline key bindings. Enter submits unless a completion is selected;
`Esc`, then Enter, and Ctrl+Enter (typically delivered as Ctrl+J) insert a
newline. A plain `input()` fallback remains for environments without
prompt_toolkit or a TTY.

**Rationale.** This mirrors Codex's command discovery and multiline authoring
without adding a new Python module or replacing the existing command dispatcher.

## Rich diffs and plan observability

**Decision.** `apply_patch` keeps the existing two-phase atomic validation, then
renders each validated file change with `difflib.unified_diff` and Rich
`Syntax("diff", theme="monokai")`. The model receives only compact operation
summaries and line/hunk deltas.

**Decision.** `todo_write` is replaced by structured `update_plan`, with
`step`/`status` objects and `pending`, `in_progress`, or `completed` states.
`todo_read` remains as the read path. Rich renders the checklist with green
checkmarks, a yellow spinner, and grey circles, while preserving old string-only
plan files during load.

**Rationale.** The model gets a precise execution plan while the user gets
Codex-style observability and patch review without a second edit surface.

## Inline status and Markdown transcript

**Decision.** A status row before the composer reports model, context estimate,
plan completion, and abbreviated cwd. During a request, a transient Rich
spinner includes the same context usage. Assistant content is buffered until
tool syntax is stripped, then rendered as Markdown with the Monokai code theme;
visible reasoning is shown above a dim rule.

**Rationale.** Buffering avoids rendering tool envelopes or partial reasoning
tags, while preserving the exact stored session content and the existing
`task_complete`, commentary nudge, DSML retry, and reasoning replay loops.

## v0.5 verification

`venv/bin/python -m pytest tests/test_mike.py -q` passes with 102 tests. New
coverage asserts slash-command/argument completion, `.mike_history` and
multiline prompt configuration, structured plan rendering and persistence,
status-bar context/plan counts, compact patch summaries, and numbered unified
diff output.

The live smoke test ran against an isolated bridge on port 8010. Slash
completion selected `/help`, Ctrl+J inserted a multiline prompt, `apply_patch`
created `t.py` and rendered a numbered diff, `update_plan` rendered
`✓`/`⠋`/`○` checklist states, and the status line advanced from `Ctx 0` to
`Ctx 548` and `Plan 0/0` to `Plan 1/3`. The model then hit an empty upstream
stream during its dependency/run check; the UI surfaced that error cleanly while
the existing loop and recovery messaging remained intact. The smoke-created
`t.py`, `.mike_history`, plan file, and sessions were removed.

## v0.6 final polish

**Decision.** The system prompt now asks for short natural-language progress
updates before tools and no longer requests literal channel labels. A defensive
case-insensitive filter removes hallucinated `[commentary]` and `[final]` tags
from assistant text and `task_complete` summaries before display or storage.

**Documentation.** Long-form design and reconnaissance notes now live under
`docs/`, with README links pointing to `docs/DECISIONS.md` and
`docs/RECON.md`. Ignore rules cover the complete session directory, persistent
prompt history, local state, and common editor/cache temporary files.

# v0.7 — reference DSML parser port + live TUI activity

## v0.7 motivation

The v0.3 parser was still regex-based and mis-parsed several real model
outputs: spaced tags (`<｜DSML｜ calls>`), self-closing or truncated invokes,
parameters without `string="true"`, and — most visibly — orphaned closing
fragments leaking into argument values, e.g. `{"cmd": "pwd</｜｜DSML｜｜>"}`,
which reached the shell and failed. The TUI also showed a static
`Working (0s)` with no motion, which looked like a hang.

## v0.7 D1. Parser modeled on the official deepseek-recipe state machine

**Decision.** `backend/server/toolcall.py` now parses DSML with a linear
scanner that follows the semantics of
`deepseek-recipe/src/stream/output_parser.rs` and `processor.rs`: tag
recognition for both `<｜DSML｜tool_calls>`/`<｜DSML｜invoke>` and the spaced
V4.1 form, `string="true"` values taken verbatim, `string="false"` (or absent)
values JSON-decoded with a raw-string fallback, empty invokes yielding `{}`,
values flushed when the next DSML tag appears, and everything after the closing
block ignored. Any run of `｜` counts as the delimiter, because the model
sometimes emits `｜｜DSML｜｜` or more.

**Deliberate deviation.** The reference drops parameters lacking a `string`
attribute; we default them to string, because the bridge sees whole responses
(not a token stream) and being forgiving loses fewer tool calls.

**Rationale.** The reference is the authoritative description of the format
DeepSeek actually emits; behavior-matching it removes the recurring migration
failures. It is stdlib-only (regex precompiled patterns + a small scanner), so
no new dependency was added.

## v0.7 D2. Orphan `｜DSML｜` fragments are stripped from arguments

**Decision.** Every extracted argument (name, string, list, dict, recursively)
passes through `_clean_arguments`, which removes stray
`</?｜+DSML｜+…>` tags and bare `｜+DSML｜+` tokens and trims the result. The
same cleaning is applied when normalizing parsed calls and when replaying
history to upstream. `</｜｜DSML｜｜>` after `pwd` therefore yields `pwd`.

**Rationale.** Orphan tags are what motivated this task; they must never reach
tool arguments, regardless of which parse path produced them (JSON envelope,
fenced JSON, DSML, or history replay).

**Alternatives.** Cleaning only in the DSML parser — rejected: the JSON
envelope path reproduces the same bug when the model inlines the fragment in a
string.

## v0.7 D3. TUI spinner and live timer

**Decision.** `StatusIndicatorWidget` renders the existing braille glyph
(`motion.rs` FRAMES, 100 ms) before the header whenever
`MotionMode::Animated`, and `render()` always schedules a frame (32 ms when
progress/shimmer are on, 1 s otherwise) so the elapsed timer ticks live from
`StatusTimer::last_resume_at` and the glyph advances. Both disappear at turn
completion because the row is only drawn while a turn is active.

**Config change.** `animations = false` was removed from `mike-config.toml` and
`~/.mike/config.toml`; the master switch therefore defaults to on, and
`[tui.effects] welcome = false` still suppresses the welcome animation.

**Rationale.** The timer was static because the row only scheduled 1 s frames
when a display start existed and progress was disabled by config. Reusing the
existing glyph keeps one spinner implementation across the TUI.

## v0.7 verification

- `cd backend && venv/bin/python -m pytest tests/` → 129 passed.
- New regression tests: orphan fragment cleanup
  (`{"cmd": "pwd</｜｜DSML｜｜>"}` → `{"cmd": "pwd"}`) asserted on the parsed
  call and on the streamed SSE events, spaced/tight tags, raw JSON and missing
  `string`, empty invoke, and trailing garbage.
- `cargo test -p codex-tui --lib status_indicator` → 26 passed with updated
  snapshots (`•` replaced by the first braille frame `⠋`); snapshots remain
  deterministic because the tests pin the timer to the first frame.
- `cargo build --release -p codex-cli` (56m) installed as `~/.local/bin/mike`.
- Manual PTY run (`Run pwd`): all ten braille frames observed at ~100 ms, the
  label ticked `0s` → `5s` live, `pwd` returned `/tmp/mike-smoke`, and the row
  was replaced by `Worked for 5s` at completion. A multi-step follow-up
  (`apply_patch` create `smoke.txt` with `hello`, read it back) executed
  end-to-end and the file contained exactly `hello\n`, confirming tool calls
  parse cleanly through the new parser.
