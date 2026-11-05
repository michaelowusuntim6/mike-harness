# Decisions

## v0.11 D1. Five retries with exponential backoff

**Decision.** Transport retries default to five attempts (`request_max_retries`
5, was 4) and the local bridge provider uses a 1 s base delay, so retries wait
1s, 2s, 4s, 8s, 16s with ±10% jitter
(`codex_client::exponential_backoff`). 5xx, connection resets, and timeouts are
retryable; provider-level retries are reset from 2 to the same default (5).

**Rationale.** On an unstable connection the old budget (200 ms base, four
attempts) gave up in under two seconds, well inside a typical Wi-Fi blip. The
bridge is matched by loopback origin (`ModelProviderInfo::is_mike_bridge`) so
remote providers keep their fast 200 ms base delay.

## v0.11 D2. Bridge stream drops reconnect in place

**Decision.** For the Mike bridge, SSE failures after headers are received
(connection reset, truncated stream, idle timeout) reconnect up to five times
on the exponential schedule, notifying the TUI with `Reconnecting... n/5`. If
the bridge reports a usable upstream thread, the reconnect's
`response.completed` carries that `conversation_id` and session state survives.
The bridge itself keeps its own recovery: one fresh-thread fallback when an
expired upstream thread fails before producing output.

**Limitation.** The bridge is stateless per request, so a mid-generation drop
that happens before any output is re-sent in full; a drop after partial output
ends the turn with an error instead of silently duplicating text.

## v0.11 D3. Context window 120,000 tokens

**Decision.** The DeepSeek catalog, the built-in fallback model metadata
(`models-manager/src/model_info.rs`), `backend/mike.py` defaults, and
`mike.toml.example` all use `context_window = 120000`, with auto-compaction at
80% (96,000 tokens, `auto_compact_token_limit = 96000`). The catalog sets
`effective_context_window_percent = 100` so the footer's window equals the
configured window; the default 95% reservation would have printed `/ 114,000`.
The TUI footer and the status-line docs show `/ 120,000`.

**Rationale.** 500,000 was aspirational; the practical window before upstream
quality degrades is far smaller, and compaction at 96k keeps turns inside it.

## v0.11 D4. Branding: `mike resume`

**Decision.** User-facing exit/resume hints use the installed binary name:
`mike resume <id>`, `Or run mike resume and select <name>`, the disconnect
hints (`mike resume`, `mike --remote <addr>`), the parent-session hint, and the
goal-resume prompt. Internal crate names, telemetry keys, protocol strings,
and config keys are unchanged.

**Rationale.** The binary is installed as `mike`, so printing `codex resume`
handed users a command that does not exist.

## v0.11 D5. Degenerate DSML output still yields a tool call

**Decision.** The bridge recovers a usable call from the model's repetition
collapse instead of ending the turn. Four shapes are now accepted, in both the
modular bridge (`server/toolcall.py`) and the standalone `mike.py` parser:
doubled bars (`<｜｜DSML｜｜ …>`); a close tag that lost its `<`
(`true｜｜DSML｜｜ parameter>`); the same parameter re-emitted until the token
limit truncates the last copy, where the *fullest* value wins rather than the
last one; and the whole payload wrapped in a single parameter named `arguments`
(or a bare `arguments` marker such as `true`), which is unpacked onto the call
or dropped. An unclosed container also no longer absorbs the parameters of the
next block.

**Rationale.** The prompt teaches the JSON envelope
`{"name": …, "arguments": { … }}`, and when the model transposes that shape into
its native DSML the result is `{"arguments": "{\"cmd\": …}"}` — `exec_command`
arrives with no `cmd`, the tool errors, and the agent stops mid-task. The
collapse variants came from live sessions in which a truncated block produced
the same dead end.

**Limitation.** Longest-value-wins assumes a repeated parameter is a collapse
artifact. A model that deliberately shortens a parameter keeps the older,
longer value.

## v0.11 verification

- `cd backend && venv/bin/python -m pytest tests/` → 141 passed (defaults,
  compaction threshold, and status-bar assertions moved to 120,000/96,000; the
  D5 collapse and wrapper shapes are covered in
  `tests/test_responses_bridge.py` and `tests/test_mike.py`).
- `cargo test -p codex-cli --lib` → 16 passed; `cargo test -p codex-utils-cli
  --lib` → 21 passed; `cargo test -p codex-client --lib` → 2 passed;
  `cargo test -p codex-model-provider-info --lib` → 40 passed; core
  `responses_retry` tests → 3 passed.
- New tests cover the `codex-client` backoff schedule, the bridge provider
  retry budget and 1 s base delay, and the bridge reconnect budget (5
  exponential steps).
- TUI `codex-tui` lib suite: 72 branding-only snapshot diffs accepted plus the
  three disconnect-exit snapshots (`daemon`, `remote`, `interrupted`); all 18
  `background_exit_tests` pass. Outside that group, 25 `app::tests` checks and
  the `replayed_in_progress_turn` snapshot still fail on drift that predates
  this work (spinner glyph, dropped `default` service-tier label, model and
  provider-selection fixtures). Each was re-run against an unmodified `aac218a`
  build and fails identically there; the only pass/fail difference,
  `agents_overview_acknowledges_inactive_steer_before_interrupt`, fails the same
  way at `aac218a` when its module runs in parallel.
- `cargo build --release -p codex-cli` (exit 0), installed via
  `./scripts/install_mike.sh` → `mike 1.0.0`.
- Manual PTY smoke test on the installed binary: send one message, footer shows
  `Ctx: 2,667 / 120,000`, and the exit hint reads
  `mike resume <id>` / `Or run mike resume and select New Coding Session.`
- Upgrading an existing install: `scripts/install_mike.sh` copies the catalog
  with `cp -n`, so an existing `~/.mike/mike-models.json` keeps its old values.
  Copy `mike-models.json` over it to pick up 120,000.
  (Superseded by v0.12 D2: the installer now replaces the catalog and keeps a
  `.bak` of whatever it overwrote.)

## v0.12 D1. Embedded DeepSeek tokenizer replaces byte/4 estimation

**Decision.** DeepSeek's published HuggingFace tokenizer
(`assets/tokenizer.json`, BPE, 129,280 tokens) is embedded in both halves of the
harness. `backend/server/token_counter.py` is the bridge's single counter and
`codex-rs/utils/tokenizer` (`codex-utils-tokenizer`) is Rust's; both load the one
asset and count with `tokenizers` 0.23.x. The bridge counts the fully serialized
prompt — base and developer instructions, user text, reasoning replay, tool-call
JSON and `<tool_result>` payloads — so the `Ctx:` the user reads is the real
request size. `tiktoken`'s `cl100k_base` stays only as a fallback in `mike.py`'s
standalone counter, behind the BPE tokenizer and ahead of `len(text)//4`.

**Rationale.** Every previous count was `ceil(bytes/4)`, which on this repo's own
source is off by +10% on prose and −30% on identifier-dense code, so compaction
could fire a third too late. `cl100k_base` is the wrong vocabulary entirely.
Counting is non-fatal by design: a missing library or asset logs one warning and
falls back to the heuristic rather than breaking the bridge.

**Scope limit.** Exact counting is used for *accounting* (context totals,
compaction thresholds, resume estimates, retention). Truncation keeps `ceil(bytes/4)`
because those sites convert a token budget **into** a byte cap — a BPE count is
not invertible — and they run on multi-hundred-kilobyte tool output where a
tokenizer would cost 4× for no decision benefit. The realtime transcript budget
(`core/src/realtime_context.rs`) also stays heuristic: the feature is off in this
deployment and its math is byte-budget `usize` code.

**Item accounting rule.** `estimate_item_token_count` counts an item exactly when
its whole model-visible payload is text, and keeps the existing whole-item byte
heuristic when the item also carries images, audio or encrypted blobs — mixing
methods inside one item would either double-count or need a subtraction the byte
estimator cannot express. Plaintext reasoning summary and content are now counted;
the byte estimator scored them zero even though this bridge replays earlier
thinking into every prompt. Transport ids and envelope framing stay out, because
the bridge's reported usage already covers them.

## v0.12 D2. Context window 400,000 tokens (supersedes v0.11 D3)

**Decision.** `context_window` and `max_context_window` become 400,000 with
`auto_compact_token_limit = 320000` (80%), in `mike-models.json`, the Rust
fallback in `models-manager/src/model_info.rs`, `backend/mike.py` defaults and
`mike.toml.example`. `effective_context_window_percent` stays 100 so the footer
prints `/ 400,000` rather than a reserved percentage. The Rust fallback needed the
same override: it still carried the upstream default of 95%, which made
`usable_context_window()` report 380,000 whenever the catalog is absent — a new
test in `model_info_tests.rs` pins the fallback to the catalog's numbers. Because the catalog is a
generated artifact of this repo and not user data, `install_mike.sh` now replaces
`~/.mike/mike-models.json` instead of `cp -n` skipping it, keeping the displaced
file as `mike-models.json.bak`.

**Rationale.** v0.11 walked 500,000 down to 120,000 on the theory that a large
window was aspirational. With exact counting the number is no longer a guess, and
the harness can actually see how close to a limit it is, so the ceiling is raised
deliberately and can be lowered by editing one JSON file (or with the
`model_context_window` / `model_auto_compact_token_limit` config keys).

**Known risk.** DeepSeek's web chat publishes no 400k guarantee and quality is
known to degrade on very long contexts; `tokenizer_config.json` advertises
`model_max_length = 1048576`, which is a model limit, not a usable session limit.
This is an intentional, reversible bet rather than a measured one.

## v0.12 D3. Parity fixtures pin both languages to the same counts

**Decision.** `assets/tokenizer_parity.json` holds text → exact token vectors
generated by `scripts/gen_tokenizer_parity.py`. The Python suite and the Rust
crate's tests both assert every vector, and the Rust crate embeds the same
`assets/tokenizer.json` rather than a second copy.

**Rationale.** Two implementations of the same BPE can disagree silently (special
token handling and normalizer plumbing both changed after `tokenizers` 0.19, which
is why both sides are pinned to 0.23.x). The fixture is the invariant: a version
bump, an edited asset, or a refactor that shifts a count fails a test instead of
quietly changing the user's context number. It includes the special-token cases
(`<｜begin▁of▁sentence｜>`), CJK, emoji with ZWJ, base64, and tool-call JSON.

## v0.12 verification

- `cd backend && venv/bin/python -m pytest tests/` → 148 passed, including
  `tests/test_token_counter.py` (parity vectors, cache behaviour, and a
  missing-asset test proving the bridge still emits a well-formed
  `response.completed` on the heuristic).
- `cargo test -p codex-utils-tokenizer` → 5 passed. `counts_match_the_python_vectors`
  asserts all 16 parity vectors from Rust, which is the cross-language proof.
- `cargo test -p codex-models-manager --lib` → 57 passed, including
  `deepseek_fallback_metadata_declares_the_mike_window`, which caught that the
  Rust fallback still reserved 95% and would have printed `/ 380,000`.
- `cargo test -p codex-core --lib` → 2836 passed, 16 failed. All 16 reproduce on
  an unmodified HEAD worktree built with the same filter (code-mode host binaries
  absent, guardian/model-catalog and config drift that predates this work); one
  of them passes when run alone, i.e. it is parallel-run sensitive. The only
  expectation changed here is
  `guardian::request_budget::tests::request_histogram_counts_admitted_steps...`,
  whose analytics bucket moved from `02_20k-40k` to `01_0-20k` because the same
  request really is under 20k tokens when counted exactly.
- `cargo test -p codex-cli --lib` → 16; `-p codex-utils-cli --lib` → 21;
  `-p codex-protocol --lib` → 369; `-p codex-client --lib` → 2;
  `-p codex-model-provider-info --lib` → 40. All green.
- Live bridge against the running service: a one-line request returned
  `input_tokens: 223, output_tokens: 3` from the BPE counter rather than `len//4`.
- `cargo build --release -p codex-cli` (90m31s, exit 0) and
  `./scripts/install_mike.sh`, which placed the identical tokenizer asset under
  `~/.local/lib/mike/backend/assets/` and left `~/.mike/mike-models.json` at
  400,000. Manual PTY smoke on the installed binary: footer read
  `Ctx: 2,511 / 400,000` for a prompt the heuristic had reported as 2,667 — the
  ~6% drop is the correction this change exists to make, not a regression.

## v0.13 D1. Compaction rewinds the DeepSeek thread instead of forking a chat

**Decision.** A compaction keeps the `chat_session_id` and drops only the
`parent_message_id`, so the summary re-roots the *same* upstream conversation.
The conversation id is an opaque `"<chat_session_id>:<message_id>"` token, and
`_decode_cid` already maps a bare session id to `(session_id, None)`, so the
rewind is expressible in the existing wire format: `deepseek/client.py` gains
`rewind_cid()` to produce that form explicitly, `server/responses.py` detects the
compaction turn and sends the whole compacted transcript on the bare session id,
and the Rust side preserves the token — `compact.rs` no longer clears it before
summarising, and `SessionState::replace_annotated_history` rewinds for
`HistoryReplacement::Compaction` while `Reset` still starts a fresh thread.

**Rationale.** Dropping the whole token made the bridge mint a new
`chat_session_id` for the next request, so every compaction added a chat to the
user's DeepSeek sidebar and abandoned the server-side thread. The web UI rewrites
a prompt through `chat/edit_message`, which the bridge does not speak; the
equivalent the `chat/completion` format affords is exactly this — same session,
`parent_message_id: null` (traced in
`/home/mike/Creator/rebuild/docs/DEEPSEEK_REWIND.md`).

**Guard.** Detection looks only at the transcript tail after the last
assistant-produced item. The summary stays in history for the rest of the
session, so scanning the whole transcript would rewind every subsequent turn and
orphan each fresh reply — there is a test for precisely that case.

## v0.13 D2. Summarisation rides the existing thread

**Decision.** The compaction request itself now passes the live conversation id
(`drain_to_completed` used to hardcode `None`), so the model is asked to summarize
on the thread that already holds the conversation, and the rewind afterwards
orphans that branch.

**Rationale.** Sending no id made the bridge open a *second* chat just to produce
the summary — the duplicate the user sees in the sidebar. The thread already has
the context, so no transcript needs re-sending for the summary, and the message is
never stored locally.

**Known gap.** The stream-error fallback still clears the id and reseeds a fresh
thread when resuming fails; that path is error recovery, not compaction, and is
unchanged.

## v0.13 verification

- `cd backend && venv/bin/python -m pytest tests/` → 154 passed. New rewind
  tests cover: the message-less cid form, full-transcript re-send on rewind, the
  summary-marker fallback when no continuation exists, typed `context_compaction`
  items, and — the case that matters — an old summary still sitting in history
  must **not** rewind a normal later turn.
- Three pre-existing bridge tests used colon-less fake conversation ids
  (`"conv-1"`, `"stale-thread"`). Those are structurally indistinguishable from
  the rewind encoding, so the fixtures now use the real `<session>:<message>`
  shape instead of the rule being weakened to accommodate them.
- `cargo test -p codex-core --lib state::session` → 9 passed, including
  `compaction_rewinds_conversation_id_while_reset_clears_it` and
  `rewind_conversation_id_keeps_the_session_part`.
- `cargo test -p codex-core --lib` (full) → 2840 passed, 15 failed; the 15 are
  byte-for-byte the same set that fails on an unmodified HEAD worktree (missing
  code-mode host binary, guardian and model-catalog drift that predates this
  work). Zero regressions.
- Live against the running bridge and real DeepSeek: four turns
  (remember code → ordinary turn → rewound compaction turn → recall) produced
  `responses: rewinding upstream thread c6be843c… to the session root`, a single
  `chat_session_id` for the whole run, the post-rewind turn returning
  `<sid>:6` so normal appends resumed, and the model answering `ALPHA-9` from the
  re-rooted branch.

- End-to-end through the TUI on the installed binary (`/compact` between turns),
  from the session rollout and the bridge log:
  `ACK` → compaction produced a `# Handoff Summary` naming **ALPHA-9** →
  `responses: rewinding upstream thread fd5ffc63… to the session root` →
  the follow-up question was answered `ALPHA-9`, with exactly one
  `chat_session_id` for the entire run. One DeepSeek chat, rewound in place.
- Observed wire shape worth noting: the core records the summary as an
  *assistant* item, not a user message, so a prefix scan on user content cannot
  be the primary signal. The message-less conversation id carried the rewind, as
  designed above.

## v0.14 D1. Wrapper-agnostic tool-call extraction

**Decision.** `server/toolcall.py` no longer tries to recognise each wrapper the
model invents. It finds candidate JSON objects by matching braces (quote- and
escape-aware, so nested objects and braces inside strings are handled), validates
them with `json.loads`, requires a `"name"` key, then deletes the object together
with whatever markup sat around it. The per-shape paths that already worked
(envelope, fenced JSON, DSML) are unchanged and still tried first, so this is a
last resort rather than a rewrite.

Wrappers now accepted, all from live output: a bare DSML container tag around a
JSON object, a plain `call` tag pair, the same with a full-width bar glued to the
open tag (with and without the bar on the close), the standard envelope, and no
wrapper at all when the model panics.

**Guard.** A JSON object is only treated as a call when a recognised wrapper tag
sits immediately adjacent to it, or when the object accounts for the whole
message. Assistant prose that happens to quote an example payload is therefore
never eaten — there is a test for exactly that, plus one for malformed JSON
inside a wrapper, which must not become a call.

**Also new.** Full-width bars are stripped from arguments and from user-visible
text, the multiple-object path removes spans right-to-left so offsets stay valid,
and `looks_like_tool_syntax` recognises the new shapes so the corrective nudge
fires when a wrapper appears but nothing parses.

## v0.14 D2. Context window 128,000 tokens (supersedes v0.12 D2)

**Decision.** `context_window` and `max_context_window` become 128,000 with
`auto_compact_token_limit = 102400` (80%), in the catalog, the Rust fallback
metadata, `backend/mike.py` defaults and `mike.toml.example`.
`effective_context_window_percent` stays 100, so the footer reads `/ 128,000`.
Sanity check against the protocol helpers: `usable_context_window()` = 128,000
and `auto_compact_token_limit()` = `min(102400, 115200)` = 102,400.

**Rationale.** The 400,000 window was a deliberate bet that the web chat would
hold up; it is being walked back to a size that stays inside the range DeepSeek
actually handles well, with compaction firing earlier. Reverting is still one
JSON file (`mike-models.json`) or the two `model_context_window` /
`model_auto_compact_token_limit` config keys.

## v0.14 D3. Two byte heuristics: 4 for budgets, 3 for guards

**Decision.** The estimator is split instead of retuned. `APPROX_BYTES_PER_TOKEN`
stays at 4 and keeps driving every token-to-byte *budget* conversion
(`approx_token_count`, `approx_bytes_for_tokens`, `approx_tokens_from_byte_count`),
so truncation caps and tool-output sizes are bit-for-bit what they were. New
`CONSERVATIVE_BYTES_PER_TOKEN = 3` backs `conservative_bytes_for_tokens()` and
`conservative_tokens_from_byte_count()`, used only by absolute-limit guards —
oversized thread instructions (`agents_md_manager.rs`) and managed developer
instructions (`managed_developer_instructions.rs`) — and by the Python fallback
estimators (`token_counter._heuristic`, `openai_format._est_tokens`,
`mike.py`'s heuristic path).

**Rationale.** DeepSeek BPE averages nearer three characters per token, so a
guard built on four lets oversized text through, and over-estimating is the safe
direction when the only consequence is an earlier refusal. It is the *unsafe*
direction for a truncation budget: dividing bytes by three there converts a token
cap into a 25% smaller byte cap, so the model would quietly see about a quarter
less tool output per call. Measured while prototyping the single-constant version:
26 additional core tests failed, not because the code broke but because every
truncation output shrank — which is a behavior change dressed up as a math fix.

**Boundary, restated.** Exact accounting is unaffected either way: context totals
and compaction triggers use the embedded BPE tokenizer. The heuristics only cover
truncation budgets (4) and the no-tokenizer fallback plus input guards (3).

## v0.14 verification

- `cd backend && venv/bin/python -m pytest tests/` → 160 passed, including the
  new `tests/test_hallucinated_wrappers.py` (all five observed wrappers, nested
  braces, two calls in one message, prose that quotes a payload, malformed JSON).
- `cargo test -p codex-core --lib` → 2840 passed, 15 failed, and the 15 are
  byte-identical to the same suite run on an unmodified HEAD worktree (the
  pre-existing drift: missing code-mode host binary, guardian/model-catalog and
  reasoning-effort fixtures). One entry,
  `exec::tests::process_exec_tool_call_cancellation_allows_sigterm_cleanup`,
  passes when run alone and is parallel-run flaky.
- `cargo test -p codex-utils-string --lib` → 25 passed (including the new
  guard-vs-budget ratio test); `codex-utils-output-truncation` → 21 passed with
  every truncation literal untouched; `codex-models-manager` → 57 passed with the
  128,000 / 102,400 pins; `codex-protocol` → 369; `codex-cli` → 16;
  `codex-utils-cli` → 21; `codex-client` → 2; `codex-model-provider-info` → 40.
- Release build (81m 30s, exit 0) and `./scripts/install_mike.sh`; manual PTY
  smoke on the installed binary rendered `Ctx: 2,511 / 128,000` with
  `Token usage: total=2,511 input=2,509 output=2` and the `mike resume` hint.
  (The first attempt hung because the bridge's session token had aged past its
  6-hour limit and a dead Chrome had left `SingletonLock` behind in the CDP
  profile; clearing those three stale symlinks and restarting restored the
  headless refresh.)
- Why the split constant is worth it: with the single-constant version, core went
  to 41 failures and two of them were not stale numbers at all — the guardian
  interruption message no longer fit its own token budget, and truncated tool
  output exceeded `policy × 1.2` bytes — i.e. the model would have seen ~25% less
  output per call. Splitting the estimators returned every one of those.

## v0.15 D1. A mike thread has one DeepSeek chat per account

**Decision.** The bridge now identifies the signed-in DeepSeek account and keeps a
persisted ledger of who owns which upstream chat
(`backend/server/accounts.py`, git-ignored at `backend/session/accounts.json`):

- `deepseek/client.py` gained `account_id()`, a one-way 16-character fingerprint
  of the `users/current` user id. It is cached for the lifetime of a session and
  dropped by `_apply_session`, which is what separates "the token rotated" from
  "somebody else signed in".
- `resolve()` answers, for the incoming conversation id: resume it (this account
  owns it, or we have never seen it and claim it), resume a *bare* session id this
  account already uses for the thread, or open a new one and link it.
  `stamp()` records the owner and thread link of whatever session a turn landed
  on, and skips the disk write when nothing changed.
- The thread key is the core's `thread_id` from `client_metadata`, falling back to
  the first session id. `/v1/responses` (the Rust core) passes the thread id;
  `/v1/chat/completions` (`mike.py`, which persists the conversation id in its own
  session files) has no such field and uses the fallback.
- Re-homing returns a bare id on purpose, so the existing compaction-rewind path
  replays the whole transcript into the new account's empty thread.

**Rationale.** A `chat_session_id` belongs to exactly one account, and the core
stores the conversation id in memory only (`state/session.rs:85`), so resuming a
thread that another account created was an error the bridge passed straight
through. The core adopts whatever id the bridge returns
(`session/turn.rs:3220` `set_conversation_id`), so the whole feature is bridge
state plus one identity call — no Rust change, no wire-format change.

**Guard rails.**

- Identity is fail-open: a client without `account_id()`, a payload with no
  recognizable user id (logged as field *names* only), or a raising lookup leaves
  the conversation id untouched, i.e. exactly the pre-v0.15 behavior. There is a
  test per case.
- A request with no conversation id always opens a fresh upstream thread, even
  when the account already has a session for that thread. Clearing the id is how
  the core signals a history reset (`HistoryReplacement::Reset` →
  `conversation_id = None`, which is also what `mike resume` does), and reusing a
  remembered session would quietly undo a reset.
- `account_id()` takes an `RLock`, not a `Lock`: a lookup that meets an expired
  token refreshes the session from inside the lock, and `_apply_session` clears
  the same cache. A plain `Lock` deadlocked on exactly that path.
- The identity lookup is cached per session and failures are retried no more than
  every 300s, so `users/current` cannot put a GET in front of every turn.
- The ledger never holds a credential: fingerprints are hashed, and only account
  id *fields* are ever logged.

## v0.15 verification

- `cd backend && venv/bin/python -m pytest tests/` → 186 passed, 26 of them new in
  `tests/test_accounts.py` (fingerprint precedence, claim / resume / re-home /
  switch-back, restart survival, damaged and absent ledger files, no-disk-write
  fast path, both endpoints' `_response_events` behaviour, and the fail-open
  cases).
- End-to-end through the real ASGI stack with a fake account-switching client
  (scratch, not committed): five turns — new thread, same-account resume (23-char
  tail prompt), other account replaying the first account's id (966-char full
  transcript in a new session), first account replaying the second's id (back into
  its own session as a bare id), and a cid-less restart (fresh thread). The saved
  ledger held `{acctA: aaaa-1, acctB: bbbb-1}` under one `mike:thread-xyz` group.
- No Rust source changed, so the core suites are the v0.14 numbers unchanged.
- **Not yet verified live.** `users/current` was confirmed to exist and to answer
  `40003 Authorization Failed (invalid token)` for an expired token, but the CDP
  profile is signed out, so the response's field names have not been observed on
  a real account. If the payload carries no `id`/`user_id`/`uid`/`email`/
  `phone_number`/`username`, the bridge logs the field names it did see and stays
  in single-account mode.
