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
