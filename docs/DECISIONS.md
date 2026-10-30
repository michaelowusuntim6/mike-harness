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

## v0.11 verification

- `cd backend && venv/bin/python -m pytest tests/` → 129 passed (defaults,
  compaction threshold, and status-bar assertions moved to 120,000/96,000).
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
