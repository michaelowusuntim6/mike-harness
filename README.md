# mike — a local DeepSeek coding agent

`mike` is a standalone terminal coding agent built on the Rust Codex CLI core,
rebranded and rewired to run entirely against a **local DeepSeek bridge**. It
does not use an OpenAI account, OpenAI telemetry, ChatGPT login, or a paid API
key. The model is **DeepSeek V4.1 Flash**, served by the Python bridge in
[`~/Deepseek-API`](../Deepseek-API), which drives the free DeepSeek web session
through your own browser login.

This repository is a fork of [openai/codex](https://github.com/openai/codex)
(the Rust workspace is under `codex-rs/`). The upstream OpenAI provider code
still exists as an internal compatibility layer in some crates, but it is not
the default, not required, and its account/cloud/update surfaces are hidden or
inert in the Mike CLI.

> Unofficial project. Not affiliated with or endorsed by OpenAI or DeepSeek.
> Use it responsibly and within DeepSeek's terms of service.

---

## What works

- Interactive TUI (`mike`) with streaming responses, session history, resume,
  fork, archive, `/compact`, `/diff`, `/status`, approvals, and sandboxing.
- Non-interactive mode: `mike exec "..."`, `mike review`, `mike resume`.
- Agent tools exposed through the bridge:
  - `exec_command` / `write_stdin` — shell commands and interactive sessions
  - `apply_patch` — atomic, Codex-style multi-file patches
  - `update_plan` — visible multi-step plans
  - `task_complete` — explicit completion signal
- Prompted tool calling for the DeepSeek web backend, including:
  - strict `<tool_call>{JSON}</tool_call>` envelopes
  - DeepSeek native **DSML** `invoke`/`parameter` fallback (V4 and V4.1 tags)
  - multiple calls per turn, hidden tool syntax, and truncated-call recovery
  - one completion nudge if the model stops after a tool call without finishing
  - empty-stream retries and clear upstream error reporting
  - reasoning replay: when thinking is enabled, prior reasoning is emitted as a
    Responses reasoning item and folded back into subsequent prompts.
- DeepSeek-only model catalog, so the model picker cannot select OpenAI models.
- OpenAI login, ChatGPT account, Cloud, plugin marketplace, update checks,
  analytics, and feedback uploads are disabled/hidden by default.

## Architecture

```
┌───────────────────────────┐     Responses API (SSE)     ┌──────────────────────────┐
│  mike (Rust Codex core)   │ ──────────────────────────> │ Python bridge            │
│  TUI · tools · sandbox    │ <────────────────────────── │ 127.0.0.1:8000/v1        │
└───────────────────────────┘   text + function calls    │ prompted-tool adapter    │
                                                          └────────────┬─────────────┘
                                                                       │ single prompt
                                                          ┌────────────▼─────────────┐
                                                          │ DeepSeek web client       │
                                                          │ (your signed-in browser)  │
                                                          └──────────────────────────┘
```

The bridge is stateless from Codex's perspective: each Responses request carries
the full transcript, which the bridge flattens into a DeepSeek prompt. Tool
definitions supplied by Codex are rendered into a strict prompted-call protocol.
The model's tool calls are translated back into Responses `function_call` /
`custom_tool_call` items and executed by the Rust agent — so patches, shell
permissions, and sandboxing stay inside Codex.

## Requirements

- Linux/macOS with Rust 1.99.1 (`rustup` recommended)
- Python 3.10+ and the dependencies in `~/Deepseek-API/requirements.txt`
- A one-time DeepSeek web login for the bridge:
  ```bash
  cd ~/Deepseek-API
  source venv/bin/activate
  python -m deepseek.auth
  ```
  The bridge reuses that session and refreshes it headlessly.

## Build

```bash
cd ~/mike-harness/codex-rs
cargo build --release -p codex-cli --bin codex
```

The binary is `codex-rs/target/release/codex`. Install the global launcher:

```bash
mkdir -p ~/.mike/bin ~/.local/bin
cp codex-rs/target/release/codex ~/.mike/bin/mike-core
cp ~/mike-harness/mike-wrapper.sh ~/.local/bin/mike
cp ~/mike-harness/mike-config.toml ~/.mike/config.toml
cp ~/mike-harness/mike-models.json ~/.mike/mike-models.json
chmod +x ~/.local/bin/mike
```

`~/.local/bin` must be on `PATH`.

## Run

```bash
mike                    # start/attach the bridge, then launch the TUI
mike exec "fix the failing test"
mike review
mike resume
mike --help
```

The launcher:

1. points `CODEX_HOME` at `~/.mike` (isolated config/state);
2. checks `http://127.0.0.1:8000/v1/models`;
3. starts `~/Deepseek-API/app.py` with its venv if the bridge is not up;
4. executes the Mike binary.

The bridge keeps running between invocations. Stop it with:

```bash
kill "$(cat ~/.mike/bridge.pid)"
```

Set `MIKE_NO_BRIDGE=1` to skip auto-start (useful for tests or an external
bridge). `MIKE_BRIDGE_DIR`, `MIKE_BRIDGE_HOST`, `MIKE_BRIDGE_PORT`, and
`MIKE_BASE_URL` override paths/addresses.

## Configuration

Default config lives at `~/.mike/config.toml`:

- provider: `deepseek` → `http://127.0.0.1:8000/v1`
- bearer key: `sk-local` (the local bridge does not validate it)
- model: `deepseek-v4.1-flash`
- approval policy: `never`; sandbox: `workspace-write` (network enabled)
- analytics, feedback, update checks, web search: disabled
- only the DeepSeek model is present in the picker
- `~/.mike` projects are pre-trusted

Useful overrides:

| Variable | Effect |
| --- | --- |
| `MIKE_BASE_URL` | Point the Rust client at another Responses endpoint. |
| `MIKE_BRIDGE_THINKING=1` | Enable DeepSeek DeepThink reasoning. The bridge emits reasoning items and replays them on later turns. |
| `MIKE_BRIDGE_DIR` | Alternative `Deepseek-API` checkout. |
| `MIKE_NO_BRIDGE=1` | Do not auto-start the Python bridge. |

The bridge exposes:

- `POST /v1/responses` — Codex's Responses API stream
- `POST /v1/chat/completions` — the original OpenAI-compatible harness endpoint
- `GET /v1/models`, `GET /healthz`

## Bridge behavior details

- **Tool protocol:** Codex tools are rendered into a compact prompt; the model
  is taught the exact JSON envelope plus freeform `input` handling for
  `apply_patch`. Both JSON envelopes and DSML are parsed.
- **Nudging:** if a text-only reply arrives after a tool result, the bridge makes
  one internal follow-up request with a completion nudge. A second text-only
  reply is accepted so the loop cannot spin.
- **Completion:** `task_complete` is a bridge-only tool. Its summary becomes the
  final assistant message; other calls in the same turn are ignored.
- **Empty streams:** up to three upstream attempts, then a clear failure event.
- **Limits:** text-only model, no image input, no hosted web search, no OpenAI
  cloud features. The unused V8/code-mode host packages were removed from the
  workspace so the release build does not ship or build V8.

## Testing

```bash
# Rust unit tests for the changed crates
cd ~/mike-harness/codex-rs
cargo test -p codex-model-provider-info -p codex-models-manager \
  -p codex-core --lib -- --test-threads=4

# Bridge parser/protocol tests
cd ~/Deepseek-API
venv/bin/python -m pytest tests/test_responses_bridge.py -q

# End-to-end with a mocked model (no DeepSeek login needed)
# See scripts/mock_bridge.py and scripts/e2e_smoke.sh in this repo.
```

## Layout

- `codex-rs/` — Rust Codex workspace (TUI, core, providers, tools)
- `mike-config.toml` — installed to `~/.mike/config.toml`
- `mike-models.json` — DeepSeek-only model catalog
- `mike-wrapper.sh` — global launcher installed as `~/.local/bin/mike`
- `scripts/` — mock bridge and smoke-test helpers
- `../Deepseek-API/server/responses.py` — Responses API adapter
- `../Deepseek-API/server/toolcall.py` — prompted/DSML parsers

Upstream documentation and history remain available in the original repository:
<https://github.com/openai/codex>.
