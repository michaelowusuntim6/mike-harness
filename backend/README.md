# DeepSeek-API — a free LLM API + the `mike` coding harness

**Using your own DeepSeek account.** No API key, no credits, no paid plan: this
turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an
OpenAI-compatible API you can call from code — and ships **`mike.py`**, a
self-contained coding-agent CLI that runs on top of it.

Two things live here:

- 🐍 **The library + server** — `deepseek/` (a pure-HTTP client) and `server/`
  (a FastAPI app at `http://127.0.0.1:8000/v1` that speaks the OpenAI format).
- 🤖 **`mike.py`** — one file, no sibling dependencies, that auto-launches the
  server and gives you an interactive coding agent with tools, thinking-block
  handling, inline compaction and resumable sessions.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It
> automates the consumer DeepSeek web experience for personal use, so use it
> responsibly and within DeepSeek's terms.

---

## The single-model story

DeepSeek's free web chat now exposes **exactly one model**. Everywhere in this
project it is surfaced as:

| | |
| --- | --- |
| **Display name** | `DeepSeek V4.1 Flash` |
| **Wire id** | `deepseek-v4.1-flash` |

That is the *only* model you will ever see, in `mike.py` or from `/v1/models`.
For compatibility, these legacy aliases are still accepted and silently routed
to it (no warning beyond one info line at startup, and a single INFO log line on
the server per alias per process):

```
deepseek-chat   deepseek-reasoner   deepseek-v3      deepseek-v4
deepseek-v4.1-flash   deepseek-flash   deepseek-web   default
```

Think of **thinking** (DeepThink reasoning) and **web search** as toggles, not
models — they are requested per call and never encoded in the model name.

---

## Install

```bash
git clone https://github.com/sums001/Deepseek-API.git
cd Deepseek-API

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt     # playwright is used only as a CDP client here

python -m deepseek.auth             # sign in once (opens your Chrome at DeepSeek)
```

**How sign-in works.** `python -m deepseek.auth` (and the server's on-demand login)
launch your **installed Chrome as a normal browser process** pointed at DeepSeek,
then attach Playwright as a *passive debugger* over the Chrome DevTools Protocol
(`connect_over_cdp`). Google's DeepSeek sign-in flow blocks browsers that were
*started by automation* (it fingerprints `navigator.webdriver` and the Playwright /
bundled-Chromium launch), showing *"Couldn't sign you in. This browser or app may not
be secure."* Because this Chrome is launched by an ordinary `subprocess` with only
`--remote-debugging-port` and a dedicated `--user-data-dir`, none of those signals are
present — `navigator.webdriver` is `false` and there is no *"unsupported command-line
flag"* banner.

Because Chrome 136+ ignores `--remote-debugging-port` on the **default** profile, the
harness drives a **dedicated** profile at `session/chrome-cdp-profile/`. The very first
run opens Chrome with an empty profile (no Google account, no DeepSeek session); you
sign in once. That profile then persists, so later runs (including the server's headless
refresh) capture the token automatically with no prompt.

- Sign in inside that dedicated profile. You can stay logged into DeepSeek in your
  regular Chrome too — DeepSeek allows multiple sessions — but the harness only reads
  the dedicated `chrome-cdp-profile/`.
- `DEEPSEEK_CDP_PROFILE=/path/to/profile` overrides the dedicated profile directory.

Your session (bearer token + cookies) is saved under `session/` and refreshed
automatically afterwards. The token value is never printed — only its length and the
cookie count.

---

## Run the harness: `mike.py`

```bash
python mike.py            # start (or attach to) the server, then chat
python mike.py resume     # pick a saved session from a table, then continue
python mike.py config     # print the resolved config (no server)
python mike.py export X   # write session X to Markdown (no REPL)
python mike.py --help     # usage (does not start the server)
```

`mike.py` is a single self-contained file. On startup it:

1. probes `127.0.0.1:8000` and reuses a server that is already running
   (it never starts a second one);
2. otherwise launches `python app.py` from this directory and polls
   `/v1/models` until it answers (30s timeout, 250 ms backoff);
3. registers exit/signal handlers that stop the child **only if mike started it**;
4. detects whether the backend supports native OpenAI `tools=` (it probes with a
   dummy `ping` tool) and caches the answer in `.mike_state.json`.

Everything is local: the only network calls are to `127.0.0.1` and, when a tool
asks for it, to URLs you fetch.

### Slash commands

```
/help                 show this help
/quit, /exit          quit (saves the session)
/clear                clear the screen
/model [name]         show the model (there is exactly one)
/thinking on|off|last thinking display: on / off / show last block
/sessions             list saved sessions
/resume <n|name>      switch to a saved session
/rename <name>        rename the current session
/export [--force] [path]  write this session to a Markdown transcript
/compact              force compaction now
/tokens               show the current token estimate + threshold
/tools                list the available tools
/rollover             force a session rollover
```

`/export` writes the current session as Markdown (see
[Export](#export-a-session)). `--force` allows overwriting an existing file;
without it you get a warning and a `Overwrite? [y/N]` prompt.

### Tools

`read_file`, `apply_patch`, `list_dir`, `glob`, `grep`, `bash`, `python`,
`web_fetch`, `update_plan`, `todo_read`, `task_complete`.

`apply_patch` is the only file-mutation surface. It accepts the line-oriented
Codex patch envelope:

```
*** Begin Patch
*** Add File: notes.txt
+hello
*** Update File: app.py
@@
-old
+new
*** Delete File: obsolete.txt
*** End Patch
```

`*** Move to:` after an update renames the file, and `*** End of File` supports
an EOF append. Mike parses and validates the entire patch against an in-memory
view before writing, so a bad context line cannot leave an earlier operation
partially applied. Successful patches render as syntax-highlighted unified
diffs with file headers and added/removed counts.

For multi-step work, the model calls `update_plan` with structured
`step`/`status` items. The terminal shows a checklist using `✓` for completed
steps, `⠋` for the active step, and `○` for pending steps. `todo_read` returns
the current plan.

The server's web backend does not implement OpenAI's `tools=` schema, so mike
runs in **prompted mode**: it describes the tools in a system prompt and parses
the model's tool calls out of the reply. If a backend ever does return native
`message.tool_calls`, mike detects it at startup and uses that path instead.

**Two tool-call syntaxes are supported** (see
[Tool calls](#tool-calls) below): the strict
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` JSON envelope *and*
DeepSeek's native DSML syntax (`invoke`/`parameter` tags), parsed by the
**reference** DSML parser (both V4 and V4.1 tag forms, full-width bar U+FF5C).
Tool results are paired back to their call id so multi-step tasks continue.
Both may even appear in the same message, and multiple calls per turn are
dispatched in order.

### Completion and commentary

A text-only reply is treated as commentary, not completion. Before the first
tool call the model is asked to send a short progress update. If it replies
without calling `task_complete`, mike appends:

> You have not called `task_complete`. Continue the task or call
> `task_complete` with a summary.

The model then gets one more chance to keep working or finish. A second
text-only reply is accepted as the final fallback, so the loop cannot run
forever. A structured `task_complete` call ends the turn immediately and its
`summary` becomes the returned final answer.

The system prompt asks for a short natural-language progress update before
tool calls, without channel labels. Mike defensively strips literal
`[commentary]` and `[final]` tags if the model hallucinates them. The
enforceable completion signal is the `task_complete` tool call.

### Codex-style composer and status

The interactive prompt uses `prompt_toolkit` with persistent `.mike_history`,
Up/Down history navigation, and a floating slash-command menu. Pressing `/`
opens command completion; commands such as `/thinking`, `/resume`, and `/export`
also complete their arguments.

Enter submits a prompt. Use `Esc` then `Enter`, or `Ctrl+Enter` (encoded as
`Ctrl+J` in most terminals), to insert a newline for pasted code or multi-line
instructions. A status row before each prompt reports the model, context usage,
plan progress, and working directory. While a request runs, the transient
status line combines an animated spinner with current context usage.

### Thinking blocks

DeepSeek streams reasoning (fragment type `THINK`). By default mike shows it in a
dimmed single-line spinner and never dumps it to scrollback; `/thinking on`
streams the full reasoning in dim text, `/thinking off` hides it, and
`/thinking last` prints the most recent full reasoning block once. Reasoning is
stored in the session under a separate `reasoning` field.

**Reasoning replay:** on later requests, mike replays that field because
DeepSeek requires prior thinking-mode reasoning when tools are present. Prompted
mode prepends it as `<reasoning>…</reasoning>` to the outgoing assistant copy;
native tools mode sends it as the top-level `reasoning_content` field. The
stored assistant content remains clean, and the server schema accepts
`reasoning_content` when flattening history into the web prompt.

Completed assistant messages are rendered as Markdown with a Monokai code
theme. With `/thinking on`, the dimmed reasoning block appears above the answer,
separated by a subtle horizontal rule.

### Sessions

Sessions live in `sessions/` as pretty-printed, sorted-key JSON named
`YYYYMMDD-HHMMSS__<slug>.json`, where the slug is derived from your first
message (or set with `/rename`). Each file holds the **full** history, the
compaction log, and the model name. Compaction shrinks only the *in-context*
view; the on-disk transcript is never truncated. When a session is still over
budget after compaction, mike transparently **rolls over** to a new
`<slug>-contN` session seeded with the summary.

Compaction summaries begin with “The conversation was compacted…” and the
system prompt marks them as continuation context. The model is told not to
treat the summary as a new user request, restart the task, or repeat completed
work.

Each session is also pinned to **one upstream DeepSeek chat**: mike stores the
server's `conversation_id` and reuses it for every follow-up request, so tool
results and later prompts continue the same chat instead of spawning a new one
per request. After the first turn, only the new messages are sent (the thread
already holds the rest).

### Changing the context budget

The max context is **500,000 tokens** by default, and auto-compaction triggers at
**80%** of it (i.e. **400,000 tokens**). Override the max with an environment
variable or a config file — the threshold follows automatically:

```bash
MIKE_MAX_CONTEXT=200000 python mike.py     # => compacts at 160,000
python mike.py --max-context 200000        # same, via a flag
```

Token counting uses `tiktoken` when available and falls back to a `len//4`
heuristic; the choice is persisted in `.mike_state.json`.

---

## Configuration

mike reads layered TOML config. **Later layers win:**

| # | Layer | Shown as source |
|-|-------|-----------------|
| 1 | built-in `DEFAULTS` in `mike.py` | `default` |
| 2 | `~/.config/mike/config.toml` (user-wide) | `user` |
| 3 | `./mike.toml` (repo-local) | `repo` |
| 4 | `MIKE_*` environment variables | `env` |
| 5 | CLI flags | `cli` |

Merging is per-key (`dict.update`), not a deep merge. `mike.toml.example` at the
repo root mirrors the schema; copy it to `./mike.toml` or
`~/.config/mike/config.toml`.

**Keys** (all optional):

| Key | Default | Meaning |
| --- | --- | --- |
| `max_context` | `500000` | total tokens before compaction is possible |
| `compact_threshold` | `0.80` | fraction of `max_context` that triggers compaction |
| `keep_recent_turns` | `6` | turns preserved verbatim across a compaction |
| `server_host` | `"127.0.0.1"` | server bind host (probe target) |
| `server_port` | `8000` | server port |
| `server_cmd` | `""` | command to launch the server (empty ⇒ auto-detect `python app.py`) |
| `model` | `"DeepSeek V4.1 Flash"` | display name |
| `model_id` | `"deepseek-v4.1-flash"` | wire id sent to the server |
| `thinking_default` | `"off"` | `off` \| `on` \| `spinner` \| `last` |
| `tool_call_mode` | `"auto"` | `auto` \| `native` \| `prompted` |
| `session_dir` | `""` | empty ⇒ `<repo>/sessions/` |
| `export_dir` | `""` | empty ⇒ `<repo>/exports/` |
| `log_level` | `"info"` | `debug` \| `info` \| `warn` \| `error` |

Every key also has an environment variable (`MIKE_MAX_CONTEXT`,
`MIKE_COMPACT_THRESHOLD`, `MIKE_KEEP_RECENT_TURNS`, `MIKE_SERVER_HOST`,
`MIKE_SERVER_PORT`, `MIKE_MODEL`, `MIKE_THINKING_DEFAULT`,
`MIKE_TOOL_CALL_MODE`, `MIKE_SESSION_DIR`, `MIKE_EXPORT_DIR`, `MIKE_LOG_LEVEL`)
and a CLI flag (`--max-context`, `--compact-threshold`, `--keep-recent-turns`,
`--host`, `--port`, `--server-cmd`, `--model`, `--thinking`,
`--tool-call-mode`, `--session-dir`, `--export-dir`, `--log-level`). Flags and
subcommands compose: `python mike.py --max-context 200000 resume` works.
`HOST`/`PORT`/`MIKE_PORT` are still honored as fallbacks for the server address.

To see exactly what won, ask mike:

```bash
python mike.py config

key               value               source
max_context       250000              env
compact_threshold 0.8                 default
…
```

`config` never starts the server and is safe to run anywhere.

---

## Export a session

`/export [--force] [path]` writes the current session as a single Markdown
transcript. With no path it goes to
`<export_dir>/<session-name>__<YYYYMMDD-HHMMSS>.md` (`export_dir` defaults to
`./exports/`, created on demand). A path ending in `/` or naming an existing
directory means "use the default filename inside it"; relative paths resolve
against the current directory.

Any session can be exported non-interactively, without entering the REPL:

```bash
python mike.py export <session-name-or-id> [path]
python mike.py --force export my-session ./my-session.md
```

The transcript has this structure:

```markdown
# Session: <name>

- **id:** <uuid>
- **created:** <iso8601>
- **updated:** <iso8601>
- **model:** DeepSeek V4.1 Flash
- **messages:** <n>
- **compactions:** <n>
- **rollover-of:** <parent-name or —>

---

## Turn 1 — user

<user text, unmodified>

## Turn 1 — assistant

<assistant visible answer>

<details><summary>reasoning</summary>

<reasoning text, verbatim, if any>

</details>

**Tool calls:**

- `read_file({"path": "..."})`
  - result: <first 500 chars, … if truncated>

---
```

Reasoning is wrapped in a collapsible `<details>` block so it stays out of the
way on GitHub. User text is preserved byte-for-byte (only a leading `---` line is
escaped). The file ends with a newline.

---

## Tool calls

mike parses **two** tool-call syntaxes, in the same message if needed:

1. the strict JSON envelope:

   ```
   <tool_call>
   {"name": "<tool_name>", "arguments": { ... }}
   </tool_call>
   ```

2. DeepSeek's native DSML syntax — a `calls` container holding one or more
   `invoke` tags (naming the tool), each containing `parameter` tags for the
   arguments. This path now uses the **reference parser** (adapted from
   DeepSeek's `encoding.py`) and accepts **both** tag shapes, keyed off the real
   `｜DSML｜` token (the full-width bar U+FF5C, never ASCII `|`):
   - **V4** — no space after the token: `<｜DSML｜tool_calls>`, `<｜DSML｜invoke>`
   - **V4.1** — leading space: `<｜DSML｜ calls>`, `<｜DSML｜ invoke>`

Covered edge cases: **multiple** calls per message (dispatched in order),
**self-closing** invokes (`invoke name="list_dir" /` → dispatched with `{}`),
**attribute-style** arguments on a self-closing invoke
(`invoke name="read_file" path="x.py" /` → `{"path": "x.py"}`),
**truncated/unclosed** invokes (the partial call is still dispatched, with a
`tool_call_truncated` warning logged), and **malformed parameter JSON** (kept as
a raw string rather than dropping the call). Tool syntax is always hidden from
your scrollback; only the clean tool header and a short result preview are shown.

**Tool results are paired to their call.** Each tool result is fed back in a
single following user message as `<tool_result name="…" id="call_…">…</tool_result>`
blocks ordered to match the assistant's `tool_calls`, mirroring DeepSeek's
reference `merge_tool_messages` / `sort_tool_results_by_call_order`. Because the
model can pair each result with the call that produced it, it continues multi-step
tasks instead of stopping after the first tool.

## The OpenAI-compatible server

Start it directly (mike does this for you):

```bash
python app.py       # -> http://127.0.0.1:8000
```

Then point any OpenAI client at it (the key is required by the SDK but ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
resp = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"thinking": True, "search": False},   # DeepThink / web search
)
print(resp.choices[0].message.content)
```

**Endpoints**

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat. Supports `stream: true`; non-OpenAI extras: `conversation_id`, `thinking`, `search` |
| `POST` | `/v1/responses` | Responses API stream used by the Rust `mike` CLI. Renders Codex tools into the prompted tool protocol and returns `function_call` / `custom_tool_call` items |
| `GET`  | `/v1/models` | Advertises the one model, with `aliases: [...]` |
| `GET`  | `/healthz` | Health check (rate-limit exempt) |

The `/v1/responses` bridge is stateless: the request carries the full transcript,
which is flattened into one DeepSeek prompt. Tool calls are accepted either as
`<tool_call>{...}</tool_call>` JSON or DeepSeek native DSML, and `task_complete`
is treated as the explicit end-of-turn signal.

`/v1/models` returns exactly one entry:

```json
{"id": "deepseek-v4.1-flash", "object": "model", "name": "DeepSeek V4.1 Flash",
 "aliases": ["deepseek-chat", "deepseek-reasoner", "deepseek-v3", "deepseek-v4",
             "deepseek-flash", "deepseek-web", "default"]}
```

Requests naming any alias are rewritten to the canonical id and logged once per
alias per process. Genuinely unknown names still return `404` (no silent
fallback — there is only one model anyway).

Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run
`uvicorn server.api:app --host 0.0.0.0 --port 8080`.

---

## Usage as a Python library (no server)

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # loads your signed-in session
reply = client.chat("Say hello in one short sentence.")
print(reply.text, "|", reply.reasoning)

reply2 = client.chat("And now in French?", conversation_id=reply.conversation_id)
print(reply2.text)

for chunk in client.stream("Tell me a short joke"):
    print(chunk, end="", flush=True)
```

See [examples/](examples/) for runnable versions.

---

## Concurrency, rate limiting, notes

- The server bridges a **single** signed-in account and serializes upstream
  calls (the PoW solver is not reentrant). Throughput is sequential.
- A self-imposed sliding-window rate limit applies per client IP
  (`RATE_LIMIT_PER_MINUTE`, default `30`); `/healthz` is exempt. Use exponential
  backoff on `429`.
- No real token counts: `usage` is a rough ~4-chars/token estimate.
- Most OpenAI params (`temperature`, `top_p`, `max_tokens`) are accepted but
  ignored; only `model`, `messages`, `stream`, `conversation_id`, `thinking` and
  `search` do anything.
- Everything in `session/`, `sessions/` and `.mike_state.json` is local and
  git-ignored.

## Project layout

| Path | What it does |
| --- | --- |
| `mike.py` | The self-contained coding harness (this is the deliverable) |
| `deepseek/` | Core client: auth, HTTP driver, PoW solver |
| `server/` | FastAPI OpenAI-compatible server |
| `tests/test_mike.py` | Hermetic pytest suite for mike |
| [`docs/RECON.md`](docs/RECON.md) | Reconnaissance notes |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Design decisions and verification history |
| `app.py` | Starts the server |

## License

MIT. `mike.py` is derived from `nanocode.py` (MIT) and ports tool/compaction
logic from `cipher-harness` (MIT); those notices are preserved in its header. As
this is an unofficial project, you remain responsible for complying with
DeepSeek's terms of service.
