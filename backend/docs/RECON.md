# RECON — DeepSeek-API + sibling harnesses

Canonical location: `docs/RECON.md`.

Read-only reconnaissance performed before any modification. All findings below
were verified against the code in `~/Deepseek-API`, `~/nanocode`, and
`~/cipher-harness` (2026-10-07).

## 1. How the local server is started

| Item | Value |
| --- | --- |
| Entrypoint | `python app.py` (from the `Deepseek-API` root) |
| What it does | `uvicorn.run("server.api:app", ...)` |
| Host | `os.getenv("HOST", "127.0.0.1")` — default `127.0.0.1` |
| Port | `int(os.getenv("PORT", "8000"))` — default `8000` |
| Equivalent | `uvicorn server.api:app --host 127.0.0.1 --port 8000` |
| Runtime | the repo's `venv` (`venv/bin/python`) |
| App object | `server.api:app` (FastAPI, title `DeepSeek OpenAI-compatible API`, version `0.1.0`) |

`app.py` calls `load_dotenv()` before starting uvicorn. Importing
`server.api` builds the FastAPI app and installs rate-limit middleware.

Verified live: `venv/bin/python app.py` serves `http://127.0.0.1:8000`.

## 2. OpenAI-compatible routes

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Supports `stream: true`; non-OpenAI extras: `conversation_id`, `thinking`, `search` |
| `GET` | `/v1/models` | Static list built from `server.config.MODEL_MAP` |
| `GET` | `/healthz` | Rate-limit exempt, returns `{"status":"ok"}` |

No `/v1/completions`, no embeddings, no native `tools=` handling. Pydantic
`ChatCompletionRequest` ignores unknown fields, so passing `tools=` does **not**
error — it is silently dropped. That means the server is a **Prompted**-mode
backend, not a Native tool-call backend.

Verified live responses: `/healthz` → `{"status":"ok"}`; `/v1/models` → the two
model ids below; a real `POST /v1/chat/completions` returned `"PONG"` with a
`conversation_id`.

## 3. Exact model IDs currently advertised

`server/config.py::MODEL_MAP`:

```python
MODEL_MAP = {
    "deepseek-chat":   "default",   # Instant — fast default model
    "deepseek-expert": "expert",    # Expert  — stronger, slower model
}
DEFAULT_MODEL = "deepseek-chat"
```

So `/v1/models` currently returns `deepseek-chat` and `deepseek-expert`.
`model` maps to DeepSeek's wire `model_type` (`default` / `expert`).

## 4. Auth handling

- **Server auth:** none. `/v1/*` endpoints take no API key. `README.md` says the
  key "is required by the SDK but ignored". Any value works (we send `sk-local`).
- **Upstream DeepSeek auth:** a captured browser session is stored in
  `session/session.json` (`{"token", "cookies", "user_agent", "captured_at"}`)
  and a persistent Chromium profile in `session/profile/`. `deepseek.auth`
  refreshes the token headlessly from the profile (valid ~6h); if that fails it
  can open a visible login window. The server builds one shared
  `DeepSeekClient` lazily; `SERVER_INTERACTIVE_LOGIN=1` (default) lets the
  first request pop a browser if no session exists.
- Verified live: the cached session is currently valid (completion succeeded).

## 5. Where session state lives on disk

- DeepSeek login/session: `~/Deepseek-API/session/session.json` +
  `~/Deepseek-API/session/profile/` (git-ignored).
- Plus completion metadata is a `conversation_id` string
  (`<chat_session_id>:<message_id>`), stateless, returned per response.
- The cipher-harness logged sessions to `<cwd>/.cipher/sessions/*.md`;
  nanocode had **no** session persistence at all.
- `mike.py` will keep its own agent sessions in `~/Deepseek-API/sessions/`.

## 6. `compact.py` function signatures (to inline)

From `~/cipher-harness/compact.py`:

```python
CONTEXT_WINDOWS: dict[str, int]      # per-model context windows (Anthropic/local)
DEFAULT_LOCAL_WINDOW = 32_768
RESERVE_TOKENS = 4_000

def estimate_tokens(messages: list[dict]) -> int: ...
def should_compact(messages: list[dict], model: str) -> bool: ...
def compact_messages(client, messages: list[dict], model: str) -> list[dict]: ...
def _compact_local(client, prompt: str, model: str) -> str: ...
def _simple_summary(messages: list[dict]) -> str: ...
def _simple_summary_from_prompt(prompt: str) -> str: ...
def _serialize_messages(messages: list[dict]) -> str: ...
```

Logic to port into a `Compactor` class: estimate tokens (`len(text)//4`), trigger
at ~80% of the usable window, keep the last N turns verbatim (`compact_messages`
kept the last 4; the brief specifies 6), replace older turns with a single
summary message, and fall back to a non-LLM extraction summary if the
summarisation call fails.

## 7. Tool schema format used by the siblings

Both use a JSON-Schema `input_schema` on an object with `name` + `description`:

- **cipher-harness** — `Tool` ABC with `name`, `description`, `input_schema`
  properties and `to_api_dict()` → `{"name", "description", "input_schema"}`.
  `input_schema` is standard JSON Schema (`type: object`, `properties`,
  `required`). `call(input: dict) -> ToolResult(output, error)`.
- **nanocode** — a `TOOLS` dict of `name -> (description, {param: "type"}, fn)`
  where `"string?"` means optional and `number` means integer; `make_schema()`
  converts it to Anthropic `{name, description, input_schema}`.

For OpenAI-compatible `tools=` the shape is
`{"type":"function","function":{"name","description","parameters"}}`. `mike.py`
will define each tool once as a JSON-Schema dict and emit it either as OpenAI
`tools=` (Native mode) or as a described catalogue in the system prompt
(Prompted mode).

tools to port (brief §4.5): `read_file`, `write_file`, `edit_file`, `list_dir`,
`glob`, `grep`, `bash`, `python`, `web_fetch`, `todo_write`, `todo_read`.

## 8. Streaming / thinking observations

`server/openai_format.py::stream_chunks` emits OpenAI SSE `data:` frames with
`delta.content` only. `deepseek/client.py::_parse_sse` currently drops any
non-`RESPONSE` fragments (e.g. `THINKING`), so the bridge never forwards
reasoning today. With `thinking: true` upstream, reasoning text arrives inline
as ordinary `content`. `mike.py` therefore treats reasoning handling as
best-effort: it understands ` thinking`/`<reasoning>`/`reasoning_content`, and
`in_reasoning` state from the server if present.

## 9. Licenses (original headers recorded)

- `~/Deepseek-API/LICENSE` — MIT, "Copyright (c) 2026 !(Not Sumit)".
- `~/nanocode/README.md` — "## License\n\nMIT" (no LICENSE file, no in-file
  header; the source is `nanocode.py` under MIT).
- `~/cipher-harness/README.md` — "MIT. See [LICENSE](LICENSE)".

`mike.py` is derived from `nanocode.py` (MIT) and ports tool logic from
`cipher-harness` (MIT); both notices are preserved in the `mike.py` header.

## 10. Dependency inventory (venv, pre-existing)

`fastapi 0.142.4`, `starlette 1.7.0`, `uvicorn 0.54.0`, `httpx 0.28.1`,
`openai 3.26.0`, `playwright 1.63.0`, `pydantic 2.13.5`,
`python-dotenv 1.2.4`, `wasmtime 49.0.0`, `anyio 4.15.1`, `certifi`,
`click`, `h11`, `httpcore`, `jiter`, `sniffio`, `truststore`, `typing_extensions`.

Not installed at recon time: `rich`, `tiktoken`, `pytest`.
