"""
Local DeepSeek bridge for Mike and other OpenAI-style clients.

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    POST /v1/responses          (Codex Responses API, streamed SSE)
    GET  /healthz

Point an OpenAI client at http://127.0.0.1:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
    r = client.chat.completions.create(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": "Hello!"}],
    )

The Responses endpoint is what the Rust Mike CLI uses: it translates Codex's
items/tools into a prompted DeepSeek call and parses DSML or JSON tool calls.

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import logging
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_ALIASES,
    MODEL_DISPLAY_NAME,
    MODEL_ID,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    canonicalize,
    is_known_model,
    resolve_model_type,
)
from .openai_format import (
    completion_response,
    error_frame,
    messages_to_prompt,
    stream_chunks,
)
from .ratelimit import RateLimiter, install_rate_limit
from .responses import build_responses_endpoint
from .schemas import ChatCompletionRequest

load_dotenv()

# Server-side operational log. Uses uvicorn's own error logger so these lines
# reach the console that already renders request logs (uvicorn does not attach a
# handler to the root logger, so a bespoke "deepseek.server" INFO would be lost).
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Mike DeepSeek Bridge", version="1.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


build_responses_endpoint(app, get_client)


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "name": MODEL_DISPLAY_NAME,
                # Non-standard extension: every id we accept and rewrite to `id`.
                "aliases": list(MODEL_ALIASES),
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. The only available model is "
            f"`{MODEL_ID}` ({MODEL_DISPLAY_NAME}); accepted aliases: "
            f"{', '.join(MODEL_ALIASES)}",
            status=404, err_type="model_not_found",
        )

    # Compatibility shim: rewrite any legacy id to the canonical one. Logged once
    # per alias per process (see server.config.canonicalize).
    canonical_id, _rewritten = canonicalize(req.model)

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(canonical_id)
    prompt = messages_to_prompt(req.messages)

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    if req.stream:
        def gen():
            # Catch ONLY exceptions raised by building the stream (client.stream
            # -> create_chat_session can 401) or by iterating it (the completion
            # POST). Anything else propagates. Turn an upstream failure into a
            # well-formed terminal SSE frame + [DONE] instead of crashing the ASGI
            # task, which truncates the response and yields a blank client reply.
            try:
                stream = client.stream(
                    prompt, conversation_id=req.conversation_id,
                    model=model_type, thinking=req.thinking, search=req.search,
                )
                yield from stream_chunks(canonical_id, stream, thinking=req.thinking)
            except Exception as exc:
                logger.exception("upstream streaming failed mid-response")
                yield error_frame(str(exc), err_type="upstream_error",
                                  model=canonical_id)
                yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, req.thinking, req.search,
        )
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    return completion_response(
        canonical_id, reply.text, prompt, reply.conversation_id,
        reasoning=getattr(reply, "reasoning", None) if req.thinking else None,
    )
