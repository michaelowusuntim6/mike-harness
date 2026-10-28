"""Server configuration: the single OpenAI-facing model and its aliases.

DeepSeek's free web chat now exposes exactly one model. The server therefore
advertises a single canonical id (``deepseek-v4.1-flash``, display name
``DeepSeek V4.1 Flash``) and transparently rewrites every legacy model id to it,
so old clients keep working without a 404 and without any silent fallback to a
different model (there is only one).
"""

import logging
import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# --- The one canonical model --------------------------------------------------
MODEL_ID = "deepseek-v4.1-flash"
MODEL_DISPLAY_NAME = "DeepSeek V4.1 Flash"

# DeepSeek's `model_type` wire value the canonical model maps to. The web chat's
# only free model runs DeepThink reasoning and web search as orthogonal toggles
# (`thinking` / `search` in the request body), never encoded in the model name.
DEFAULT_MODEL_TYPE = "default"

# Public model ids we ACCEPT. The canonical id is the only one advertised by
# /v1/models; the rest are compatibility aliases rewritten to it.
MODEL_ALIASES = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v3",
    "deepseek-v4",
    "deepseek-flash",
    "deepseek-web",
    "default",
]

# Every accepted id maps to the same single wire model.
MODEL_MAP = {name: DEFAULT_MODEL_TYPE for name in [MODEL_ID, *MODEL_ALIASES]}

DEFAULT_MODEL = MODEL_ID

# --- one-time alias-rewrite logging -------------------------------------------
_log = logging.getLogger("deepseek.server.model")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False

_rewritten_aliases: set = set()


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 genuinely unknown ids)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate an accepted public id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]


def canonicalize(name: str) -> tuple:
    """Rewrite a legacy alias to the canonical id.

    Returns ``(canonical_id, was_rewritten)``. The rewrite is logged at INFO level
    exactly once per alias per process so repeated traffic doesn't spam the log.
    """
    if name == MODEL_ID:
        return MODEL_ID, False
    if name in MODEL_ALIASES:
        if name not in _rewritten_aliases:
            _rewritten_aliases.add(name)
            _log.info("model alias %r rewritten to %r", name, MODEL_ID)
        return MODEL_ID, True
    return name, False
