#!/usr/bin/env python3
"""Regenerate assets/tokenizer_parity.json from the embedded DeepSeek tokenizer.

The vector file is what proves the Rust and Python counters agree: both suites
assert these exact numbers, so a tokenizer, asset or library-version change that
shifts a count fails a test instead of silently changing the user's `Ctx:`
number.  Re-run only after deliberately changing assets/tokenizer.json:

    backend/venv/bin/python scripts/gen_tokenizer_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from server.token_counter import VOCAB, asset_candidates, count_tokens  # noqa: E402

SPECIAL = "<｜begin▁of▁sentence｜>"
TAIL = "<｜end▁of▁sentence｜>"

VECTORS = [
    "",
    "hello",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def main():\n    print('hi')\n",
    "cd /home/mike/slnm && git status --short",
    "你好，世界",
    "▁leading▁underscores▁",
    SPECIAL,
    TAIL,
    f"{SPECIAL}user{TAIL}",
    '{"name": "exec_command", "arguments": {"cmd": "ls -la"}}',
    "SGVsbG8sIHdvcmxkIQo=" * 8,
    "👩‍👩‍👧‍👦 family 🚀 launch ✅",
    "Line of tool output\n" * 120,
    "User: do the task\n\nAssistant: ",
]


def main() -> int:
    asset = next((path for path in asset_candidates() if path.is_file()), None)
    if asset is None:
        print("tokenizer asset not found; run scripts/install_mike.sh", file=sys.stderr)
        return 1
    out = ROOT / "assets" / "tokenizer_parity.json"
    payload = {
        "vocab": VOCAB,
        "asset": str(asset.relative_to(ROOT)),
        "note": (
            "Token counts taken with add_special_tokens=False. Both the Rust "
            "(codex-utils-tokenizer) and Python (server.token_counter) suites "
            "assert these, so the two can never drift apart silently."
        ),
        "vectors": [{"text": text, "tokens": count_tokens(text)} for text in VECTORS],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(VECTORS)} vectors)")
    for vector in payload["vectors"]:
        preview = vector["text"].replace("\n", "\\n")[:48]
        print(f"  {vector['tokens']:>6}  {preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
