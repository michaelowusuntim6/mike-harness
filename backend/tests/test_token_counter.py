"""Tests for the embedded DeepSeek tokenizer counter (server/token_counter.py).

The parity vectors in assets/tokenizer_parity.json are asserted here AND by the
Rust codex-utils-tokenizer suite, which is what keeps the two implementations
producing identical numbers.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from server import responses, token_counter  # noqa: E402

FIXTURE = json.loads((ROOT / "assets" / "tokenizer_parity.json").read_text("utf-8"))
VECTORS = [(v["text"], v["tokens"]) for v in FIXTURE["vectors"]]


@pytest.fixture(autouse=True)
def _restore_counter():
    """Keep the fallback tests from leaking into the rest of the suite."""
    yield
    token_counter.reset_for_tests()


def test_expected_vectors():
    assert token_counter.implementation() == token_counter.VOCAB, (
        "tokenizer asset missing; run scripts/install_mike.sh"
    )
    for text, expected in VECTORS:
        assert token_counter.count_tokens(text) == expected, repr(text[:60])


def test_hello_world_is_four_tokens():
    assert token_counter.count_tokens("Hello, world!") == 4


def test_empty_text_counts_zero_not_one():
    assert token_counter.count_tokens("") == 0


def test_warm_reports_exact_and_describes_itself():
    assert token_counter.warm() is True
    assert token_counter.is_exact() is True
    assert token_counter.VOCAB in token_counter.describe()


def test_segments_match_the_joined_prompt_within_a_rounding():
    parts = ["## Tool protocol", "User: do the task", "Assistant: " + "x" * 2000]
    joined = token_counter.count_tokens("\n\n".join(parts))
    segmented = token_counter.count_segments(parts)
    # Join boundaries can hide a merge or two; the tolerance is what the
    # generator measured over real prompts (~0.3%), widened for small inputs.
    assert abs(segmented - joined) <= max(4, round(joined * 0.01))


def test_repeated_segments_are_served_from_cache():
    segment = "Line of tool output\n" * 400
    first = token_counter.count_segments([segment])
    assert token_counter.count_segments([segment] * 50) == first * 50


def test_missing_asset_falls_back_without_crashing(monkeypatch):
    monkeypatch.setenv("MIKE_TOKENIZER_PATH", str(ROOT / "does-not-exist.json"))
    token_counter.reset_for_tests()

    assert token_counter.implementation() == "heuristic"
    assert token_counter.is_exact() is False
    assert token_counter.count_tokens("Hello, world!") == len("Hello, world!") // 4
    assert token_counter.count_tokens("") == 0

    event = json.loads(
        responses._emit_completed("resp_1", "a prompt of some length", "an output", "")
        .removeprefix("data: ")
    )
    usage = event["response"]["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens"] >= 1
