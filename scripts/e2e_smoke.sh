#!/usr/bin/env bash
# End-to-end smoke test: real Mike binary + real bridge protocol renderer,
# deterministic mock model. No DeepSeek login required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/codex-rs/target/release/codex}"
WORK="$(mktemp -d)"
HOME_DIR="$WORK/home"
mkdir -p "$HOME_DIR"

cleanup() {
  if [[ -n "${MOCK_PID:-}" ]]; then
    kill "$MOCK_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

if [[ ! -x "$BIN" ]]; then
  echo "e2e: binary not found: $BIN" >&2
  echo "e2e: build with: cd codex-rs && cargo build --release -p codex-cli --bin codex" >&2
  exit 127
fi

python3 "$ROOT/scripts/mock_bridge.py" >"$WORK/mock.log" 2>&1 &
MOCK_PID=$!
for _ in $(seq 1 40); do
  if curl -fsS --max-time 1 http://127.0.0.1:8123/v1/models >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
curl -fsS --max-time 2 http://127.0.0.1:8123/v1/models >/dev/null

cat >"$HOME_DIR/config.toml" <<TOML
model = "deepseek-v4.1-flash"
model_provider = "mike_test"
model_catalog_json = "$ROOT/mike-models.json"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
check_for_update_on_startup = false

[analytics]
enabled = false

[feedback]
enabled = false

[skills]
include_instructions = false

[agents]
enabled = false

[tools]
experimental_request_user_input = { enabled = false }
update_plan = { enabled = true }

[features]
goals = false
view_image = false
image_generation = false
multi_agent = false
multi_agent_v2 = false
apps = false
plugins = false
tool_search = false
guardian_approval = false

[model_providers.mike_test]
name = "Mike Mock"
base_url = "http://127.0.0.1:8123/v1"
wire_api = "responses"
requires_openai_auth = false
experimental_bearer_token = "sk-local"
TOML

mkdir -p "$WORK/project"
cd "$WORK/project"
CODEX_HOME="$HOME_DIR" timeout 120 "$BIN" exec \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  "create and verify the mike e2e file" >"$WORK/run.log" 2>&1 || {
    echo "e2e: mike exec failed" >&2
    tail -n 60 "$WORK/run.log" >&2
    tail -n 30 "$WORK/mock.log" >&2
    exit 1
  }

expected="hello from mike e2e"
actual="$(cat mike_e2e.txt 2>/dev/null || true)"
if [[ "$actual" != "$expected" ]]; then
  echo "e2e: unexpected file contents: '$actual'" >&2
  tail -n 60 "$WORK/run.log" >&2
  exit 1
fi

grep -q "Done: created mike_e2e.txt" "$WORK/run.log" || {
  echo "e2e: final assistant message missing" >&2
  tail -n 60 "$WORK/run.log" >&2
  exit 1
}

echo "e2e: PASS — apply_patch, exec_command, and completion all work"
