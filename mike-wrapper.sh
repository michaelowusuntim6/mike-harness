#!/usr/bin/env bash
# Global launcher for Mike. Starts the local DeepSeek bridge if needed, then
# hands the terminal to the Rust TUI/CLI.
set -euo pipefail

MIKE_HOME="${MIKE_HOME:-$HOME/.mike}"
BRIDGE_DIR="${MIKE_BRIDGE_DIR:-$HOME/Deepseek-API}"
BRIDGE_HOST="${MIKE_BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${MIKE_BRIDGE_PORT:-8000}"
BIN="${MIKE_BIN:-$MIKE_HOME/bin/mike-core}"
export CODEX_HOME="$MIKE_HOME"
export MIKE_BASE_URL="${MIKE_BASE_URL:-http://$BRIDGE_HOST:$BRIDGE_PORT/v1}"

if [[ ! -x "$BIN" ]]; then
  echo "mike: binary not found at $BIN" >&2
  echo "mike: build it with: cargo build --release -p codex-cli --bin codex" >&2
  exit 127
fi

bridge_ready() {
  curl -fsS --max-time 2 "$MIKE_BASE_URL/models" 2>/dev/null \
    | grep -q '"deepseek-v4.1-flash"'
}

start_bridge() {
  if bridge_ready; then
    return 0
  fi

  if [[ -n "${MIKE_NO_BRIDGE:-}" && "${MIKE_NO_BRIDGE}" != "0" ]]; then
    return 0
  fi

  if [[ ! -f "$BRIDGE_DIR/app.py" ]]; then
    echo "mike: DeepSeek bridge not found at $BRIDGE_DIR/app.py" >&2
    echo "mike: set MIKE_BRIDGE_DIR to the Deepseek-API checkout" >&2
    exit 1
  fi

  local py="python3"
  if [[ -x "$BRIDGE_DIR/venv/bin/python" ]]; then
    py="$BRIDGE_DIR/venv/bin/python"
  fi

  mkdir -p "$MIKE_HOME/logs"
  local log="$MIKE_HOME/logs/bridge.log"
  if [[ -f "$MIKE_HOME/bridge.pid" ]]; then
    local old_pid
    old_pid="$(cat "$MIKE_HOME/bridge.pid" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && ps -p "$old_pid" -o args= 2>/dev/null | grep -q 'app.py'; then
      kill "$old_pid" 2>/dev/null || true
    fi
    rm -f "$MIKE_HOME/bridge.pid"
  fi

  (
    cd "$BRIDGE_DIR"
    HOST="$BRIDGE_HOST" PORT="$BRIDGE_PORT" \
      nohup "$py" app.py >>"$log" 2>&1 &
    echo $! >"$MIKE_HOME/bridge.pid"
  )

  for _ in $(seq 1 60); do
    if bridge_ready; then
      return 0
    fi
    if ! kill -0 "$(cat "$MIKE_HOME/bridge.pid" 2>/dev/null)" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done

  echo "mike: bridge failed to start; see $log" >&2
  tail -n 20 "$log" >&2 || true
  exit 1
}

case "${1:-}" in
  -h|--help|-V|--version|help|completion|login|logout)
    ;;
  *)
    start_bridge
    ;;
esac

exec "$BIN" "$@"
