#!/usr/bin/env bash
# Build and install mike as a self-contained application.
#
# Layout after install:
#   ~/.local/bin/mike                 Rust CLI (spawns and reaps the backend)
#   ~/.local/lib/mike/backend/        bundled DeepSeek bridge + venv + session
#   ~/.mike/config.toml               runtime configuration
#
# Override the prefix with PREFIX=/somewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
LIB_DIR="$PREFIX/lib/mike"
BACKEND_SRC="$ROOT/backend"
BACKEND_DST="$LIB_DIR/backend"

if [[ ! -f "$BACKEND_SRC/app.py" ]]; then
  echo "install: backend not found at $BACKEND_SRC" >&2
  exit 1
fi

if [[ ! -x "$BACKEND_SRC/venv/bin/python" ]]; then
  echo "install: creating backend virtualenv..."
  python3 -m venv "$BACKEND_SRC/venv"
  "$BACKEND_SRC/venv/bin/python" -m pip install -r "$BACKEND_SRC/requirements.txt"
fi

echo "install: building mike..."
cargo build --release -p codex-cli --bin codex --manifest-path "$ROOT/codex-rs/Cargo.toml"

install -d "$BIN_DIR" "$LIB_DIR" "$HOME/.mike"
install -m 0755 -s "$ROOT/codex-rs/target/release/codex" "$BIN_DIR/mike"

echo "install: copying backend..."
rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "$BACKEND_SRC/" "$BACKEND_DST/"

cp -n "$ROOT/mike-config.toml" "$HOME/.mike/config.toml" 2>/dev/null || true
cp -n "$ROOT/mike-models.json" "$HOME/.mike/mike-models.json" 2>/dev/null || true

echo "install: done"
echo "  binary:  $BIN_DIR/mike"
echo "  backend: $BACKEND_DST"
echo "  config:  $HOME/.mike/config.toml"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "note: add $BIN_DIR to PATH" ;;
esac
