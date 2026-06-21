#!/usr/bin/env bash
# Install GrainStore as a standalone server + CLI, optionally as a background
# service. Run from the grainstore-server crate root: ./deploy/install.sh
set -euo pipefail

cd "$(dirname "$0")/.."
echo "==> building release binaries"
cargo build --release

BIN="${GS_BIN:-$HOME/.local/bin}"
DATA="${GS_DATA:-$HOME/.grainstore}"
mkdir -p "$BIN" "$DATA"

echo "==> installing binaries to $BIN"
cp target/release/grainstored target/release/grainstore target/release/grainstore-mcp "$BIN/"

echo
echo "Installed: grainstored (server), grainstore (CLI), grainstore-mcp (agent MCP)"
echo "Data dir : $DATA"
case ":$PATH:" in
  *":$BIN:"*) : ;;
  *) echo "NOTE: add to PATH →  export PATH=\"$BIN:\$PATH\"" ;;
esac

echo
read -r -p "Install grainstored as a launchd service (auto-start at login)? [y/N] " yn
if [[ "${yn:-N}" =~ ^[Yy]$ ]]; then
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "Not macOS — use deploy/grainstore.service with systemd instead:"
    echo "  sudo cp deploy/grainstore.service /etc/systemd/system/ && sudo systemctl enable --now grainstore"
    exit 0
  fi
  PLIST_DIR="$HOME/Library/LaunchAgents"
  PLIST="$PLIST_DIR/com.grainstore.grainstored.plist"
  mkdir -p "$PLIST_DIR"
  sed "s#__HOME__#$HOME#g" deploy/com.grainstore.grainstored.plist > "$PLIST"
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  sleep 1
  echo "Service loaded. It is now running and will restart at login."
  echo "  logs   : $DATA/grainstored.log"
  echo "  stop   : launchctl unload $PLIST"
  echo "  start  : launchctl load $PLIST"
  echo "  health : grainstore health"
fi
