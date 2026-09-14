#!/usr/bin/env bash
# ClipForge launcher. Activates .venv, ensures ffmpeg, starts the web UI
# (or runs the daily job headless with --auto).
set -euo pipefail
cd "$(dirname "$0")"
VENV=".venv"
HOST="127.0.0.1"
PORT="8000"

if [ ! -d "$VENV" ] || [ ! -x "$VENV/bin/python" ]; then
  echo "No .venv found. Run ./bootstrap.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- resolve ffmpeg, persist to .env if bundled under ./tools ---------------
FF="$(command -v ffmpeg || true)"
if [ -z "$FF" ]; then
  CAND="$(ls tools/ffmpeg/bin/ffmpeg tools/ffmpeg-linux*/bin/ffmpeg tools/**/bin/ffmpeg 2>/dev/null | head -n1 || true)"
  if [ -n "$CAND" ]; then
    FF="$(cd "$(dirname "$CAND")" && pwd)/$(basename "$CAND")"
    chmod +x "$FF" 2>/dev/null || true
  fi
fi
if [ -z "$FF" ]; then
  echo "ffmpeg not found. Run ./bootstrap.sh or install ffmpeg, then retry." >&2
  exit 1
fi
# record into .env when it's a bundled tools/ path and not already set
case "$FF" in
  */tools/*)
    if ! grep -q '^FFMPEG_PATH=' .env 2>/dev/null; then
      echo "FFMPEG_PATH=$FF" >> .env
    fi ;;
esac

if [ "${1:-}" = "--auto" ]; then
  echo "[ClipForge] running daily job headless..."
  exec python -m app.main --auto
fi

# open browser after the server boots, then serve
( sleep 1.2; if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://$HOST:$PORT"; \
  elif command -v open >/dev/null 2>&1; then open "http://$HOST:$PORT"; fi ) &
echo "[ClipForge] starting http://$HOST:$PORT"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
