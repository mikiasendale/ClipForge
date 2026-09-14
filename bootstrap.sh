#!/usr/bin/env bash
# ClipForge bootstrap (Linux/macOS). Installs python3.12 + ffmpeg if missing,
# creates an isolated .venv, installs pinned deps, fetches the Scrapling browser,
# and seeds .env. NEVER installs Python packages into system Python.
# All failures print exact manual instructions and exit non-zero.
set -euo pipefail

cd "$(dirname "$0")"
PY="python3.12"
VENV=".venv"
OS="$(uname -s)"

log(){ printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
err(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# --- privilege helper -------------------------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo";
  else log "not root and no sudo; package installs may fail — manual steps below if so"; fi
fi

have(){ command -v "$1" >/dev/null 2>&1; }

# --- package manager detection ----------------------------------------------
detect_pm(){
  if have apt-get; then echo apt; elif have dnf; then echo dnf;
  elif have pacman; then echo pacman; elif have brew; then echo brew; else echo none; fi
}
PM="$(detect_pm)"

# --- ffmpeg -----------------------------------------------------------------
install_ffmpeg(){
  log "ffmpeg not found — installing via $PM"
  case "$PM" in
    apt)    $SUDO apt-get update && $SUDO apt-get install -y ffmpeg ;;
    dnf)    $SUDO dnf install -y ffmpeg || $SUDO dnf install -y ffmpeg-free ;;
    pacman) $SUDO pacman -Sy --noconfirm ffmpeg ;;
    brew)   brew install ffmpeg ;;
    *)      err "No supported package manager found."; \
            err "Install ffmpeg manually: https://ffmpeg.org/download.html"; return 1 ;;
  esac
}

if have ffmpeg && have ffprobe; then
  log "ffmpeg present: $(command -v ffmpeg)"
else
  install_ffmpeg || { log "trying bundled static build fallback into ./tools"; mkdir -p tools; \
      err "Automatic ffmpeg install failed. See README for manual tools/ffmpeg install."; exit 1; }
fi

# --- python 3.12 ------------------------------------------------------------
ensure_python(){
  if have "$PY"; then return 0; fi
  log "python3.12 not found — installing via $PM"
  case "$PM" in
    apt)    $SUDO apt-get update
            $SUDO apt-get install -y software-properties-common
            $SUDO add-apt-repository -y ppa:deadsnakes/ppa || true
            $SUDO apt-get update
            $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev ;;
    dnf)    $SUDO dnf install -y python3.12 python3.12-pip ;;
    pacman) $SUDO pacman -Sy --noconfirm python ;;   # Arch tracks latest 3.x
    brew)   brew install python@3.12 ;;
    *)      return 1 ;;
  esac
}

if ! ensure_python || ! have "$PY"; then
  err "python3.12 is required but could not be installed automatically."
  err "Install it manually, then re-run bootstrap:"
  err "  https://www.python.org/downloads/release/python-3120/"
  exit 1
fi
log "python: $("$PY" --version 2>&1)"

# --- venv + deps ------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  log "creating virtualenv in ./$VENV"
  "$PY" -m venv "$VENV"
fi
log "installing pinned requirements"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r requirements.txt

# --- scrapling browser (best-effort) ----------------------------------------
log "fetching Scrapling stealth browser (best-effort)"
"$VENV/bin/scrapling" install || err "scrapling install failed — enrichment will degrade; run manually later: .venv/bin/scrapling install"

# --- .env -------------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  log "created .env (add your OPENROUTER_API_KEY)"
fi

log "Done — run ./run.sh to start the web UI"
