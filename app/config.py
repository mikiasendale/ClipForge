"""ClipForge configuration loader.

Reads ``config.yaml`` (user-editable, committed defaults) and ``.env``
(secrets, gitignored). Exposes a validated, cached ``Config`` object and
helpers to resolve the ffmpeg/ffprobe binaries across Windows and Linux.

Nothing here installs into system Python or mutates anything outside the
project tree.
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any

import yaml

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = DATA_DIR / "logs"
FRAMES_DIR = DATA_DIR / "frames"
DB_PATH = DATA_DIR / "clipforge.db"
TOOLS_DIR = PROJECT_ROOT / "tools"

_DIRS = [DATA_DIR, DOWNLOADS_DIR, OUTPUT_DIR, LOGS_DIR, FRAMES_DIR]


def ensure_dirs() -> None:
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --- Minimal .env parser (no dependency) ------------------------------------
def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines from .env into os.environ (does not override existing)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _deep_get(data: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


class Config:
    """Read-only-ish view over config.yaml with dotted access + resolution."""

    def __init__(self, raw: dict[str, Any]):
        self._raw = raw or {}
        self._ffmpeg: str | None = None
        self._ffprobe: str | None = None

    # access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        val = _deep_get(self._raw, dotted, default)
        return default if val is None else val

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    # secrets -----------------------------------------------------------
    @property
    def openrouter_api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY") or None

    # ffmpeg resolution -------------------------------------------------
    @staticmethod
    def _is_executable(p: str | Path) -> bool:
        p = Path(p)
        return p.is_file() and os.access(p, os.X_OK)

    def _resolve_binary(self, name: str, cfg_key: str, env_key: str) -> str | None:
        cfg_val = self.get(cfg_key, "auto")
        # 1) explicit path in config
        if cfg_val and cfg_val != "auto" and self._is_executable(cfg_val):
            return str(Path(cfg_val).resolve())
        # 2) explicit path in env
        env_val = os.environ.get(env_key)
        if env_val and self._is_executable(env_val):
            return str(Path(env_val).resolve())
        # 3) PATH
        found = shutil.which(name)
        if found:
            return found
        # 4) bundled tools/ (bootstrap drops static builds here)
        for cand in (
            TOOLS_DIR / f"{name}.exe",
            TOOLS_DIR / name,
            TOOLS_DIR / "ffmpeg" / f"{name}.exe",
            TOOLS_DIR / "ffmpeg" / name,
        ):
            if self._is_executable(cand):
                return str(cand.resolve())
        # 5) nested extracted folder pattern tools/ffmpeg-*/bin
        if TOOLS_DIR.exists():
            for exe in TOOLS_DIR.glob(f"**/bin/{name}*"):
                if self._is_executable(exe):
                    return str(exe.resolve())
            for exe in TOOLS_DIR.glob(f"**/{name}.exe"):
                if self._is_executable(exe):
                    return str(exe.resolve())
        return None

    @property
    def ffmpeg(self) -> str | None:
        if self._ffmpeg is None:
            self._ffmpeg = self._resolve_binary("ffmpeg", "paths.ffmpeg", "FFMPEG_PATH")
        return self._ffmpeg

    @property
    def ffprobe(self) -> str | None:
        if self._ffprobe is None:
            self._ffprobe = self._resolve_binary("ffprobe", "paths.ffprobe", "FFPROBE_PATH")
        return self._ffprobe


# --- Cached singleton -------------------------------------------------------
_lock = threading.Lock()
_config: Config | None = None


def load_config(path: Path = CONFIG_PATH) -> Config:
    load_dotenv()
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(raw)


def get_config(reload: bool = False) -> Config:
    global _config
    with _lock:
        if _config is None or reload:
            ensure_dirs()
            _config = load_config()
        return _config
