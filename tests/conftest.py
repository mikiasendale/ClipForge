"""Shared pytest fixtures.

Every test runs against a throwaway data dir (config module paths are patched)
and never touches the network: OpenRouter and the heavy analyzers are
monkeypatched, and video media is synthesized locally with ffmpeg.
"""
from __future__ import annotations

import subprocess

import pytest

from app import config as cfg
from app import db


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Point all project paths at tmp_path and init a fresh DB.

    Also clears any ambient OPENROUTER_API_KEY so single-shot tests stay
    deterministic regardless of the developer's local .env; agent tests inject
    their own fake transport.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg._config = None  # drop any cached config carrying a real key
    data = tmp_path / "data"
    for name in ("downloads", "output", "logs", "frames"):
        (data / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "DOWNLOADS_DIR", data / "downloads")
    monkeypatch.setattr(cfg, "OUTPUT_DIR", data / "output")
    monkeypatch.setattr(cfg, "LOGS_DIR", data / "logs")
    monkeypatch.setattr(cfg, "FRAMES_DIR", data / "frames")
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path / "tools")
    dbp = data / "clipforge.db"
    monkeypatch.setattr(cfg, "DB_PATH", dbp)
    db.init_db(dbp)
    return tmp_path


@pytest.fixture()
def synth_720p(tmp_path):
    """Generate a real 1280x720 mp4 with audio + hard cuts (for cutter/analyzer)."""
    out = tmp_path / "src720.mp4"
    # 3 concatenated solid-colour 4s segments -> guaranteed scene changes, + sine audio
    parts = []
    for i, color in enumerate(("red", "green", "blue")):
        p = tmp_path / f"seg{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c={color}:s=1280x720:r=30:d=4",
            "-f", "lavfi", "-i", f"sine=frequency={220*(i+1)}:duration=4",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", str(p),
        ], capture_output=True, check=True, timeout=120)
        parts.append(p)
    lst = tmp_path / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(out),
    ], capture_output=True, check=True, timeout=120)
    return out
