"""Video download via yt-dlp (spec §2 / §5).

    yt-dlp -S "res:720,codec:h264" --merge-output-format mp4 -o <downloads>/<id>.%(ext)s <url>

Returns the resolved local path (asks yt-dlp for it via ``--print after_move:filepath``).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config as cfg


def _yt_dlp() -> str | None:
    return shutil.which("yt-dlp")


def _safe_stem(video_id: str) -> str:
    return "".join(ch for ch in video_id if ch.isalnum() or ch in ("-", "_"))[:80] or "video"


def resolve_source(video_id: str, dest_dir: Path | None = None) -> Path | None:
    """Locate an already-downloaded source file for a video id (for review/re-render)."""
    dest_dir = Path(dest_dir or cfg.DOWNLOADS_DIR)
    if not dest_dir.exists():
        return None
    stem = _safe_stem(video_id)
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        cand = dest_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    matches = [f for f in dest_dir.glob(f"{stem}.*") if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
    return matches[0] if matches else None


def download(video_id: str, url: str | None = None, dest_dir: Path | None = None) -> Path | None:
    exe = _yt_dlp()
    if not exe:
        return None
    dest_dir = Path(dest_dir or cfg.DOWNLOADS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = url or f"https://www.youtube.com/watch?v={video_id}"
    out_tmpl = str(dest_dir / f"{_safe_stem(video_id)}.%(ext)s")
    cmd = [
        exe,
        "-S", "res:720,codec:h264",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--restrict-filenames",
        "-o", out_tmpl,
        "--print", "after_move:filepath",
        "--no-simulate",
        source,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError):
        return None

    # last stdout line printed by --print is the final file path
    for line in reversed(proc.stdout.strip().splitlines()):
        cand = Path(line.strip())
        if cand.is_file():
            return cand

    # fall back to locating the expected mp4
    expected = dest_dir / f"{_safe_stem(video_id)}.mp4"
    if expected.is_file():
        return expected
    for f in dest_dir.glob(f"{_safe_stem(video_id)}.*"):
        if f.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return f
    if proc.returncode != 0:
        _log_failure(proc)
    return None


def _log_failure(proc: subprocess.CompletedProcess) -> None:
    try:
        cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (cfg.LOGS_DIR / "download.log").open("a", encoding="utf-8") as fh:
            fh.write(f"yt-dlp rc={proc.returncode}\n{proc.stderr[-2000:]}\n")
    except Exception:
        pass
