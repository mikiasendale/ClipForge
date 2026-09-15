"""Health checks (feature P2, spec §3.6).

Run at boot, after each job, and daily. Each check raises a notification with a
severity + embedded action. Checks:
  * yt-dlp failure rate ≥ 50% today → "likely broken" + [Update yt-dlp]
  * OpenRouter 401 or ≥3 consecutive failures → [Open settings]
  * disk: downloads free < 2 GB or downloads dir > 10 GB → cleanup card
  * CapCut draft root unwritable → disable notice (radio already reflects)
  * quota unmet is emitted by job.daily_job itself (not here).

[Update yt-dlp] runs `pip install -U yt-dlp` into the venv and sets
``state.restart_required`` — it NEVER auto-restarts.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from . import config as cfg, db, notifier


def run_all(log=lambda *_a: None) -> None:
    for check in (check_ytdlp, check_openrouter, check_disk, check_draft_root):
        try:
            check()
        except Exception as e:  # a bad check must never break a run
            log(f"[health] {check.__name__} error: {e}")


def check_ytdlp() -> bool:
    m = db.metric("ytdlp")
    ok, fail = int(m.get("ok", 0)), int(m.get("fail", 0))
    total = ok + fail
    if total >= 4 and fail / total >= 0.5:
        rate = int(round(fail / total * 100))
        notifier.notify(
            "health", "error", "yt-dlp likely broken",
            f"{rate}% of yt-dlp calls failed today — YouTube may have changed. "
            "Update yt-dlp to try to fix discovery/downloads.",
            actions=[{"label": "Update yt-dlp", "action": "update_ytdlp", "confirm": True},
                     {"label": "Dismiss", "action": "dismiss", "args": {}}],
            dedupe_key="health:ytdlp")
        return True
    return False


def check_openrouter() -> bool:
    if db.get_state("openrouter_401") == "1":
        notifier.notify(
            "health", "error", "OpenRouter key rejected (401)",
            "Your API key was rejected. Check it in settings.",
            actions=[{"label": "Open settings", "action": "nav", "args": {"to": "/settings"}}],
            dedupe_key="health:or_401")
        return True
    m = db.metric("openrouter")
    if int(m.get("consec_fail", 0)) >= 3:
        notifier.notify(
            "health", "error", "OpenRouter failing",
            "3 consecutive editor API failures. Check your key, model name, or credits.",
            actions=[{"label": "Open settings", "action": "nav", "args": {"to": "/settings"}}],
            dedupe_key="health:or_consec")
        return True
    return False


def check_disk() -> bool:
    c = cfg.get_config()
    dl = Path(c.DOWNLOADS_DIR)
    free = 0
    dir_size = _dir_bytes(dl)
    try:
        du = shutil.disk_usage(str(dl.parent if dl.exists() else Path.cwd()))
        free = du.free
    except OSError:
        pass
    low_free = free and free < 2 * (1024 ** 3)
    big_dir = dir_size > 10 * (1024 ** 3)
    if low_free or big_dir:
        detail = (f"only {free/1e9:.1f} GB free" if low_free
                  else f"downloads folder is {dir_size/1e9:.1f} GB")
        notifier.notify(
            "health", "warn", "Low disk space",
            f"{detail}. Clean up old sources to free space.",
            actions=[{"label": "Review cleanup", "action": "nav", "args": {"to": "/dashboard"}},
                     {"label": "Clean up now", "action": "cleanup_stale", "confirm": True}],
            dedupe_key="health:disk")
        return True
    return False


def check_draft_root() -> bool:
    from . import capcut_export
    if not capcut_export.CAPCUT_AVAILABLE:
        return False
    rep = capcut_export.detect_draft_root()
    if rep.get("available") and rep.get("draft_root") and not rep.get("writable"):
        notifier.notify(
            "health", "warn", "CapCut draft root unwritable",
            "CapCut mode is disabled until the draft root is writable (mp4 mode unaffected).",
            actions=[{"label": "Open settings", "action": "nav", "args": {"to": "/settings"}}],
            dedupe_key="health:draft_root")
        return True
    return False


def _dir_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def update_ytdlp(log=lambda *_a: None) -> tuple[bool, str]:
    """[Update yt-dlp] — pip install -U into the venv; sets restart_required.

    Never restarts the app; the UI shows a 'restart to apply' banner.
    """
    exe = sys.executable
    try:
        proc = subprocess.run([exe, "-m", "pip", "install", "-U", "yt-dlp"],
                              capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"update failed: {e}"
    if proc.returncode == 0:
        db.set_state("restart_required", "yt-dlp updated")
        db.set_state("ytdlp_updated_at", db.now_iso())
        log("yt-dlp updated — restart to apply")
        return True, "yt-dlp updated — restart ClipForge to apply."
    return False, (proc.stderr or proc.stdout)[-300:]
