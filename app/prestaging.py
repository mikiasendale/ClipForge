"""Pre-staging — the anticipation engine (feature P0, spec §3.2).

Runs ``hours_before`` the next scheduled job and performs tomorrow's pipeline UP
TO (but not including) render: rotate active channels → pick a fresh video →
insert ``status='staged'`` → download → transcribe (cached) → compute candidate
windows. It never sets ``used_at``, never counts against quota, never renders.
The morning job consumes staged rows and skips download/transcribe entirely.

Backlog is capped at one staged video per rotation slot (channel); staged rows
older than 3 days are purged (files + rows, never marked used). A per-channel
failure emits a notification and moves to the next slot instead of silently
failing at 6 AM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import config as cfg, db, selector, downloader, analyzer, notifier, events

LogFn = Callable[[str], None]


def _noop(_m: str) -> None:
    return None


def prestage_once(log: LogFn = _noop) -> int:
    """Stage up to the daily quota of fresh channels. Returns count staged."""
    c = cfg.get_config()
    if not c.get("prestaging.enabled", True):
        return 0
    purge_stale(log)

    rotation = selector.ordered_rotation()
    if not rotation:
        log("[prestage] no active channels")
        return 0

    quota = int(c.get("job.daily_quota", 4))
    # how many slots still need a staged video
    slots = [ch for ch in rotation if not db.staged_for_channel(ch["id"])]
    to_stage = min(quota, len(slots))
    staged = 0
    for ch in slots[:to_stage]:
        try:
            vid = _stage_channel(ch, log)
            if vid:
                staged += 1
                events.job_progress("prestage", f"staged {vid}", 0)
        except Exception as e:  # dead link, download failure, etc.
            log(f"[prestage] {ch['title']} failed: {e}")
            notifier.notify("prestage_fail", "warn",
                            f"Pre-staging skipped {ch['title']}",
                            str(e)[:180],
                            actions=[{"label": "Retry now", "action": "prestage_retry",
                                      "args": {"channel_id": ch["id"]}}],
                            dedupe_key=f"prestage_fail:{ch['id']}")
            continue
    log(f"[prestage] staged {staged} video(s)")
    return staged


def _stage_channel(ch: dict, log: LogFn) -> str | None:
    c = cfg.get_config()
    min_s = int(c.get("job.min_duration_s", 180))
    max_s = int(c.get("job.max_duration_s", 1800))
    videos = selector.list_channel_videos(ch)
    ranked = selector.score_videos(videos, min_s, max_s, db.known_video_ids())
    if not ranked:
        log(f"[prestage] {ch['title']}: no eligible fresh videos")
        return None
    best = ranked[0]
    vid = best["video_id"]
    db.insert_video(vid, ch["id"], best.get("title"), best.get("duration_s"),
                    best.get("views"), best.get("upload_date"),
                    thumbnail=best.get("thumbnail"))
    log(f"[prestage] {ch['title']}: staging {vid} ({best.get('title')})")

    db.set_video_status(vid, "pending")
    src = downloader.download(vid)
    if not src or not Path(src).is_file():
        db.set_video_status(vid, "failed")
        raise RuntimeError("download failed")
    db.set_video_status(vid, "downloaded")
    topic = ch["topic"]
    mode = c.topic_mode(topic)
    try:
        analysis = analyzer.analyze(Path(src), topic, mode, video_id=vid)
    except Exception as e:
        # transcription/analysis failed -> drop the row so it can retry cleanly
        db.delete_video(vid)
        raise RuntimeError(f"analysis failed: {e}")
    db.mark_video_staged(vid)
    log(f"[prestage] staged {vid}: {len(analysis.windows)} candidate windows")
    return vid


def purge_stale(log: LogFn = _noop) -> int:
    """Delete staged rows + their downloads older than 3 days (never 'used')."""
    removed = 0
    for vid in db.stale_staged_ids(3):
        try:
            src = downloader.resolve_source(vid)
            if src:
                Path(src).unlink(missing_ok=True)
                wav = src.with_suffix(".16k.wav")
                wav.unlink(missing_ok=True)
        except OSError:
            pass
        db.delete_video(vid)
        removed += 1
    if removed:
        log(f"[prestage] purged {removed} stale staged video(s)")
    return removed
