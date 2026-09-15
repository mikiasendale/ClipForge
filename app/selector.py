"""Video picker + channel rotation + quota math (spec §5).

Rotation
--------
All picked channels are flattened into one order by round-robin across the
configured ``rotation_order`` topics, so the leading sequence is
``cats_silent, football, boxing, cats_compilations, cats_silent, ...`` exactly
as the spec describes. ``state.rotation_index`` advances one step per channel
we attempt; because it is persisted, a restart continues the rotation and never
re-picks the same slot (spec §11.4).

No-repeat
---------
``known_video_ids`` (every row ever inserted) is excluded when listing a
channel, so a video is never processed twice. ``used_at`` is only set AFTER a
successful render, so a mid-job crash leaves the video reusable on the next
run (spec §11.7).
"""
from __future__ import annotations

import math
from typing import Sequence

from . import config as cfg
from . import db
from . import discovery


# --- quota / window math (pure, testable) -----------------------------------
def windows_for_duration(duration_s: float, max_windows: int = 4) -> int:
    """N = min(4, max(1, round(duration_min / 5))) (spec §5)."""
    if not duration_s or duration_s <= 0:
        return 1
    minutes = duration_s / 60.0
    return int(max(1, min(max_windows, round(minutes / 5))))


def clamp_windows(requested: int, quota_remaining: int) -> int:
    """N = min(N, quota remaining)."""
    return max(0, min(requested, quota_remaining))


# --- rotation (pure) --------------------------------------------------------
def build_rotation(channels_by_topic: dict[str, Sequence[dict]],
                   rotation_order: Sequence[str]) -> list[dict]:
    """Interleave channels by topic round-robin following rotation_order.

    With pools cats_silent[2], football[3], boxing[3], cats_comp[2] the result
    starts cats_silent, football, boxing, cats_comp, cats_silent, ... matching
    the spec's stated rotation.
    """
    pools = {t: list(channels_by_topic.get(t, [])) for t in rotation_order}
    max_len = max((len(p) for p in pools.values()), default=0)
    rotation: list[dict] = []
    for rank in range(max_len):
        for topic in rotation_order:
            pool = pools.get(topic, [])
            if rank < len(pool):
                rotation.append(pool[rank])
    return rotation


# --- channel rotation across the DB -----------------------------------------
def ordered_rotation() -> list[dict]:
    c = cfg.get_config()
    order = c.get("discovery.rotation_order", []) or []
    grouped: dict[str, list[dict]] = {}
    for row in db.active_channels():
        grouped.setdefault(row["topic"], []).append(dict(row))
    return build_rotation(grouped, order)


def pick_next_channel(skip_channel_ids: set[int] | None = None) -> dict | None:
    """Advance rotation and return the next untried channel row (or None)."""
    skip = skip_channel_ids or set()
    rotation = ordered_rotation()
    if not rotation:
        return None
    idx = db.get_int_state("rotation_index", 0)
    n = len(rotation)
    # advance until we find a channel not already tried this run
    for step in range(n):
        ch = rotation[(idx + step) % n]
        if ch["id"] not in skip:
            db.set_int_state("rotation_index", (idx + step + 1) % n)
            return ch
    db.set_int_state("rotation_index", (idx + 1) % n)
    return None


# --- video scoring (pure, testable) -----------------------------------------
def _mid_high_curve(p: float, center: float = 0.75, sigma: float = 0.18) -> float:
    """Gaussian favouring mid-high views; penalises the viral outlier & duds."""
    return math.exp(-((p - center) ** 2) / (2 * sigma * sigma))


def score_videos(videos: list[dict], min_s: int, max_s: int,
                 exclude_ids: set[str]) -> list[dict]:
    """Filter by duration + exclusions, then rank by normalized mid-high views."""
    pool = [
        v for v in videos
        if v.get("video_id")
        and v["video_id"] not in exclude_ids
        and v.get("duration_s") is not None
        and min_s <= float(v["duration_s"]) <= max_s
    ]
    if not pool:
        return []
    views = [float(v.get("views") or 0) for v in pool]
    lo, hi = min(views), max(views)
    span = (hi - lo) or 1.0
    for v in pool:
        p = (float(v.get("views") or 0) - lo) / span
        v["_score"] = round(_mid_high_curve(p), 5)
    pool.sort(key=lambda v: v["_score"], reverse=True)
    return pool


def list_channel_videos(channel_row: dict) -> list[dict]:
    """yt-dlp /videos flat playlist -> [{video_id, title, duration_s, views, upload_date}]."""
    cid = channel_row["platform_channel_id"]
    url = f"https://www.youtube.com/channel/{cid}/videos" if cid.startswith("UC") else \
          f"https://www.youtube.com/{cid}/videos"
    info = discovery.ytdlp_json([url, "--flat-playlist", "--playlist-items", "1-100"], timeout=90)
    out: list[dict] = []
    if not info:
        return out
    for e in info.get("entries") or []:
        if not isinstance(e, dict) or not e.get("id"):
            continue
        thumb = None
        for t in (e.get("thumbnails") or [])[::-1]:
            if t.get("url"):
                thumb = t["url"]
                break
        out.append({
            "video_id": e["id"],
            "title": e.get("title"),
            "duration_s": e.get("duration"),
            "views": e.get("view_count"),
            "upload_date": e.get("upload_date"),
            "thumbnail": thumb,
        })
    return out


def pick_video(channel_row: dict) -> dict | None:
    """Return the best not-yet-used 3–30 min video for a channel, or None."""
    c = cfg.get_config()
    min_s = int(c.get("job.min_duration_s", 180))
    max_s = int(c.get("job.max_duration_s", 1800))
    videos = list_channel_videos(channel_row)
    ranked = score_videos(videos, min_s, max_s, db.known_video_ids())
    if not ranked:
        return None
    best = ranked[0]
    db.insert_video(
        best["video_id"], channel_row["id"], best.get("title"),
        best.get("duration_s"), best.get("views"), best.get("upload_date"),
    )
    return best
