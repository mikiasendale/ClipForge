"""Suggestions engine (feature P2, spec §3.7) — pure rules, no LLM.

Generates ``kind='suggestion'`` notifications (deduped 7 days). Rules:
  1. Channel yield  — a picked channel producing < 0.34 clips/video over ≥3
     attempts → offer to swap in the next unused onboarding candidate.
  2. Adjustment     — ≥60% of a topic's agent clips were Adjusted this week →
     offer to append a tuning line to that topic's saved prompt.
  3. Disk cleanup   — sources whose clips are all approved/discarded and older
     than ``review.cleanup_days`` → offer to delete them.
  4. Streak         — N consecutive days at full quota (informational).

Only rule 2's [Apply] and auto-approve self-execute; everything else needs an
explicit user action, which main.py executes behind a confirm flag.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from . import config as cfg, db, notifier


def run_all(log=lambda *_a: None) -> int:
    n = 0
    for rule in (yield_rule, adjustment_rule, cleanup_rule, streak_rule):
        try:
            if rule():
                n += 1
        except Exception as e:  # a bad rule must never break a run
            log(f"[suggest] {rule.__name__} error: {e}")
    return n


# --- rule 1: channel yield --------------------------------------------------
def _channel_yield(cid: int, days: int = 14) -> tuple[float, int, int]:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    attempted = db.query_one(
        "SELECT COUNT(*) c FROM videos WHERE channel_id=? AND used_at IS NOT NULL "
        "AND used_at>=?", (cid, cutoff))["c"]
    rendered = db.query_one(
        "SELECT COUNT(*) c FROM clips c JOIN videos v ON v.video_id=c.video_id "
        "WHERE v.channel_id=? AND c.created_at>=?", (cid, cutoff))["c"]
    att = int(attempted)
    rend = int(rendered)
    ratio = (rend / att) if att else 1.0
    return ratio, att, rend


def yield_rule() -> bool:
    emitted = False
    for ch in db.active_channels():
        ratio, att, rend = _channel_yield(ch["id"])
        if att >= 3 and ratio < 0.34:
            replacement = _next_candidate(ch["topic"], exclude_ids={ch["id"]})
            if not replacement:
                continue
            notifier.notify(
                "suggestion", "info",
                f"{ch['title']} is underperforming",
                f"{rend} clip(s) from {att} videos. Swap to {replacement['title']}?",
                actions=[{"label": "Swap", "action": "swap_channel",
                          "args": {"deactivate_id": ch["id"],
                                   "activate_id": replacement["id"]}},
                         {"label": "Dismiss", "action": "dismiss", "args": {}}],
                dedupe_key=f"yield:{ch['id']}")
            emitted = True
    return emitted


def _next_candidate(topic: str, exclude_ids: set[int]) -> dict | None:
    """Next inactive stored candidate for the topic (from the onboarding pool)."""
    rows = db.query(
        "SELECT * FROM channels WHERE topic=? AND is_active=0 ORDER BY subs DESC LIMIT 1",
        (topic,))
    return dict(rows[0]) if rows else None


# --- rule 2: adjustment pattern ---------------------------------------------
def adjustment_rule() -> bool:
    c = cfg.get_config()
    week_ago = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
    topics = (c.get("discovery.topics", {}) or {}).keys()
    emitted = False
    for topic in topics:
        agent_clips = db.query_one(
            "SELECT COUNT(*) c FROM clips cl JOIN videos v ON v.video_id=cl.video_id "
            "JOIN channels ch ON ch.id=v.channel_id "
            "WHERE ch.topic=? AND cl.engine LIKE 'agent%' AND cl.created_at>=?",
            (topic, week_ago))["c"]
        adjusted = db.query_one(
            "SELECT COUNT(*) c FROM clips cl JOIN videos v ON v.video_id=cl.video_id "
            "JOIN channels ch ON ch.id=v.channel_id "
            "WHERE ch.topic=? AND cl.engine='review' AND cl.created_at>=?",
            (topic, week_ago))["c"]
        if int(agent_clips) >= 5 and int(adjusted) / max(1, int(agent_clips)) >= 0.6:
            delta = _median_start_shift(topic, week_ago)
            sign = "+" if delta >= 0 else ""
            notifier.notify(
                "suggestion", "info",
                f"You often adjust {topic} clips",
                f"{adjusted}/{agent_clips} recent {topic} clips were edited "
                f"(median in-point shift {sign}{delta:.0f}s). "
                f"Add 'skip first {max(1,int(abs(delta)))}s of windows' to the {topic} prompt?",
                actions=[{"label": "Apply", "action": "apply_prompt_tune",
                          "args": {"topic": topic, "line":
                                   f"Prefer windows that start at least {max(1,int(abs(delta)))}s "
                                   "into the clip."}},
                         {"label": "Dismiss", "action": "dismiss", "args": {}}],
                dedupe_key=f"adjust:{topic}")
            emitted = True
    return emitted


def _median_start_shift(topic: str, week_ago: str) -> float:
    rows = db.query(
        "SELECT cl.start_s base_start, child.start_s new_start FROM clips cl "
        "JOIN clips child ON child.parent_clip_id=cl.id "
        "JOIN videos v ON v.video_id=cl.video_id JOIN channels ch ON ch.id=v.channel_id "
        "WHERE ch.topic=? AND cl.created_at>=?", (topic, week_ago))
    if not rows:
        return 0.0
    deltas = sorted(float(r["new_start"]) - float(r["base_start"]) for r in rows)
    return deltas[len(deltas) // 2]


# --- rule 3: disk cleanup ---------------------------------------------------
def cleanup_rule() -> bool:
    c = cfg.get_config()
    days = int(c.get("review.cleanup_days", 7) or 7)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    stale = []
    total_bytes = 0
    for v in db.query("SELECT video_id FROM videos WHERE used_at IS NOT NULL AND used_at<=?",
                      (cutoff,)):
        vid = v["video_id"]
        active = db.query_one(
            "SELECT COUNT(*) c FROM clips WHERE video_id=? AND revised_at IS NULL", (vid,))["c"]
        pending = db.query_one(
            "SELECT COUNT(*) c FROM clips WHERE video_id=? AND revised_at IS NULL "
            "AND status='pending'", (vid,))["c"]
        if int(active) == 0 or int(pending) > 0:
            continue
        src = _source_path(vid)
        if src:
            stale.append((vid, src))
            total_bytes += src.stat().st_size
    if stale:
        notifier.notify(
            "suggestion", "info", "Old sources can be deleted",
            f"Delete {len(stale)} reviewed source file(s), frees {total_bytes/1e9:.1f} GB.",
            actions=[{"label": "Clean up", "action": "cleanup_sources",
                      "args": {"video_ids": [vid for vid, _ in stale]}, "confirm": True},
                     {"label": "Dismiss", "action": "dismiss", "args": {}}],
            dedupe_key="cleanup")
    return bool(stale)


def _source_path(video_id: str):
    from . import downloader
    return downloader.resolve_source(video_id)


# --- rule 4: streak (informational) -----------------------------------------
def streak_rule() -> bool:
    c = cfg.get_config()
    quota = int(c.get("job.daily_quota", 4))
    streak = 0
    d = datetime.now().date()
    for _ in range(60):
        day = d.isoformat()
        cnt = db.query_one(
            "SELECT COUNT(*) c FROM clips WHERE substr(created_at,1,10)=?", (day,))["c"]
        if int(cnt) >= quota:
            streak += 1
            d -= timedelta(days=1)
        else:
            break
    if streak >= 2:
        notifier.notify(
            "streak", "info", f"{streak}-day full-quota streak",
            "You've hit the daily quota several days running.",
            dedupe_key=f"streak:{datetime.now().date()}")  # refreshes at most once/day
        return True
    return False
