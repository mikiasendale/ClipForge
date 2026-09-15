"""SQLite persistence layer for ClipForge.

WAL mode for safe concurrent read during background writes. A single module
connection guard keeps writes serialized (the app runs exactly one job at a
time, but the UI polls concurrently).

Schema (spec §3):
  channels, videos, clips, state
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config as cfg

_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_channel_id TEXT UNIQUE NOT NULL,
    title               TEXT NOT NULL,
    topic               TEXT NOT NULL,
    subtopic            TEXT,
    subs                INTEGER,
    video_count         INTEGER,
    joined_year         INTEGER,
    avatar_url          TEXT,
    added_at            TEXT NOT NULL,
    is_active           INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    channel_id  INTEGER NOT NULL REFERENCES channels(id),
    title       TEXT,
    duration_s  REAL,
    views       INTEGER,
    upload_date TEXT,
    thumbnail   TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    used_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status  ON videos(status);

CREATE TABLE IF NOT EXISTS clips (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id),
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    path          TEXT,
    caption       TEXT,
    model         TEXT,
    prompt_source TEXT NOT NULL DEFAULT 'default',
    engine        TEXT NOT NULL DEFAULT 'single_shot',
    render_mode   TEXT NOT NULL DEFAULT 'mp4',
    draft_path    TEXT,
    hook_title    TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    approved_at   TEXT,
    revised_at    TEXT,
    parent_clip_id INTEGER REFERENCES clips(id),
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    title       TEXT NOT NULL,
    body        TEXT,
    actions_json TEXT,
    created_at  TEXT NOT NULL,
    read_at     TEXT,
    acted_at    TEXT,
    dedupe_key  TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_notif_created ON notifications(created_at);
"""


# Columns added after the initial release; ALTER is idempotent via pragma check.
_CLIP_MIGRATIONS = [
    ("engine", "TEXT NOT NULL DEFAULT 'single_shot'"),
    ("revised_at", "TEXT"),
    ("parent_clip_id", "INTEGER REFERENCES clips(id)"),
    ("hook_title", "TEXT"),
    ("render_mode", "TEXT NOT NULL DEFAULT 'mp4'"),
    ("draft_path", "TEXT"),
    ("status", "TEXT NOT NULL DEFAULT 'pending'"),
    ("approved_at", "TEXT"),
]
_CHANNEL_MIGRATIONS = [
    ("is_active", "INTEGER NOT NULL DEFAULT 1"),
]
_VIDEO_MIGRATIONS = [
    ("thumbnail", "TEXT"),
    ("staged_at", "TEXT"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(db_path or cfg.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    with _LOCK:
        conn = connect(db_path)
        try:
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
        finally:
            conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the baseline, if missing."""
    for table, migrations in (
        ("clips", _CLIP_MIGRATIONS),
        ("channels", _CHANNEL_MIGRATIONS),
        ("videos", _VIDEO_MIGRATIONS),
    ):
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, decl in migrations:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def query(sql: str, params: Iterable[Any] = (), db_path: Path | str | None = None) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return list(conn.execute(sql, tuple(params)).fetchall())
    finally:
        conn.close()


def query_one(sql: str, params: Iterable[Any] = (), db_path: Path | str | None = None) -> sqlite3.Row | None:
    rows = query(sql, params, db_path)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = (), db_path: Path | str | None = None) -> int:
    """Single write. Returns lastrowid (inserts) or rowcount."""
    with _LOCK:
        conn = connect(db_path)
        try:
            cur = conn.execute(sql, tuple(params))
            conn.commit()
            return cur.lastrowid if cur.lastrowid else cur.rowcount
        finally:
            conn.close()


# --- channels ---------------------------------------------------------------
def add_channel(platform_channel_id: str, title: str, topic: str, subtopic: str | None,
                subs: int | None, video_count: int | None, joined_year: int | None,
                avatar_url: str | None, db_path: Path | str | None = None,
                *, is_active: int = 1) -> int:
    existing = query_one(
        "SELECT id FROM channels WHERE platform_channel_id=?", (platform_channel_id,), db_path
    )
    if existing:
        execute(
            """UPDATE channels SET title=?, topic=?, subtopic=?, subs=?, video_count=?,
                      joined_year=?, avatar_url=?, is_active=? WHERE id=?""",
            (title, topic, subtopic, subs, video_count, joined_year, avatar_url,
             int(is_active), existing["id"]),
            db_path,
        )
        return existing["id"]
    return execute(
        """INSERT INTO channels(platform_channel_id, title, topic, subtopic, subs, video_count,
                                joined_year, avatar_url, added_at, is_active)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (platform_channel_id, title, topic, subtopic, subs, video_count, joined_year,
         avatar_url, now_iso(), int(is_active)),
        db_path,
    )


def all_channels(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM channels ORDER BY topic, id", (), db_path)


def active_channels(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM channels WHERE is_active=1 ORDER BY topic, id", (), db_path)


def channels_by_topic(topic: str, only_active: bool = False,
                      db_path: Path | str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM channels WHERE topic=?"
    if only_active:
        sql += " AND is_active=1"
    return query(sql + " ORDER BY id", (topic,), db_path)


def set_channel_active(channel_id: int, active: bool, db_path: Path | str | None = None) -> None:
    execute("UPDATE channels SET is_active=? WHERE id=?", (1 if active else 0, channel_id), db_path)


def swap_channels(deactivate_id: int, activate_id: int, db_path: Path | str | None = None) -> None:
    """Deactivate one channel, activate its replacement (suggestion [Swap])."""
    set_channel_active(deactivate_id, False, db_path)
    set_channel_active(activate_id, True, db_path)


def channel_count(only_active: bool = True, db_path: Path | str | None = None) -> int:
    sql = "SELECT COUNT(*) AS c FROM channels"
    if only_active:
        sql += " WHERE is_active=1"
    row = query_one(sql, (), db_path)
    return int(row["c"]) if row else 0


# --- videos -----------------------------------------------------------------
def insert_video(video_id: str, channel_id: int, title: str | None, duration_s: float | None,
                 views: int | None, upload_date: str | None, db_path: Path | str | None = None,
                 *, status: str = "pending", thumbnail: str | None = None) -> None:
    execute(
        """INSERT OR REPLACE INTO videos(video_id, channel_id, title, duration_s, views,
                                         upload_date, thumbnail, status, used_at)
           VALUES(?,?,?,?,?,?,?,?,NULL)""",
        (video_id, channel_id, title, duration_s, views, upload_date, thumbnail, status),
        db_path,
    )


def set_video_status(video_id: str, status: str, db_path: Path | str | None = None) -> None:
    execute("UPDATE videos SET status=? WHERE video_id=?", (status, video_id), db_path)


def mark_video_used(video_id: str, db_path: Path | str | None = None) -> None:
    """used_at NOT NULL => never reuse (spec: set AFTER successful clip render)."""
    execute(
        "UPDATE videos SET used_at=?, status='done' WHERE video_id=?",
        (now_iso(), video_id), db_path,
    )


def known_video_ids(db_path: Path | str | None = None) -> set[str]:
    rows = query("SELECT video_id FROM videos", (), db_path)
    return {r["video_id"] for r in rows}


def video_with_channel(video_id: str, db_path: Path | str | None = None) -> sqlite3.Row | None:
    return query_one(
        """SELECT v.video_id, v.title, v.duration_s, v.status, v.thumbnail, v.channel_id,
                  c.title AS channel_title, c.topic
           FROM videos v JOIN channels c ON c.id = v.channel_id
           WHERE v.video_id=?""",
        (video_id,), db_path,
    )


def staged_videos(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query(
        """SELECT v.video_id, v.title, v.duration_s, v.thumbnail, v.channel_id,
                  v.upload_date, c.title AS channel_title, c.topic
           FROM videos v JOIN channels c ON c.id=v.channel_id
           WHERE v.status='staged' ORDER BY v.upload_date DESC""",
        (), db_path,
    )


def staged_for_channel(channel_id: int, db_path: Path | str | None = None) -> sqlite3.Row | None:
    return query_one(
        "SELECT * FROM videos WHERE channel_id=? AND status='staged' ORDER BY views DESC LIMIT 1",
        (channel_id,), db_path,
    )


def mark_video_staged(video_id: str, thumbnail: str | None = None,
                      db_path: Path | str | None = None) -> None:
    execute("UPDATE videos SET status='staged', staged_at=?, thumbnail=COALESCE(?, thumbnail) "
            "WHERE video_id=?", (now_iso(), thumbnail, video_id), db_path)


def stale_staged_ids(days: int, db_path: Path | str | None = None) -> list[str]:
    cutoff = _days_ago_iso(days)
    rows = query(
        "SELECT video_id FROM videos WHERE status='staged' AND staged_at IS NOT NULL AND staged_at < ?",
        (cutoff,), db_path)
    return [r["video_id"] for r in rows]


def delete_video(video_id: str, db_path: Path | str | None = None) -> None:
    """Hard-delete a staged row (pre-staging purge; never touches used rows)."""
    execute("DELETE FROM videos WHERE video_id=? AND used_at IS NULL", (video_id,), db_path)


def _days_ago_iso(days: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc).astimezone() - timedelta(days=days)).isoformat(timespec="seconds")


# --- clips ------------------------------------------------------------------
def add_clip(video_id: str, start_s: float, end_s: float, path: str | None, caption: str | None,
             model: str | None, prompt_source: str, engine: str = "single_shot",
             parent_clip_id: int | None = None, hook_title: str | None = None,
             render_mode: str = "mp4", draft_path: str | None = None,
             db_path: Path | str | None = None) -> int:
    return execute(
        """INSERT INTO clips(video_id, start_s, end_s, path, caption, model,
                             prompt_source, engine, parent_clip_id, hook_title,
                             render_mode, draft_path, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (video_id, start_s, end_s, path, caption, model, prompt_source,
         engine, parent_clip_id, hook_title, render_mode, draft_path, now_iso()),
        db_path,
    )


def all_clips(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM clips ORDER BY created_at DESC", (), db_path)


def current_clips(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Unrevised clips only (superseded versions hidden from the gallery)."""
    return query("SELECT * FROM clips WHERE revised_at IS NULL ORDER BY created_at DESC",
                 (), db_path)


def get_clip(clip_id: int, db_path: Path | str | None = None) -> sqlite3.Row | None:
    return query_one("SELECT * FROM clips WHERE id=?", (clip_id,), db_path)


def video_clips_active(video_id: str, exclude_id: int | None = None,
                       db_path: Path | str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM clips WHERE video_id=? AND revised_at IS NULL"
    params: list[Any] = [video_id]
    if exclude_id:
        sql += " AND id!=?"
        params.append(exclude_id)
    return query(sql, params, db_path)


def mark_clip_revised(clip_id: int, db_path: Path | str | None = None) -> None:
    execute("UPDATE clips SET revised_at=? WHERE id=?", (now_iso(), clip_id), db_path)


def set_clip_status(clip_id: int, status: str, db_path: Path | str | None = None) -> None:
    """approved | discarded (review decisions). Never refunds video quota."""
    execute("UPDATE clips SET status=? WHERE id=?", (status, clip_id), db_path)


def approve_clip(clip_id: int, db_path: Path | str | None = None) -> None:
    execute("UPDATE clips SET status='approved', approved_at=? WHERE id=?",
            (now_iso(), clip_id), db_path)


def discard_clip(clip_id: int, db_path: Path | str | None = None) -> None:
    execute("UPDATE clips SET status='discarded' WHERE id=?", (clip_id,), db_path)


def unreviewed_clips(older_than_hours: float | None = None,
                     db_path: Path | str | None = None) -> list[sqlite3.Row]:
    sql = ("SELECT c.*, v.title AS video_title FROM clips c "
           "JOIN videos v ON v.video_id=c.video_id "
           "WHERE c.revised_at IS NULL AND c.status='pending'")
    params: list[Any] = []
    if older_than_hours is not None:
        cutoff = _hours_ago_iso(older_than_hours)
        sql += " AND c.created_at <= ?"
        params.append(cutoff)
    return query(sql + " ORDER BY c.created_at ASC", params, db_path)


def clips_by_date(day: str, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM clips WHERE substr(created_at,1,10)=? AND revised_at IS NULL "
                 "ORDER BY created_at ASC", (day,), db_path)


def last_run_date(db_path: Path | str | None = None) -> str | None:
    return get_state("last_run_date", None, db_path)


def _hours_ago_iso(hours: float) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc).astimezone() - timedelta(hours=hours)).isoformat(timespec="seconds")


def clips_today(db_path: Path | str | None = None) -> int:
    today = datetime.now().date().isoformat()
    row = query_one(
        "SELECT COUNT(*) AS c FROM clips WHERE substr(created_at,1,10)=?", (today,), db_path
    )
    return int(row["c"]) if row else 0


# --- state ------------------------------------------------------------------
def get_state(key: str, default: str | None = None, db_path: Path | str | None = None) -> str | None:
    row = query_one("SELECT value FROM state WHERE key=?", (key,), db_path)
    return row["value"] if row else default


def set_state(key: str, value: str, db_path: Path | str | None = None) -> None:
    execute(
        "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)), db_path,
    )


def get_int_state(key: str, default: int = 0, db_path: Path | str | None = None) -> int:
    raw = get_state(key, None, db_path)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def set_int_state(key: str, value: int, db_path: Path | str | None = None) -> None:
    set_state(key, str(value), db_path)


# --- rolling metrics (per-day counters for health checks) -------------------
def metrics_bump(name: str, success: bool, db_path: Path | str | None = None) -> None:
    """Increment a per-day ok/fail counter (yt-dlp, openrouter)."""
    import json
    today = datetime.now().date().isoformat()
    raw = get_state("metrics", "{}", db_path)
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        m = {}
    if m.get("day") != today:
        m = {"day": today, "counters": {}}
    counters = m.setdefault("counters", {})
    slot = counters.setdefault(name, {"ok": 0, "fail": 0})
    slot["ok" if success else "fail"] = slot.get("ok" if success else "fail", 0) + 1
    if success:
        slot["consec_fail"] = 0
    else:
        slot["consec_fail"] = slot.get("consec_fail", 0) + 1
    set_state("metrics", json.dumps(m), db_path)


def metrics_snapshot(db_path: Path | str | None = None) -> dict:
    import json
    raw = get_state("metrics", "{}", db_path)
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        m = {}
    if m.get("day") != datetime.now().date().isoformat():
        return {"day": datetime.now().date().isoformat(), "counters": {}}
    return m


def metric(name: str, db_path: Path | str | None = None) -> dict:
    return metrics_snapshot(db_path).get("counters", {}).get(name, {"ok": 0, "fail": 0, "consec_fail": 0})


# --- reset helpers (tests / crash recovery) ---------------------------------
def reset_run_state(db_path: Path | str | None = None) -> None:
    """Clear any in-flight job markers left behind by a crash (spec §11.7)."""
    execute("DELETE FROM state WHERE key='job_active'", (), db_path)


# --- notifications ----------------------------------------------------------
def insert_notification(kind: str, severity: str, title: str, body: str | None,
                        actions_json: str | None, dedupe_key: str | None,
                        db_path: Path | str | None = None) -> int | None:
    """Insert (or refresh an aged-out duplicate) notification; returns its row id.

    Dedup-within-window is enforced by notifier via dedupe_seen(); this upsert
    keeps history and refreshes the card in place when a keyed notification
    re-fires after its window, so the UNIQUE(dedupe_key) never hard-blocks it.
    """
    ts = now_iso()
    if dedupe_key is None:
        return execute(
            """INSERT INTO notifications(kind, severity, title, body, actions_json,
                                         created_at) VALUES(?,?,?,?,?,?)""",
            (kind, severity, title, body, actions_json, ts), db_path)
    execute(
        """INSERT INTO notifications(kind, severity, title, body, actions_json,
                                     created_at, dedupe_key)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(dedupe_key) DO UPDATE SET
             kind=excluded.kind, severity=excluded.severity, title=excluded.title,
             body=excluded.body, actions_json=excluded.actions_json,
             created_at=excluded.created_at, read_at=NULL, acted_at=NULL""",
        (kind, severity, title, body, actions_json, ts, dedupe_key), db_path)
    row = query_one("SELECT id FROM notifications WHERE dedupe_key=?", (dedupe_key,), db_path)
    return row["id"] if row else None


def dedupe_seen(dedupe_key: str, within_hours: float, db_path: Path | str | None = None) -> bool:
    cutoff = _hours_ago_iso(within_hours)
    row = query_one(
        "SELECT id FROM notifications WHERE dedupe_key=? AND created_at>=? "
        "ORDER BY created_at DESC LIMIT 1", (dedupe_key, cutoff), db_path)
    return row is not None


def list_notifications(limit: int = 50, unread_only: bool = False,
                       db_path: Path | str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM notifications"
    if unread_only:
        sql += " WHERE read_at IS NULL"
    return query(sql + " ORDER BY created_at DESC LIMIT ?", (limit,), db_path)


def unread_count(db_path: Path | str | None = None) -> int:
    row = query_one("SELECT COUNT(*) AS c FROM notifications WHERE read_at IS NULL", (), db_path)
    return int(row["c"]) if row else 0


def mark_notification_read(notif_id: int, acted: bool = False,
                           db_path: Path | str | None = None) -> None:
    if acted:
        execute("UPDATE notifications SET read_at=?, acted_at=? WHERE id=?",
                (now_iso(), now_iso(), notif_id), db_path)
    else:
        execute("UPDATE notifications SET read_at=? WHERE id=?", (now_iso(), notif_id), db_path)
