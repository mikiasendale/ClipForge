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
    added_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    channel_id  INTEGER NOT NULL REFERENCES channels(id),
    title       TEXT,
    duration_s  REAL,
    views       INTEGER,
    upload_date TEXT,
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
    hook_title    TEXT,
    revised_at    TEXT,
    parent_clip_id INTEGER REFERENCES clips(id),
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# Columns added after the initial release; ALTER is idempotent via pragma check.
_CLIP_MIGRATIONS = [
    ("engine", "TEXT NOT NULL DEFAULT 'single_shot'"),
    ("revised_at", "TEXT"),
    ("parent_clip_id", "INTEGER REFERENCES clips(id)"),
    ("hook_title", "TEXT"),
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
    """Add clips columns introduced after the baseline, if missing."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    for col, decl in _CLIP_MIGRATIONS:
        if col not in have:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {decl}")


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
                avatar_url: str | None, db_path: Path | str | None = None) -> int:
    existing = query_one(
        "SELECT id FROM channels WHERE platform_channel_id=?", (platform_channel_id,), db_path
    )
    if existing:
        execute(
            """UPDATE channels SET title=?, topic=?, subtopic=?, subs=?, video_count=?,
                      joined_year=?, avatar_url=? WHERE id=?""",
            (title, topic, subtopic, subs, video_count, joined_year, avatar_url, existing["id"]),
            db_path,
        )
        return existing["id"]
    return execute(
        """INSERT INTO channels(platform_channel_id, title, topic, subtopic, subs, video_count,
                                joined_year, avatar_url, added_at)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (platform_channel_id, title, topic, subtopic, subs, video_count, joined_year,
         avatar_url, now_iso()),
        db_path,
    )


def all_channels(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM channels ORDER BY topic, id", (), db_path)


def channels_by_topic(topic: str, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    return query("SELECT * FROM channels WHERE topic=? ORDER BY id", (topic,), db_path)


def channel_count(db_path: Path | str | None = None) -> int:
    row = query_one("SELECT COUNT(*) AS c FROM channels", (), db_path)
    return int(row["c"]) if row else 0


# --- videos -----------------------------------------------------------------
def insert_video(video_id: str, channel_id: int, title: str | None, duration_s: float | None,
                 views: int | None, upload_date: str | None, db_path: Path | str | None = None) -> None:
    execute(
        """INSERT OR REPLACE INTO videos(video_id, channel_id, title, duration_s, views,
                                         upload_date, status, used_at)
           VALUES(?,?,?,?,?,?,'pending',NULL)""",
        (video_id, channel_id, title, duration_s, views, upload_date, ),
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


# --- clips ------------------------------------------------------------------
def add_clip(video_id: str, start_s: float, end_s: float, path: str | None, caption: str | None,
             model: str | None, prompt_source: str, engine: str = "single_shot",
             parent_clip_id: int | None = None, hook_title: str | None = None,
             db_path: Path | str | None = None) -> int:
    return execute(
        """INSERT INTO clips(video_id, start_s, end_s, path, caption, model,
                             prompt_source, engine, parent_clip_id, hook_title, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (video_id, start_s, end_s, path, caption, model, prompt_source,
         engine, parent_clip_id, hook_title, now_iso()),
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


# --- reset helpers (tests / crash recovery) ---------------------------------
def reset_run_state(db_path: Path | str | None = None) -> None:
    """Clear any in-flight job markers left behind by a crash (spec §11.7)."""
    execute("DELETE FROM state WHERE key='job_active'", (), db_path)
