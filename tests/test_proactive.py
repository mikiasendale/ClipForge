"""Feature P0/P1/P2 acceptance tests (spec §3.8)."""
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from fastapi.testclient import TestClient
from app import (config as cfg, db, job, downloader, analyzer, editor_ai,
                 prestaging, scheduler as sched_mod, notifier, events, suggest,
                 main as m, cutter)
from app.analyzer import Word, Window, Analysis


def _mk_channel(topic="football", active=1, subs=1000):
    cid = db.add_channel(f"UC{topic}{subs}{active}", f"Chan{subs}", topic, None,
                         subs, 10, 2015, None, is_active=active)
    return db.query_one("SELECT * FROM channels WHERE id=?", (cid,))


# --- 3.1 scheduler ----------------------------------------------------------
def test_scheduler_registers_daily_job_and_next_run(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    fired = {"n": 0}
    sched_mod.configure(daily=lambda: fired.__setitem__("n", fired["n"] + 1),
                        prestage=lambda: None)
    s = sched_mod.AppScheduler()

    async def scenario():
        s.start()
        assert s.sched.get_job("clipforge_daily") is not None
        assert s.next_run_dt() is not None
        s.shutdown()
    asyncio.run(scenario())


def test_catch_up_runs_once_when_missed(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    c = cfg.get_config()
    c.raw.setdefault("schedule", {})["enabled"] = True
    c.raw["schedule"]["catch_up"] = True
    c.raw["schedule"]["time"] = "00:00"   # already passed today
    db.set_state("last_run_date", "")      # not run today
    fired = {"n": 0}
    s = sched_mod.AppScheduler()
    sched_mod.configure(daily=lambda: fired.__setitem__("n", fired["n"] + 1), prestage=lambda: None)
    s.catch_up()
    assert fired["n"] == 1
    s.catch_up()                            # second boot same day -> no double
    assert fired["n"] == 1


def test_schedule_disabled_no_jobs(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    cfg.get_config().raw.setdefault("schedule", {})["enabled"] = False
    s = sched_mod.AppScheduler()
    sched_mod.configure(daily=lambda: None, prestage=lambda: None)
    async def scenario():
        s.start()
        assert s.sched.get_job("clipforge_daily") is None
        s.shutdown()
    asyncio.run(scenario())


# --- 3.2 pre-staging + morning consumption ----------------------------------
def test_prestage_then_morning_skips_download(home, monkeypatch, synth_720p):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    ch = _mk_channel()
    monkeypatch.setattr(prestaging.selector, "ordered_rotation", lambda: [dict(ch)])
    monkeypatch.setattr(job.selector, "list_channel_videos",
        lambda row: [{"video_id": "sv1", "title": "T", "duration_s": 600,
                      "views": 500, "upload_date": "20200101", "thumbnail": None}])
    monkeypatch.setattr(downloader, "download", lambda vid, url=None, dest_dir=None: synth_720p)
    monkeypatch.setattr(analyzer, "analyze",
        lambda p, topic, mode, video_id=None: Analysis("speech", 600.0,
            [Word(i, i + 1, "goal") for i in range(300)], [Window(2, 6, 1)]))
    monkeypatch.setattr(analyzer, "cached_words", lambda vid: [Word(1, 2, "goal")])
    monkeypatch.setattr(editor_ai, "call_openrouter", lambda mm, mo: (
        json.dumps({"clips": [{"start_s": 2.0, "end_s": 6.0, "hook_title": "h", "caption": "c", "reason": "r"}]}), {}, 1.0))
    cfg.get_config().raw.setdefault("agent", {})["enabled"] = False

    staged = prestaging.prestage_once(lambda m: None)
    assert staged == 1
    assert db.staged_for_channel(ch["id"]) is not None

    # morning: staged consumption must NOT call download again
    calls = {"dl": 0}
    def boom_dl(*a, **k):
        calls["dl"] += 1
        raise AssertionError("download must be skipped for prestaged")
    monkeypatch.setattr(downloader, "download", boom_dl)
    monkeypatch.setattr(downloader, "resolve_source", lambda vid, dest_dir=None: synth_720p)
    logs = []
    paths = job.process_video("sv1", dict(ch), None, lambda x: logs.append(x),
                              force_n=1, output_mode="mp4", staged=True)
    assert paths and calls["dl"] == 0
    assert any("PRESTAGED" in l for l in logs)


def test_prestage_backlog_cap_one_per_slot(home, monkeypatch, synth_720p):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    ch = _mk_channel()
    monkeypatch.setattr(job.selector, "list_channel_videos",
        lambda row: [{"video_id": f"v{i}", "title": "T", "duration_s": 600,
                      "views": 500, "upload_date": "20200101", "thumbnail": None} for i in range(5)])
    monkeypatch.setattr(downloader, "download", lambda vid, url=None, dest_dir=None: synth_720p)
    monkeypatch.setattr(analyzer, "analyze",
        lambda p, topic, mode, video_id=None: Analysis("speech", 600.0, [], [Window(2, 6, 1)]))
    monkeypatch.setattr(prestaging.selector, "ordered_rotation", lambda: [dict(ch)])
    prestaging.prestage_once(lambda m: None)
    prestaging.prestage_once(lambda m: None)   # second pass must not double-stage the slot
    staged = [r for r in db.staged_videos() if r["channel_id"] == ch["id"]]
    assert len(staged) == 1


# --- 3.4 SSE ----------------------------------------------------------------
def test_sse_subscriber_receives_events():
    sub = events.bus.subscribe()
    events.job_progress("render", "clip 3", 0.5)
    msg = sub.q.get_nowait()
    assert msg["event"] == "job_progress" and msg["data"]["stage"] == "render"
    events.bus.unsubscribe(sub)


def test_events_endpoint_registered():
    paths = {r.path for r in m.app.routes}
    assert "/events" in paths
    # the SSE generator + subscriber plumbing is exercised in
    # test_sse_subscriber_receives_events; an infinite stream is not read here
    # to avoid deadlocking the synchronous TestClient.


# --- 3.5 notifications dedupe ----------------------------------------------
def test_notification_dedupe_within_window(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    id1 = notifier.notify("health", "warn", "t", "b", dedupe_key="k", within_hours=24)
    id2 = notifier.notify("health", "warn", "t", "b", dedupe_key="k", within_hours=24)
    assert id1 is not None and id2 is None
    assert len(db.list_notifications()) == 1


# --- 3.6/3.7 suggestion swap ------------------------------------------------
def test_yield_suggestion_and_swap(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    active = _mk_channel("football", active=1, subs=100)
    alt = _mk_channel("football", active=0, subs=99999)   # inactive candidate
    # 4 used videos, only 1 clip -> ratio 0.25 < 0.34
    for i in range(4):
        db.insert_video(f"v{i}", active["id"], "T", 600, 100, "20200101")
        db.mark_video_used(f"v{i}")
    db.add_clip("v0", 1, 5, "p", "c", "ai", "default", engine="agent")
    assert suggest.yield_rule() is True
    notif = [n for n in db.list_notifications() if n["kind"] == "suggestion"][0]
    actions = json.loads(notif["actions_json"])
    swap = [a for a in actions if a["action"] == "swap_channel"][0]
    assert swap["args"]["activate_id"] == alt["id"]
    # perform swap
    client = TestClient(m.app)
    r = client.post("/api/actions/swap_channel", json={"args": swap["args"], "confirm": True})
    assert r.status_code == 200
    assert db.query_one("SELECT is_active FROM channels WHERE id=?", (active["id"],))["is_active"] == 0
    assert db.query_one("SELECT is_active FROM channels WHERE id=?", (alt["id"],))["is_active"] == 1


# --- 3.3 review: discard + fill + quota not refunded ------------------------
def test_discard_keeps_quota_and_fill_from_staged(home, monkeypatch, synth_720p):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    cfg.get_config().raw.setdefault("job", {})["daily_quota"] = 4
    ch = _mk_channel()
    db.insert_video("c1", ch["id"], "T", 600, 100, "20200101")
    out = cutter.output_path_for("football", "Chan", "T", 0)
    p = cutter.render_clip(synth_720p, 1.0, 5.0, out, words=None)
    cid = db.add_clip("c1", 1, 5, str(p), "cap", "agent", "default", engine="agent")
    db.mark_video_used("c1")
    client = TestClient(m.app)
    # discard without confirm -> 400
    assert client.post(f"/api/clips/{cid}/discard", json={}).status_code == 400
    # with confirm -> file gone, status discarded, video still used (quota not refunded)
    r = client.post(f"/api/clips/{cid}/discard", json={"confirm": True})
    assert r.status_code == 200 and not Path(p).exists()
    assert db.get_clip(cid)["status"] == "discarded"
    assert db.query_one("SELECT used_at FROM videos WHERE video_id='c1'")["used_at"] is not None
    # a staged video can fill a slot (produces an extra clip, video not reused)
    db.insert_video("stg", ch["id"], "Staged", 600, 100, "20200101")
    db.mark_video_staged("stg")
    monkeypatch.setattr(downloader, "resolve_source", lambda vid, dest_dir=None: synth_720p if vid == "stg" else None)
    monkeypatch.setattr(analyzer, "cached_words", lambda vid: [Word(1, 2, "goal")])
    monkeypatch.setattr(analyzer, "analyze", lambda p, topic, mode, video_id=None:
        Analysis("speech", 600.0, [Word(i, i + 1, "goal") for i in range(300)], [Window(2, 6, 1)]))
    monkeypatch.setattr(editor_ai, "call_openrouter", lambda mm, mo: (
        json.dumps({"clips": [{"start_s": 2.0, "end_s": 6.0, "hook_title": "h", "caption": "c", "reason": "r"}]}), {}, 1.0))
    cfg.get_config().raw.setdefault("agent", {})["enabled"] = False
    made = job.process_video("stg", dict(ch), None, lambda x: None, force_n=1,
                             output_mode="mp4", staged=True)
    assert made and db.clips_today() <= 5
    assert db.query_one("SELECT used_at FROM videos WHERE video_id='stg'")["used_at"] is not None


# --- 3.7 auto-review --------------------------------------------------------
def test_auto_review_approves_old_clips(home, monkeypatch, synth_720p):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    c = cfg.get_config()
    c.raw.setdefault("review", {})["mode"] = "auto"
    c.raw["review"]["auto_approve_hours"] = 1
    c.raw.setdefault("clip", {})["face_tracking"] = "off"
    ch = _mk_channel()
    db.insert_video("old", ch["id"], "T", 600, 100, "20200101")
    out = cutter.output_path_for("football", "Chan", "T", 0)
    p = cutter.render_clip(synth_720p, 1.0, 5.0, out, words=None)
    cid = db.add_clip("old", 1, 5, str(p), "cap", "agent", "default", engine="agent")
    # backdate created_at so it's older than 1h
    db.execute("UPDATE clips SET created_at=? WHERE id=?",
               ((datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds"), cid))
    n = m.auto_review_pass(lambda x: None)
    assert n == 1
    assert db.get_clip(cid)["status"] == "approved"
    assert "approved" in Path(db.get_clip(cid)["path"]).parts  # moved to approved/
    log = (cfg.LOGS_DIR / "auto_actions.jsonl").read_text()
    assert "auto_approve" in log


# --- HARD RULE: destructive endpoints require confirm -----------------------
def test_destructive_requires_confirm(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    client = TestClient(m.app)
    # cleanup_sources without confirm -> 400
    r = client.post("/api/actions/cleanup_sources", json={"args": {"video_ids": ["x"]}})
    assert r.status_code == 400 and "confirm" in r.json()["detail"].lower()
    # update_ytdlp without confirm -> 400
    assert client.post("/api/actions/update_ytdlp", json={}).status_code == 400
    # non-destructive (dismiss) without confirm -> allowed
    assert client.post("/api/actions/dismiss", json={}).status_code == 200
