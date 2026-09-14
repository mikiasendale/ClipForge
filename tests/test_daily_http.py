"""Daily job through the HTTP runner thread (acceptance §11.1/4/6/7-ish)."""
import time
import pytest
from fastapi.testclient import TestClient
from app import main as m
from app import db, selector, downloader, analyzer, editor_ai, config as cfg
from app.analyzer import Word, Window, Analysis


@pytest.fixture()
def wired(home, monkeypatch, synth_720p):
    cfg.DB_PATH = home / "data" / "clipforge.db"
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    cfg.get_config().raw.setdefault("job", {})["daily_quota"] = 2
    # seed rotation-eligible channels (football needs 3 for the count rule)
    cids = []
    for t, n in (("cats_silent", 2), ("football", 3), ("boxing", 3), ("cats_compilations", 2)):
        for i in range(n):
            cids.append(db.add_channel(f"UC{t}{i}", f"{t}{i}", t, None, 1000, 10, 2015, None))
    # stub the whole media + AI layer; rotation still picks real channel rows
    _vid = {"n": 0}
    def fake_list(row):
        return [{"video_id": f"v{row['title']}", "title": "T", "duration_s": 600,
                 "views": 5000, "upload_date": "20200101"}]
    monkeypatch.setattr(selector, "list_channel_videos", fake_list)
    monkeypatch.setattr(downloader, "download", lambda *a, **k: synth_720p)
    monkeypatch.setattr(analyzer, "analyze", lambda path, topic, mode, video_id=None:
        Analysis("speech", 600.0, [Word(1 + i, 1.5 + i, "goal") for i in range(300)],
                 [Window(2, 6, 1)]))
    def fake_openrouter(messages, model):
        return ('{"clips":[{"start_s":2.0,"end_s":6.0,"hook_title":"h","caption":"c","reason":"r"}]}',
                {}, 1.0)
    monkeypatch.setattr(editor_ai, "call_openrouter", fake_openrouter)
    client = TestClient(m.app)
    return client, home


def test_run_daily_via_http(wired):
    client, home = wired
    db.set_int_state("rotation_index", 0)
    r = client.post("/run")
    assert r.status_code == 200
    # wait for the background thread to finish
    for _ in range(120):
        s = client.get("/run/status").json()
        if not s["running"]:
            break
        time.sleep(0.5)
    s = client.get("/run/status").json()
    assert s["error"] is None
    assert s["clips_today"] >= 1
    # exactly one video used per rotation position so far; no dupes
    used = db.query("SELECT video_id, used_at FROM videos WHERE used_at IS NOT NULL")
    assert len(used) == len({r["video_id"] for r in used})
    assert db.get_state("job_active") == "0"


def test_second_run_advances_rotation(wired):
    client, home = wired
    db.set_int_state("rotation_index", 0)
    client.post("/run")
    for _ in range(120):
        if not client.get("/run/status").json()["running"]:
            break
        time.sleep(0.5)
    idx_after = db.get_int_state("rotation_index")
    assert idx_after >= 1
