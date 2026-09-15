"""Feature B — CapCut draft output mode (acceptance B5)."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import capcut_export as cc, config as cfg, db, job, downloader, analyzer, editor_ai, main as m
from app.analyzer import Word, Window, Analysis
import subprocess


@pytest.fixture()
def root(home):
    """A real, writable CapCut-style draft root wired into config."""
    root = home / "capcut_drafts"
    root.mkdir(parents=True, exist_ok=True)
    cfg.DB_PATH = home / "data" / "clipforge.db"
    c = cfg.get_config()
    c.raw.setdefault("capcut", {})["draft_root"] = str(root)
    c.raw.setdefault("clip", {})["face_tracking"] = "off"
    return root


@pytest.fixture()
def source_720p(home):
    out = home / "data" / "downloads" / "capvid.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=brown:s=1280x720:r=30:d=8",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(out)], capture_output=True, check=True, timeout=90)
    return out


def _seed_video(source, root):
    cid = db.add_channel("UCcap", "CapChan", "football", None, 100, 5, 2015, None)
    db.insert_video("capvid", cid, "Old Match", 600.0, 900, "20200101")
    return dict(db.query_one("SELECT * FROM channels WHERE id=?", (cid,)))


def _stub_editor_single(monkeypatch):
    monkeypatch.setattr(downloader, "download", lambda *a, **k: source_path_holder["p"])
    monkeypatch.setattr(analyzer, "analyze",
                        lambda p, topic, mode, video_id=None:
                        Analysis("speech", 600.0, [Word(i, i + 1, "goal") for i in range(300)],
                                 [Window(2, 6, 1)]))
    monkeypatch.setattr(analyzer, "cached_words", lambda vid: [Word(1, 2, "goal")])
    monkeypatch.setattr(editor_ai, "call_openrouter", lambda m, mo: (
        json.dumps({"clips": [{"start_s": 3.0, "end_s": 8.0, "hook_title": "H", "caption": "cap", "reason": "r"}]}),
        {}, 1.0))


source_path_holder = {}


def test_draft_root_requires_config_on_non_windows(home):
    # explicit non-auto path resolves; auto on non-windows yields guidance
    cfg.get_config().raw.setdefault("capcut", {})["draft_root"] = "auto"
    rep = cc.detect_draft_root()
    if not cc.is_windows():
        assert rep["available"] and rep["draft_root"] is None
        assert "config.yaml" in (rep["error"] or "")


def test_export_draft_creates_9x16_trims(root, source_720p):
    source_path_holder["p"] = source_720p
    _seed_video(source_720p, root)
    clips = [{"start_s": 1.0, "end_s": 6.0, "caption": "First", "hook_title": "H1"},
             {"start_s": 6.5, "end_s": 8.0, "caption": "Second", "hook_title": "H2"}]
    path = cc.export_draft({"video_id": "capvid", "title": "Old Match", "duration_s": 8.0,
                            "_clip_index": 0}, clips, src_path=source_720p)
    P = Path(path)
    assert P.is_dir() and P.parent == root
    assert P.name.startswith("ClipForge_") and "Old_Match" in P.name
    content = json.loads((P / "draft_content.json").read_text())
    assert content["canvas_config"]["width"] == 1080 and content["canvas_config"]["height"] == 1920
    vsegs = [s for t in content["tracks"] if t["type"] == "video" for s in t["segments"]]
    trims = [(s["source_timerange"]["start"] / 1e6, s["source_timerange"]["duration"] / 1e6)
             for s in vsegs]
    assert trims == [(1.0, 5.0), (6.5, 1.5)]   # trimmed in-timeline, source not pre-cut
    tsegs = [s for t in content["tracks"] if t["type"] == "text" for s in t["segments"]]
    assert len(tsegs) == 2                      # one caption per clip


def test_export_never_overwrites_and_no_orphans(root, source_720p):
    vr = {"video_id": "x", "title": "T", "duration_s": 8.0, "_clip_index": 0}
    clips = [{"start_s": 1.0, "end_s": 6.0, "caption": "c", "hook_title": "h"}]
    p1 = cc.export_draft(vr, clips, dest_name="ClipForge_fixed", src_path=source_720p)
    p2 = cc.export_draft(vr, clips, dest_name="ClipForge_fixed", src_path=source_720p)
    assert Path(p1).name != Path(p2).name and Path(p2).name == "ClipForge_fixed_2"
    assert cc.remove_draft(p1) and not Path(p1).exists()


def test_daily_job_capcut_writes_drafts_not_mp4(root, source_720p, monkeypatch):
    source_path_holder["p"] = source_720p
    ch = _seed_video(source_720p, root)
    _stub_editor_single(monkeypatch)
    monkeypatch.setattr(db, "clips_today", lambda db_path=None: 0)  # quota not yet hit
    c = cfg.get_config(); c.raw.setdefault("agent", {})["enabled"] = False
    paths = job.process_video("capvid", ch, None, lambda m: None, force_n=1, output_mode="capcut")
    assert paths and all(Path(p).is_dir() for p in paths)
    clip = db.query_one("SELECT * FROM clips WHERE video_id='capvid'")
    assert clip["render_mode"] == "capcut" and clip["path"] is None and clip["draft_path"]
    assert db.query_one("SELECT used_at FROM videos WHERE video_id='capvid'")["used_at"] is not None


def test_capcut_unavailable_disables_mode_but_mp4_unaffected(root, source_720p, monkeypatch):
    monkeypatch.setattr(cc, "CAPCUT_AVAILABLE", False)
    rep = cc.detect_draft_root()
    assert rep["available"] is False and "pip install failed" in rep["error"]
    # export_draft refuses
    with pytest.raises(RuntimeError):
        cc.export_draft({"video_id": "x"}, [{"start_s": 1, "end_s": 6}], src_path=source_720p)
    # but the mp4 pipeline still runs
    source_path_holder["p"] = source_720p
    ch = _seed_video(source_720p, root)
    _stub_editor_single(monkeypatch)
    cfg.get_config().raw.setdefault("agent", {})["enabled"] = False
    mp4 = job.process_video("capvid", ch, None, lambda m: None, force_n=1, output_mode="mp4")
    assert mp4 and Path(mp4[0]).is_file() and Path(mp4[0]).suffix == ".mp4"


def test_unwritable_root_falls_back_to_mp4(home, source_720p, monkeypatch):
    # point draft_root at a path whose parent is a regular file -> not writable
    blocker = home / "blocker.txt"
    blocker.write_text("x")
    cfg.DB_PATH = home / "data" / "clipforge.db"
    c = cfg.get_config()
    c.raw.setdefault("capcut", {})["draft_root"] = str(blocker / "nope")
    c.raw.setdefault("clip", {})["face_tracking"] = "off"
    source_path_holder["p"] = source_720p
    ch = _seed_video(source_720p, None)
    _stub_editor_single(monkeypatch)
    c.raw.setdefault("agent", {})["enabled"] = False
    # capcut requested but unwritable -> logs a fall back and still produces mp4
    logs = []
    mp4 = job.process_video("capvid", ch, None, lambda m: logs.append(m), force_n=1, output_mode="capcut")
    assert mp4 and Path(mp4[0]).suffix == ".mp4"
    assert any("falling back to mp4" in l for l in logs)
    assert db.query_one("SELECT render_mode FROM clips WHERE video_id='capvid'")["render_mode"] == "mp4"


def test_api_clips_capcut_503_when_unavailable(root, source_720p, monkeypatch):
    monkeypatch.setattr(cc, "CAPCUT_AVAILABLE", False)
    _seed_video(source_720p, root)
    monkeypatch.setattr(downloader, "resolve_source", lambda vid, dest_dir=None: source_720p if vid == "capvid" else None)
    client = TestClient(m.app)
    r = client.post("/api/clips", json={"video_id": "capvid", "start_s": 1.0, "end_s": 6.0,
                                        "caption": "x", "mode": "new", "render_mode": "capcut"})
    assert r.status_code == 503 and "pip install failed" in r.json()["detail"]


def test_api_clips_capcut_replace_regenerates_no_orphan(root, source_720p, monkeypatch):
    _seed_video(source_720p, root)
    monkeypatch.setattr(downloader, "resolve_source", lambda vid, dest_dir=None: source_720p if vid == "capvid" else None)
    client = TestClient(m.app)
    base = client.post("/api/clips", json={"video_id": "capvid", "start_s": 1.0, "end_s": 6.0,
                                           "caption": "orig", "mode": "new", "render_mode": "capcut"}).json()
    old_path = base["clip"]["draft_path"]
    assert Path(old_path).is_dir()
    rev = client.post("/api/clips", json={"video_id": "capvid", "start_s": 0.5, "end_s": 7.5,
                                          "caption": "revised", "mode": "replace", "clip_id": base["clip"]["id"]})
    assert rev.status_code == 200
    new = rev.json()["clip"]
    assert new["render_mode"] == "capcut" and new["parent_clip_id"] == base["clip"]["id"]
    assert not Path(old_path).exists()          # orphan draft removed
    assert Path(new["draft_path"]).is_dir()     # fresh draft present
    assert db.get_clip(base["clip"]["id"])["revised_at"] is not None


def test_auto_output_flag_sets_state_and_passes(monkeypatch, home):
    import app.main as m
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    captured = {}
    monkeypatch.setattr(job, "daily_job", lambda log, output_mode=None: captured.setdefault("out", output_mode) or [])
    rc = m._run_auto("capcut")
    assert rc == 0 and captured["out"] == "capcut"
    assert db.get_state("output_mode") == "capcut"
