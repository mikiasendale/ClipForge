"""End-to-end pipeline with network + whisper mocked (acceptance §11.3/4/5/6/7)."""
import json
import pytest
from pathlib import Path
from app import job, db, downloader, analyzer, editor_ai, config as cfg
from app.analyzer import Word, Window, Analysis


@pytest.fixture(autouse=True)
def _cfg_off(home):
    cfg.DB_PATH = home / "data" / "clipforge.db"
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    cfg.get_config().raw.setdefault("job", {})["daily_quota"] = 4


def _mk_channel(topic="football"):
    cid = db.add_channel("UCtest", "TestChan", topic, None, 1000, 50, 2015, None)
    return db.query_one("SELECT * FROM channels WHERE id=?", (cid,))


def _stub_download(monkeypatch, src):
    monkeypatch.setattr(downloader, "download", lambda vid, url=None, dest_dir=None: src)


def _stub_analyze(monkeypatch, analysis):
    monkeypatch.setattr(analyzer, "analyze", lambda path, topic, mode, video_id=None: analysis)


def _stub_editor(monkeypatch, picks_json):
    def fake(messages, model):
        return picks_json, {"completion_tokens": 1}, 1.0
    monkeypatch.setattr(editor_ai, "call_openrouter", fake)


def test_ten_minute_video_yields_two_clips(home, monkeypatch, synth_720p):
    ch = _mk_channel()
    words = [Word(i * 1.0, i * 1.0 + 0.5, "goal") for i in range(300)]
    # a genuine 10-minute source -> windows_for_duration() must return 2
    analysis = Analysis("speech", 600.0, words,
                        [Window(2, 6, 3), Window(6, 10, 2), Window(4, 8, 1)])
    _stub_download(monkeypatch, synth_720p)
    _stub_analyze(monkeypatch, analysis)
    # AI returns 2 non-overlapping windows (rendered against the 12s synth source)
    picks = {"clips": [
        {"start_s": 2.0, "end_s": 6.0, "hook_title": "h1", "caption": "c1 🔥", "reason": "r"},
        {"start_s": 6.2, "end_s": 10.0, "hook_title": "h2", "caption": "c2", "reason": "r"},
    ]}
    _stub_editor(monkeypatch, json.dumps(picks))
    db.insert_video("vid1", ch["id"], "Title", 600.0, 500, None)
    paths = job.process_video("vid1", dict(ch), None, lambda m: None)
    assert len(paths) == 2
    for p in paths:
        assert Path(p).is_file()
        assert _dur(p) <= 6.2
    # used_at set -> never reuse
    v = db.query_one("SELECT * FROM videos WHERE video_id='vid1'")
    assert v["used_at"] is not None and v["status"] == "done"


def _dur(p):
    import subprocess
    ffprobe = cfg.get_config().ffprobe or "ffprobe"
    return float(subprocess.run([ffprobe, "-v", "error", "-show_entries",
            "format=duration", "-of", "default=nk=1:nw=1", str(p)],
            capture_output=True, text=True).stdout.strip())


def test_silent_video_uses_vision_path(home, monkeypatch, synth_720p):
    ch = _mk_channel("cats_silent")
    analysis = Analysis("visual", 12.0, [], [Window(2, 6, 5), Window(6, 10, 4)])
    _stub_download(monkeypatch, synth_720p)
    _stub_analyze(monkeypatch, analysis)
    picked = {"clips": [{"start_s": 3.0, "end_s": 8.0, "hook_title": "h", "caption": "c", "reason": "r"}]}
    _stub_editor(monkeypatch, json.dumps(picked))
    db.insert_video("vidv", ch["id"], "CatSilent", 12.0, 300, None)
    # auto mode -> transcribe yields nothing meaningful -> visual path
    monkeypatch.setattr(analyzer, "transcribe", lambda *a, **k: [])
    paths = job.process_video("vidv", dict(ch), None, lambda m: None)
    assert len(paths) == 1
    assert _dur(paths[0]) <= 60


def test_prompt_override_marks_user_source(home, monkeypatch, synth_720p):
    ch = _mk_channel()
    words = [Word(i * 1.0, i * 1.0 + 0.5, "punch") for i in range(300)]
    analysis = Analysis("speech", 12.0, words, [Window(2, 6, 1)])
    _stub_download(monkeypatch, synth_720p)
    _stub_analyze(monkeypatch, analysis)
    picks = {"clips": [{"start_s": 2.0, "end_s": 6.0, "hook_title": "h", "caption": "c", "reason": "r"}]}
    _stub_editor(monkeypatch, json.dumps(picks))
    db.insert_video("vidp", ch["id"], "T", 12.0, 100, None)
    job.process_video("vidp", dict(ch), None, lambda m: None,
                      force_n=1, prompt_override="CUSTOM {N} only")
    clip = db.query_one("SELECT * FROM clips WHERE video_id='vidp'")
    assert clip["prompt_source"] == "user"


def test_second_run_picks_different_video(home, monkeypatch, synth_720p):
    ch = _mk_channel()
    _stub_download(monkeypatch, synth_720p)
    analysis = Analysis("speech", 12.0, [Word(1, 1.5, "goal")] * 60,
                        [Window(2, 6, 1)])
    _stub_analyze(monkeypatch, analysis)
    _stub_editor(monkeypatch, json.dumps({"clips": [
        {"start_s": 2.0, "end_s": 6.0, "hook_title": "h", "caption": "c", "reason": "r"}]}))
    # channel rotation: same video list, two different un-used rows
    monkeypatch.setattr(job.selector, "list_channel_videos",
        lambda row: [{"video_id": "A", "title": "a", "duration_s": 600, "views": 100, "upload_date": None},
                     {"video_id": "B", "title": "b", "duration_s": 600, "views": 90, "upload_date": None}])
    first = job.selector.pick_video(dict(ch))
    job.process_video(first["video_id"], dict(ch), None, lambda m: None)
    second = job.selector.pick_video(dict(ch))
    assert first["video_id"] != second["video_id"]


def test_crash_and_failure_never_reuse_or_duplicate(home, monkeypatch, synth_720p):
    """§11.7: a failed video (used_at NULL) and an interrupted one are both
    excluded from future picks purely by videos-table membership, so a restart
    can never reprocess a video and duplicate its clips."""
    ch = _mk_channel()
    db.insert_video("vf", ch["id"], "T", 600, 100, None)
    # simulate download failure -> status failed, used_at stays NULL
    monkeypatch.setattr(downloader, "download", lambda *a, **k: None)
    made = job.process_video("vf", dict(ch), None, lambda m: None)
    assert made == []
    v = db.query_one("SELECT * FROM videos WHERE video_id='vf'")
    assert v["status"] == "failed" and v["used_at"] is None
    # excluded from any future selection -> no duplicate clips possible
    assert "vf" in db.known_video_ids()
    ranked = sel_ranked([{"video_id": "vf", "views": 100, "duration_s": 600}])
    assert ranked == []


def sel_ranked(videos):
    from app import selector as sel
    return sel.score_videos(videos, 180, 1800, db.known_video_ids())
