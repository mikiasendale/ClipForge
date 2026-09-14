"""Feature C — review/edit + Range streaming (acceptance C2 media/clip tests)."""
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import main as m, db, config as cfg, downloader


@pytest.fixture()
def client_with_source(home, monkeypatch, synth_720p):
    """DB + a real downloaded source (30s) resolvable by id, face tracking off."""
    cfg.DB_PATH = home / "data" / "clipforge.db"
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    # make downloads dir resolve this file for our fake video id
    target = cfg.DOWNLOADS_DIR / "vidX.mp4"
    target.write_bytes(Path(synth_720p).read_bytes())
    cid = db.add_channel("UCv", "Chan", "football", None, 100, 5, 2015, None)
    # declare a long source duration so 60s windows are within bounds; the real
    # 12s synth file is simply truncated by ffmpeg at EOF during render.
    db.insert_video("vidX", cid, "Title", 300.0, 500, None)
    db.mark_video_used("vidX")
    monkeypatch.setattr(downloader, "resolve_source",
                        lambda vid, dest_dir=None: target if vid == "vidX" else None)
    return TestClient(m.app)


def test_media_source_range_206(client_with_source):
    total = (cfg.DOWNLOADS_DIR / "vidX.mp4").stat().st_size
    r = client_with_source.get("/media/source/vidX", headers={"Range": "bytes=100-299"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 100-299/{total}"
    assert r.headers["accept-ranges"] == "bytes"
    assert len(r.content) == 200


def test_media_source_full_and_bad_range(client_with_source):
    r = client_with_source.get("/media/source/vidX")
    assert r.status_code == 200 and r.headers["accept-ranges"] == "bytes"
    bad = client_with_source.get("/media/source/vidX", headers={"Range": "bytes=99999999-"})
    assert bad.status_code == 416


def test_manual_clip_creation_respects_60s(client_with_source):
    ok = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 1.0, "end_s": 61.0, "caption": "c", "mode": "new"})
    # 60s is allowed
    assert ok.status_code == 200
    clip = ok.json()["clip"]
    assert clip["engine"] == "manual"
    assert Path(clip["path"]).is_file()
    w, h = _dims(clip["path"])
    assert (w, h) == (1080, 1920)
    # 61s window rejected
    too_long = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 0.5, "end_s": 65.0, "caption": "x", "mode": "new"})
    assert too_long.status_code == 400 and "60" in too_long.json()["detail"]
    # <5s rejected
    short = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 5.0, "end_s": 7.0, "caption": "x", "mode": "new"})
    assert short.status_code == 400 and "5s" in short.json()["detail"]


def test_adjust_out_point_replaces_and_supersedes(client_with_source):
    base = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 1.0, "end_s": 61.0, "caption": "orig", "mode": "new"}).json()["clip"]
    # shrink out-point by 10s -> replace
    rev = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 1.0, "end_s": 51.0, "caption": "shorter",
        "mode": "replace", "clip_id": base["id"]})
    assert rev.status_code == 200
    new = rev.json()["clip"]
    assert new["parent_clip_id"] == base["id"] and new["engine"] == "review"
    # old row marked revised; gallery shows ONLY the new version
    assert db.get_clip(base["id"])["revised_at"] is not None
    active = [c["id"] for c in db.current_clips() if c["video_id"] == "vidX"]
    assert base["id"] not in active and new["id"] in active
    page = client_with_source.get("/review").text
    assert "shorter" in page and "orig" not in page


def test_adjust_overlap_rejected(client_with_source):
    client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 5.0, "end_s": 20.0, "caption": "a", "mode": "new"})
    dup = client_with_source.post("/api/clips", json={
        "video_id": "vidX", "start_s": 10.0, "end_s": 25.0, "caption": "b", "mode": "new"})
    assert dup.status_code == 400 and "overlap" in dup.json()["detail"].lower()


def _dims(path):
    ffprobe = cfg.get_config().ffprobe or "ffprobe"
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)
