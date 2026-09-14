"""Cutter: real ffmpeg renders (9:16 target, duration, caption burn, crop geometry)."""
import json
import subprocess
import pytest
from app import cutter
from app import config as cfg
from app.analyzer import Word


def _probe_dims(path):
    ffprobe = cfg.get_config().ffprobe or "ffprobe"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return st["width"], st["height"]


def _probe_duration(path):
    ffprobe = cfg.get_config().ffprobe or "ffprobe"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True).stdout.strip()
    return float(out)


@pytest.mark.skipif(cfg.get_config().ffmpeg is None, reason="ffmpeg not available")
def test_crop_geometry_720p(home):
    dims = cutter.SourceDims(1280, 720)
    cw = cutter.crop_width(dims)
    assert cw == 404                       # spec §8: 404 from 1280x720
    assert cutter.center_crop(dims, cw) == 438


@pytest.mark.skipif(cfg.get_config().ffmpeg is None, reason="ffmpeg not available")
def test_build_ass_words_per_line(home):
    words = [Word(i, i + 0.4, w) for i, w in enumerate("the quick brown fox jumps".split())]
    a = home / "c.ass"
    assert cutter.build_ass(words, 0, 5, a, "Arial Black", 54, 3, 1080, 1920)
    txt = a.read_text()
    assert "PlayResX: 1080" in txt and "THE QUICK BROWN" in txt
    assert "BROWN FOX JUMPS" not in txt   # wraps at 3 words


@pytest.mark.skipif(cfg.get_config().ffmpeg is None, reason="ffmpeg not available")
def test_render_9x16_duration_and_center_crop(home, synth_720p):
    cfg.get_config().raw.setdefault("clip", {})["face_tracking"] = "off"
    out = cutter.output_path_for("football", "Chan", "Title", 1)
    # 5s window on an 8s clip, <=60s, captions burned
    words = [Word(1 + i * 0.5, 1.4 + i * 0.5, "goal") for i in range(6)]
    path = cutter.render_clip(synth_720p, 1.0, 6.0, out, words=words)
    assert path and Path_exists(path)
    w, h = _probe_dims(path)
    assert (w, h) == (1080, 1920)
    assert 4.0 <= _probe_duration(path) <= 6.5


def Path_exists(p):
    from pathlib import Path
    return Path(p).is_file()
