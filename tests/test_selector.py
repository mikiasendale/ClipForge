"""Selector math: windows, rotation, scoring (no network)."""
import pytest
from app import selector as sel


@pytest.mark.parametrize("dur,expected", [
    (180, 1), (299, 1), (300, 1), (330, 1), (600, 2), (660, 2),
    (900, 3), (1200, 4), (1800, 4), (5400, 4), (0, 1),
])
def test_windows_for_duration(dur, expected):
    assert sel.windows_for_duration(dur) == expected


def test_clamp_windows():
    assert sel.clamp_windows(4, 2) == 2
    assert sel.clamp_windows(2, 5) == 2
    assert sel.clamp_windows(3, 0) == 0


def test_rotation_interleaves_topics():
    grouped = {
        "cats_silent": [{"id": 1}, {"id": 2}],
        "football": [{"id": 3}, {"id": 4}, {"id": 5}],
        "boxing": [{"id": 6}],
        "cats_compilations": [{"id": 7}, {"id": 8}],
    }
    order = ["cats_silent", "football", "boxing", "cats_compilations"]
    rot = sel.build_rotation(grouped, order)
    ids = [c["id"] for c in rot]
    # leading sequence follows topic rotation, round-robin within pools
    assert ids[:4] == [1, 3, 6, 7]
    assert len(ids) == 8


def test_score_prefers_mid_high_and_filters():
    videos = [
        {"video_id": "a", "views": 5_000_000, "duration_s": 600},   # viral outlier
        {"video_id": "b", "views": 120_000, "duration_s": 600},     # mid-high
        {"video_id": "c", "views": 10, "duration_s": 600},          # dud
        {"video_id": "d", "views": 300_000, "duration_s": 60},      # too short -> excluded
        {"video_id": "e", "views": 300_000, "duration_s": 5000},    # too long -> excluded
    ]
    ranked = sel.score_videos(videos, 180, 1800, exclude_ids={"a"})
    ids = [v["video_id"] for v in ranked]
    assert "d" not in ids and "e" not in ids   # duration filter
    assert "a" not in ids                       # exclusion
    assert ids[0] == "b"                        # mid-high wins over dud 'c'


def test_score_excludes_known(home):
    from app import db
    cid = db.add_channel("UCx", "Chan", "football", None, 10, 1, 2015, None)
    db.insert_video("a", cid, "A", 600, 100, None)
    known = db.known_video_ids()
    ranked = sel.score_videos(
        [{"video_id": "a", "views": 100, "duration_s": 600},
         {"video_id": "b", "views": 200, "duration_s": 600}],
        180, 1800, known)
    assert [v["video_id"] for v in ranked] == ["b"]
