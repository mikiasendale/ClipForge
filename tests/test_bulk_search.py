"""Bulk search curation (no network — search results mocked)."""
from app import bulk_search as bs


def _e(id, title, dur, views, url=None):
    return {"id": id, "title": title, "duration": dur, "view_count": views,
            "url": url or f"https://www.youtube.com/watch?v={id}", "channel": "C"}


def test_similarity():
    assert bs.similarity("Steve Jobs Interview", "steve jobs interview") == 1.0
    assert bs.similarity("Steve Jobs The Lost Interview 1987",
                         "Steve Jobs The Lost Interview") >= 0.8
    assert bs.similarity("Steve Jobs Interview", "Cooking Pasta at Home") < 0.3


def test_filter_duration_above_one_hour():
    entries = [_e("long1", "A", 4340, 10), _e("short", "B", 100, 999),
               _e("edge", "C", 3600, 5), _e("unknown", "D", None, 7)]
    kept, dropped = bs.filter_duration(entries, 3600)
    assert [k["id"] for k in kept] == ["long1"]     # strictly ABOVE an hour
    assert dropped == 3                              # short + edge(=1h not >) + unknown


def test_dedupe_titles_matching_and_similar():
    entries = [
        _e("dup", "Steve Jobs The Lost Interview", 4340, 999),
        _e("orig", "Steve Jobs   The Lost Interview", 4200, 5000),  # same title, higher views
        _e("near", "Steve Jobs The Lost Interview 1987", 4100, 400),  # similar
        _e("other", "Steve Jobs MIT 1992 Talk", 4374, 300),           # distinct
    ]
    kept = bs.dedupe_titles(entries)
    ids = [k["id"] for k in kept]
    assert ids == ["orig", "other"]   # dup (lower views) + near (similar) dropped


def test_dedupe_durations_same_length_kept_once():
    entries = [_e("a", "A", 4340, 200), _e("b", "B", 4340, 900),
               _e("c", "C", 4374, 100)]
    kept = bs.dedupe_durations(entries)
    ids = [k["id"] for k in kept]
    assert "b" in ids and "a" not in ids and "c" in ids   # higher-viewed wins per length


def test_curate_enforces_all_rules():
    sample = [
        _e("b", "Steve Jobs The Lost Interview", 4340, 999),
        _e("a", "Steve Jobs   The Lost Interview", 4340, 274162),  # dup title+length
        _e("c", "Steve Jobs MIT 1992 Talk", 4374, 455142),
        _e("d", "short clip", 100, 6410682),        # too short (high views!)
        _e("e", "Steve Jobs Interview 1981", 1180, 3065636),  # too short
        _e("f", "unknown length", None, 1),          # unknown duration
    ]
    kept = bs.curate(sample)
    ids = [k["id"] for k in kept]
    assert ids == ["c", "a"]            # c first (higher views), a kept over b
    assert "d" not in ids and "e" not in ids   # short ones excluded despite views
    durations = [k["duration"] for k in kept]
    assert len(durations) == len(set(durations))  # unique lengths guaranteed
    titles = [bs.norm_title(k["title"]) for k in kept]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            assert bs.similarity(titles[i], titles[j]) < 0.85  # no similar titles


def test_curate_caps_at_limit():
    names = ["Steve Jobs on Design", "Jobs on Recruiting", "The Lost Interview",
             "MIT 1992 Talk", "Secrets of Life", "Jobs at Stanford", "NeXT Era",
             "Pixar Years", "Apple Keynote 1997", "Jobs on Failure", "Rollerblade Day"]
    entries = [_e(f"v{i}", f"{n} (full)", 3601 + i, 100 + i) for i, n in enumerate(names)]
    kept = bs.curate(entries, limit=10)
    assert len(kept) == 10


def test_save_and_load_roundtrip(tmp_path):
    entries = [_e("a", "Steve Jobs The Lost Interview", 4340, 274162)]
    paths = bs.save(entries, "steve jobs interview", tmp_path / "bulk.json")
    assert paths["json"].is_file() and paths["txt"].is_file()
    txt = (tmp_path / "bulk.txt").read_text().strip().splitlines()
    assert txt == ["https://www.youtube.com/watch?v=a"]
    loaded = bs.load(tmp_path / "bulk.json")
    assert loaded["count"] == 1
    assert loaded["videos"][0]["title"] == "Steve Jobs The Lost Interview"
    assert loaded["rules"]["unique_durations"] is True


def test_load_missing_file(tmp_path):
    assert bs.load(tmp_path / "nope.json") == {}
