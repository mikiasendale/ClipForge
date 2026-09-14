"""DB layer + prompts: schema, no-reuse invariant, state, prompt fill."""
from app import db
from app import prompts


def test_schema_and_crud(home):
    dbp = home / "data" / "clipforge.db"
    cid = db.add_channel("UC1", "Chan", "football", None, 5000, 120, 2010, "http://x", dbp)
    assert cid > 0
    db.insert_video("v1", cid, "Title", 600, 100, "20200101", dbp)
    assert db.query_one("SELECT * FROM videos WHERE video_id='v1'", (), dbp)["status"] == "pending"
    db.mark_video_used("v1", dbp)
    v = db.query_one("SELECT * FROM videos WHERE video_id='v1'", (), dbp)
    assert v["used_at"] is not None and v["status"] == "done"


def test_used_at_never_reuse(home):
    dbp = home / "data" / "clipforge.db"
    cid = db.add_channel("UC1", "C", "football", None, None, None, None, None, dbp)
    db.insert_video("dup", cid, "T", 300, 10, None, dbp)
    db.mark_video_used("dup", dbp)
    assert "dup" in db.known_video_ids(dbp)


def test_state_roundtrip(home):
    dbp = home / "data" / "clipforge.db"
    db.set_int_state("rotation_index", 3, dbp)
    assert db.get_int_state("rotation_index", 0, dbp) == 3
    db.set_int_state("rotation_index", 4, dbp)
    assert db.get_int_state("rotation_index", 0, dbp) == 4


def test_clips_today_counting(home):
    dbp = home / "data" / "clipforge.db"
    cid = db.add_channel("UC1", "C", "football", None, None, None, None, None, dbp)
    db.insert_video("v", cid, "T", 300, 1, None, dbp)
    assert db.clips_today(dbp) == 0
    db.add_clip("v", 1, 30, "p", "cap", "ai", "default", dbp)
    assert db.clips_today(dbp) == 1


def test_format_prompt_preserves_other_braces():
    out = prompts.format_prompt("give {N} clips in {json} form", 3)
    assert out == "give 3 clips in {json} form"
