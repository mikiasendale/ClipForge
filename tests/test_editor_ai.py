"""Editor AI: robust JSON parse, validation, retry, heuristic fallback."""
import json
from pathlib import Path
from app import editor_ai as ea
from app.analyzer import Analysis, Word, Window


def test_extract_json_plain():
    assert ea.extract_json('{"clips":[{"start_s":1,"end_s":2}]}')["clips"]


def test_extract_json_fenced():
    txt = "Sure!\n```json\n{\"clips\": []}\n```\nDone."
    assert ea.extract_json(txt) == {"clips": []}


def test_extract_json_with_prose():
    txt = 'Here is the answer: {"clips":[{"start_s":1,"end_s":2,"hook_title":"a","caption":"b","reason":"c"}]} hope that helps'
    out = ea.extract_json(txt)
    assert out and len(out["clips"]) == 1


def test_extract_json_garbage():
    assert ea.extract_json("no json here at all") is None


def test_validate_clips_bounds_and_overlap():
    duration = 100.0
    clips = [
        {"start_s": 0, "end_s": 30},      # <2s from start -> rejected
        {"start_s": 10, "end_s": 80},     # too long -> rejected
        {"start_s": 10, "end_s": 40},     # ok
        {"start_s": 20, "end_s": 35},     # overlaps prev -> rejected
        {"start_s": 60, "end_s": 99},     # >duration-2 -> rejected
        {"start_s": 60, "end_s": 97},     # ok
    ]
    good = ea.validate_clips(clips, 5, duration, 60, edge_pad=2.0)
    spans = [(c["start_s"], c["end_s"]) for c in good]
    assert spans == [(10, 40), (60, 97)]


def test_heuristic_picks_fill_when_ai_fails(home, monkeypatch):
    # force call_openrouter to raise -> fallback heuristic windows
    def boom(*a, **k):
        raise RuntimeError("no api")
    monkeypatch.setattr(ea, "call_openrouter", boom)
    analysis = Analysis("visual", 300.0, [],
                        [Window(10, 60, 5), Window(120, 180, 4), Window(200, 260, 3)])
    res = ea.pick_clips(Path("x.mp4"), analysis, "T", "C", "cats_silent", 3)
    assert res.used_fallback
    assert len(res.clips) == 3
    assert all(p.source == "heuristic" for p in res.clips)


def test_retry_then_success(home, monkeypatch):
    calls = {"n": 0}
    good = {"clips": [
        {"start_s": 10, "end_s": 40, "hook_title": "a", "caption": "b", "reason": "c"},
        {"start_s": 100, "end_s": 140, "hook_title": "d", "caption": "e", "reason": "f"},
    ]}

    def fake(messages, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json", {}, 1.0
        return json.dumps(good), {"completion_tokens": 5}, 1.0

    monkeypatch.setattr(ea, "call_openrouter", fake)
    words = [Word(i, i + 1, "goal") for i in range(120)]
    analysis = Analysis("speech", 300.0, words, [])
    res = ea.pick_clips(Path("x.mp4"), analysis, "T", "C", "football", 2)
    assert calls["n"] == 2 and not res.used_fallback
    assert len(res.clips) == 2
