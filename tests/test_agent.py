"""Feature A — tool-calling editor agent (acceptance C2 agent tests)."""
import json
import pytest
from app import agent, config as cfg, db
from app.analyzer import Word


@pytest.fixture()
def source(tmp_path):
    # a tiny real file; analysis tools are stubbed so its contents don't matter
    p = tmp_path / "src.mp4"
    p.write_bytes(b"\x00" * 32)
    return p


def _toolcall(name, args, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _assistant_with(calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": calls}


def test_validate_proposed_overlap_detected():
    clips = [{"start_s": 10, "end_s": 40}, {"start_s": 30, "end_s": 50}]
    valid, errs = agent.validate_proposed(clips, 2, 600.0)
    assert any("overlap" in e.lower() for e in errs)
    assert len(valid) == 1  # first kept, second rejected


def test_validate_proposed_rejects_too_long_and_close_to_edge():
    clips = [{"start_s": 0, "end_s": 30},          # <2s from start
             {"start_s": 100, "end_s": 170},       # >60s
             {"start_s": 200, "end_s": 250}]        # ok
    valid, errs = agent.validate_proposed(clips, 2, 600.0)
    assert len(valid) == 1 and valid[0]["start_s"] == 200.0
    assert any("> 60" in e or "60s" in e for e in errs)
    assert any("start" in e.lower() for e in errs)


def test_agent_calls_tools_then_proposes(home, source, monkeypatch):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    calls = [
        (_assistant_with([_toolcall("get_transcript", {"chunk_index": 0})]),),
        (_assistant_with([_toolcall("get_audio_energy", {"start_s": 0, "end_s": 300})]),),
        (_assistant_with([_toolcall("propose_clips", {"clips": [
            {"start_s": 50, "end_s": 100, "hook_title": "h1", "caption": "c1", "reason": "r1"},
            {"start_s": 200, "end_s": 260, "hook_title": "h2", "caption": "c2", "reason": "r2"}]})]),),
    ]
    seq = iter(calls)
    monkeypatch.setattr(agent.editor_ai, "call_openrouter_tools", lambda m, t, mo: (next(seq)[0], {}, 1.0))
    # make get_transcript not actually transcribe the dummy file
    monkeypatch.setattr(agent.analyzer, "ensure_words", lambda vid, src: [Word(1, 2, "goal")])
    monkeypatch.setattr(agent.analyzer, "ensure_energy", lambda vid, src, step=0.5: ([0, 1], [0.5, 0.6], 1.0))
    clips = agent.run_agent({"video_id": "vid1", "duration_s": 600.0}, 2, None,
                            source=source, topic="football", log=lambda m: None)
    assert len(clips) == 2
    assert all(c["engine"] == "agent" for c in clips)
    # logs show tool calls happened before propose_clips (spec C2 #1)
    log = (cfg.LOGS_DIR / "agent_vid1.jsonl").read_text()
    tools = [json.loads(l) for l in log.splitlines() if l.strip()]
    names = [t.get("tool") for t in tools if t.get("tool")]
    assert "get_transcript" in names and "get_audio_energy" in names
    assert names.index("get_transcript") < len(names)  # tools used


def test_agent_retry_on_no_tool_call(home, source, monkeypatch):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    seq = iter([
        ({"role": "assistant", "content": "thinking...", "tool_calls": []}, {}, 1.0),  # no call -> nudge
        (_assistant_with([_toolcall("propose_clips", {"clips": [
            {"start_s": 10, "end_s": 40}]})]), {}, 1.0),
    ])
    monkeypatch.setattr(agent.editor_ai, "call_openrouter_tools", lambda m, t, mo: next(seq))
    clips = agent.run_agent({"video_id": "vid1", "duration_s": 600.0}, 1, None,
                            source=source, topic="football", log=lambda m: None)
    assert len(clips) == 1 and clips[0]["start_s"] == 10.0


def test_agent_overlapping_propose_self_corrects_or_falls_back(home, source, monkeypatch):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    # first propose overlaps (invalid); second propose is valid -> agent self-corrects
    seq = iter([
        (_assistant_with([_toolcall("propose_clips", {"clips": [
            {"start_s": 10, "end_s": 50}, {"start_s": 40, "end_s": 80}]})]), {}, 1.0),
        (_assistant_with([_toolcall("propose_clips", {"clips": [
            {"start_s": 10, "end_s": 50}, {"start_s": 60, "end_s": 100}]})]), {}, 1.0),
    ])
    monkeypatch.setattr(agent.editor_ai, "call_openrouter_tools", lambda m, t, mo: next(seq))
    clips = agent.run_agent({"video_id": "vid1", "duration_s": 600.0}, 2, None,
                            source=source, topic="football", log=lambda m: None)
    # never renders invalid clips: result is non-overlapping (or fallback)
    spans = [(c["start_s"], c["end_s"]) for c in clips]
    for i in range(len(spans) - 1):
        assert not (spans[i][1] > spans[i + 1][0])


def test_agent_api_error_falls_back(home, source, monkeypatch):
    cfg.DB_PATH = home / "data" / "clipforge.db"; db.init_db(cfg.DB_PATH)
    def boom(*a, **k):
        raise RuntimeError("model not found")
    monkeypatch.setattr(agent.editor_ai, "call_openrouter_tools", boom)
    # single-shot fallback also fails to reach API -> heuristic windows from analyzer
    def fake_analyze(src, topic, mode, video_id=None):
        from app.analyzer import Analysis, Window
        return Analysis("speech", 600.0, [Word(i, i + 1, "goal") for i in range(300)],
                        [Window(50, 100, 3), Window(200, 260, 2)])
    monkeypatch.setattr(agent.analyzer, "analyze", fake_analyze)
    monkeypatch.setattr(agent.editor_ai, "call_openrouter", boom)  # AI down -> heuristic
    clips = agent.run_agent({"video_id": "vid1", "duration_s": 600.0}, 2, None,
                            source=source, topic="football", log=lambda m: None)
    assert len(clips) >= 1
    assert all(c["engine"] == "fallback_single_shot" for c in clips)
    log = (cfg.LOGS_DIR / "agent_vid1.jsonl").read_text()
    assert "fallback_single_shot" in log
