"""Transcript fetching: fallback chain, garbage detection, VTT parse (mocked)."""
import pytest
from app import transcript_fetch as tf, config as cfg


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


GARBAGE = '0:{"a":"$@1","f":"","b":"i-dEv7N7_G2B1apWtEAsb"} 1:null'


@pytest.fixture(autouse=True)
def fast(tmp_path, monkeypatch):
    """Point transcripts at tmp and remove pacing sleeps for tests."""
    cfg.TRANSCRIPTS_DIR = tmp_path / "transcripts"

    class _NoSleep:
        @staticmethod
        def sleep(*_a):
            return None
    monkeypatch.setattr(tf, "time", _NoSleep)


def test_fetch_one_service_success(monkeypatch):
    monkeypatch.setattr(tf, "_post",
                        lambda url, payload, timeout: FakeResp(200, {"transcript": "hello world"}))
    status, text, detail, segs = tf.fetch_one("https://www.youtube.com/watch?v=abc")
    assert status == tf.STATUS_OK and text == "hello world"
    assert detail == "hosted service" and segs == []


def test_service_payload_uses_url_field(monkeypatch):
    sent = {}
    def fake(url, payload, timeout):
        sent.update(payload)
        return FakeResp(200, {"transcript": "ok"})
    monkeypatch.setattr(tf, "_post", fake)
    tf.fetch_one("https://www.youtube.com/watch?v=abc")
    assert "url" in sent and "video_url" not in sent   # live schema (README is stale)


def test_service_garbage_refused(monkeypatch):
    monkeypatch.setattr(tf, "_post",
                        lambda url, payload, timeout: FakeResp(200, {"transcript": GARBAGE}))
    status, text, detail, _ = tf.fetch_one("u1")
    assert status == tf.STATUS_ERROR and text is None
    assert "garbage" in detail


def test_404_no_transcript(monkeypatch):
    monkeypatch.setattr(tf, "_post", lambda url, payload, timeout: FakeResp(404, {"detail": "x"}))
    status, text, _, _ = tf.fetch_one("u")
    assert status == tf.STATUS_EMPTY and text is None


def test_429_retries_then_ok(monkeypatch):
    seq = iter([FakeResp(429), FakeResp(200, {"transcript": "after backoff"})])
    monkeypatch.setattr(tf, "_post", lambda url, payload, timeout: next(seq))
    status, text, _, _ = tf.fetch_one("u")
    assert status == tf.STATUS_OK and text == "after backoff"


def test_fallback_chain_library_rescues(monkeypatch):
    # hosted returns garbage; canonical library succeeds
    monkeypatch.setattr(tf, "_post", lambda url, payload, timeout: FakeResp(200, {"transcript": GARBAGE}))
    monkeypatch.setattr(tf, "_fetch_library",
                        lambda vid: (tf.STATUS_OK, "real text", "youtube-transcript-api library",
                                     [[0.0, 2.0, "real"]]))
    monkeypatch.setattr(tf, "_fetch_ytdlp",
                        lambda vid: pytest.fail("ytdlp must not run when the library succeeded"))
    status, text, detail, segs = tf.fetch_one("https://www.youtube.com/watch?v=abcdefghi")
    assert status == tf.STATUS_OK and text == "real text"
    assert detail == "youtube-transcript-api library" and segs == [[0.0, 2.0, "real"]]


def test_fallback_chain_all_fail_reports_each(monkeypatch):
    monkeypatch.setattr(tf, "_post", lambda url, payload, timeout: FakeResp(500, {}))
    monkeypatch.setattr(tf, "_fetch_library", lambda vid: (tf.STATUS_ERROR, None, "IP blocked", []))
    monkeypatch.setattr(tf, "_fetch_ytdlp", lambda vid: (tf.STATUS_ERROR, None, "bot-gated", []))
    status, text, detail, _ = tf.fetch_one("https://www.youtube.com/watch?v=abcdefghi")
    assert status == tf.STATUS_ERROR and text is None
    assert "hosted: HTTP 500" in detail and "library: IP blocked" in detail and "ytdlp: bot-gated" in detail


def test_vtt_parse():
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500
Hello world

00:00:03.500 --> 00:00:05.000
<underline>Second</underline> cue
"""
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "c.vtt"
    p.write_text(vtt, encoding="utf-8")
    segs = tf._parse_vtt(p)
    assert segs[0] == [1.0, 3.5, "Hello world"]
    assert segs[1][2] == "Second cue"          # markup stripped


def test_fetch_bulk_writes_and_is_resumable(monkeypatch):
    vids = [{"video_id": f"v{i}", "url": f"u{i}", "title": f"T{i}"} for i in range(3)]
    monkeypatch.setattr(tf, "_fetch_service",
                        lambda url: (tf.STATUS_OK, f"text {url}", "hosted service"))
    summary = tf.fetch_bulk(vids, log=lambda *_: None)
    assert summary == {"fetched": 3, "skipped": 0, "failed": 0}
    rec = tf.load("v1")
    assert rec["transcript"] == "text u1" and rec["source"] == "hosted service"
    assert rec["url"] == "u1"
    summary2 = tf.fetch_bulk(vids, log=lambda *_: None)   # resumable: skips all
    assert summary2 == {"fetched": 0, "skipped": 3, "failed": 0}


def test_fetch_bulk_fail_soft_per_video(monkeypatch):
    vids = [{"video_id": "ok1", "url": "u1", "title": "A"},
            {"video_id": "bad", "url": "u2", "title": "B"},
            {"video_id": "ok2", "url": "u3", "title": "C"}]
    seq = iter([(tf.STATUS_OK, "t1", "hosted service", []),
                (tf.STATUS_EMPTY, None, "no transcript available", []),
                (tf.STATUS_OK, "t3", "hosted service", [])])
    monkeypatch.setattr(tf, "fetch_one", lambda url, vid=None: next(seq))
    summary = tf.fetch_bulk(vids, log=lambda *_: None)
    assert summary["fetched"] == 2 and summary["failed"] == 1
    assert tf.load("bad")["status"] == tf.STATUS_EMPTY
    assert tf.load("ok2")["transcript"] == "t3"


def test_load_from_bulk(tmp_path):
    import json
    (tmp_path / "bulk.json").write_text(json.dumps({
        "videos": [{"video_id": "a", "url": "ua", "title": "A"}]}), encoding="utf-8")
    vids = tf.load_from_bulk(tmp_path / "bulk.json")
    assert vids == [{"video_id": "a", "url": "ua", "title": "A"}]
    assert tf.load_from_bulk(tmp_path / "nope.json") == []
