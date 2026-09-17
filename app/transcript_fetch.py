"""Direct transcript fetching via the youtube-transcript-api REST service.

README (github.com/jaypaun007/youtube-transcript-api): a hosted FastAPI service
— ``POST {service}/transcript`` returning a transcript. The **live schema
differs from its README**: the body must be ``{"url": ...}`` (the README shows
``video_url``, which the deployed service rejects with 422).

Two hard-won caveats, both detected fail-soft here:
  * the hosted instance is rate-limited to 5 requests/minute, AND as of now it
    returns the same canned serialization garbage for EVERY video
    (``0:{"a":"$@1",...}``) — :func:`_fetch_service` detects that and refuses to
    store it as a transcript;
  * the canonical PyPI package (``youtube-transcript-api``) and yt-dlp caption
    fetches are IP/bot-blocked from cloud IPs (work fine on home connections).

So :func:`fetch_one` walks a fallback chain — hosted service → canonical
library → yt-dlp captions — and reports which path (if any) produced text.
Results are paced (~13 s apart), resumable (already-fetched skipped), and
stored per video as ``data/transcripts/{video_id}.fetch.json`` — a distinct
suffix so this never collides with the whisper word-timestamp cache
(``{video_id}.json``) the agent and captions rely on.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

from . import config as cfg

DEFAULT_FROM = "data/bulk_search.json"

STATUS_OK = "ok"
STATUS_EMPTY = "no_transcript"
STATUS_ERROR = "error"

_GARBAGE_RE = re.compile(r'\{"a":"\$@')  # React-Flight artifacts leaked by the service


def _service_url() -> str:
    return str(cfg.get_config().get("transcript_fetch.service_url",
                                    "https://youtube-transcript-api-tau-one.vercel.app/transcript"))


def _min_interval() -> float:
    return float(cfg.get_config().get("transcript_fetch.min_interval_s", 13))


def _timeout() -> float:
    return float(cfg.get_config().get("transcript_fetch.timeout_s", 60))


def _safe(video_id: str) -> str:
    return "".join(ch for ch in video_id if ch.isalnum() or ch in ("-", "_"))[:80] or "video"


def _file_for(video_id: str) -> Path:
    return cfg.TRANSCRIPTS_DIR / f"{_safe(video_id)}.fetch.json"


# --- transports (monkeypatched in tests) ------------------------------------
def _post(url: str, payload: dict, timeout: float) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


# --- path 1: hosted service -------------------------------------------------
class _ServiceState:
    """Circuit breaker: a broken deploy returns garbage for EVERY video — after
    3 consecutive garbage responses, skip the service for the rest of the run
    instead of pacing through 90 pointless calls."""
    consec_garbage = 0
    broken = False


_SERVICE = _ServiceState()


def _fetch_service(video_url: str) -> tuple[str, str | None, str]:
    """(status, text, detail). 429 -> one backoff retry; garbage is refused."""
    if _SERVICE.broken:
        return STATUS_ERROR, None, "service known broken (skipped)"
    payload = {"url": video_url}   # live schema (README's `video_url` is stale)
    for attempt in range(2):
        try:
            resp = _post(_service_url(), payload, _timeout())
        except requests.RequestException as e:
            return STATUS_ERROR, None, f"request failed: {e}"
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                return STATUS_ERROR, None, "invalid JSON from service"
            text = (data.get("transcript") or "").strip()
            if not text:
                return STATUS_EMPTY, None, "empty transcript"
            if _GARBAGE_RE.search(text) or (len(text) < 120 and "{" in text):
                _SERVICE.consec_garbage += 1
                if _SERVICE.consec_garbage >= 3:
                    _SERVICE.broken = True
                return STATUS_ERROR, None, "service returned serialization garbage (broken deploy)"
            _SERVICE.consec_garbage = 0
            return STATUS_OK, text, "hosted service"
        if resp.status_code == 429 and attempt == 0:
            time.sleep(25.0)  # rate-limit window: back off once, then retry
            continue
        if resp.status_code == 404:
            return STATUS_EMPTY, None, "no transcript available"
        return STATUS_ERROR, None, f"HTTP {resp.status_code}"
    return STATUS_ERROR, None, "rate limited"


# --- path 2: canonical library (real transcripts, no rate limit) ------------
def _fetch_library(video_id: str) -> tuple[str, str | None, str, list]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, IpBlocked, NoTranscriptFound
    except Exception as e:
        return STATUS_ERROR, None, f"library unavailable: {e}", []
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        segs = [[round(s.start, 2), round(s.end, 2), s.text.strip()] for s in fetched]
        text = " ".join(s[2] for s in segs)
        if not text:
            return STATUS_EMPTY, None, "empty transcript", []
        return STATUS_OK, text, "youtube-transcript-api library", segs
    except IpBlocked:
        return STATUS_ERROR, None, "IP blocked by YouTube (cloud IP)", []
    except TranscriptsDisabled:
        return STATUS_EMPTY, None, "captions disabled", []
    except NoTranscriptFound:
        return STATUS_EMPTY, None, "no transcript found", []
    except Exception as e:
        return STATUS_ERROR, None, f"library error: {e}", []


# --- path 3: yt-dlp caption fetch (works with cookies) ----------------------
def _fetch_ytdlp(video_id: str) -> tuple[str, str | None, str, list]:
    exe = shutil_which("yt-dlp")
    if not exe:
        return STATUS_ERROR, None, "yt-dlp not installed", []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cap.%(ext)s"
        try:
            subprocess.run(
                [exe, "--skip-download", "--write-auto-subs", "--write-subs",
                 "--sub-format", "vtt", "--sub-langs", "en.*",
                 "-o", str(out), f"https://www.youtube.com/watch?v={video_id}"],
                capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError) as e:
            return STATUS_ERROR, None, f"yt-dlp failed: {e}", []
        vtts = list(Path(td).glob("*.vtt"))
        if not vtts:
            return STATUS_ERROR, None, "no caption track (bot-gated or none)", []
        segs = _parse_vtt(vtts[0])
        if not segs:
            return STATUS_EMPTY, None, "empty captions", []
        return STATUS_OK, " ".join(s[2] for s in segs), "yt-dlp captions", segs


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def _parse_vtt(path: Path) -> list[list]:
    """VTT cues -> [[start, end, text], ...] (deduped rolling captions)."""
    segs: list[list] = []
    start = end = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if "-->" in line:
            a, _, b = line.partition("-->")
            start = _vtt_s(a.strip())
            end = _vtt_s(b.split()[0].strip())
            continue
        if line and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) and start is not None:
            text = re.sub(r"<[^>]+>", "", line).strip()
            if text and (not segs or segs[-1][2] != text):
                segs.append([start, end, text])
            start = end = None
    return segs


def _vtt_s(stamp: str) -> float:
    m = re.match(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", stamp)
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# --- fallback chain ---------------------------------------------------------
def fetch_one(video_url: str, video_id: str | None = None) -> tuple[str, str | None, str, list]:
    """Hosted service -> canonical library -> yt-dlp captions.

    Only a transport/garbage FAILURE triggers the fallbacks; a definitive
    "no transcript" (404/empty) is returned as-is. Returns
    (status, transcript|None, source_detail, segments).
    """
    vid = video_id or _id_from_url(video_url)
    status, text, detail = _fetch_service(video_url)
    if status == STATUS_OK:
        return status, text, detail, []
    if status == STATUS_EMPTY:
        return status, None, detail, []

    errors = [f"hosted: {detail}"]
    s2, t2, d2, segs2 = _fetch_library(vid)
    if s2 == STATUS_OK:
        return s2, t2, d2, segs2
    errors.append(f"library: {d2}")
    s3, t3, d3, segs3 = _fetch_ytdlp(vid)
    if s3 == STATUS_OK:
        return s3, t3, d3, segs3
    errors.append(f"ytdlp: {d3}")
    worst = STATUS_EMPTY if (s2 == STATUS_EMPTY or s3 == STATUS_EMPTY) else STATUS_ERROR
    return worst, None, "; ".join(errors), []


def _id_from_url(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else None


# --- bulk, resumable, paced --------------------------------------------------
def fetch_bulk(videos: list[dict], out_dir: Path | None = None,
               log: Callable = print) -> dict[str, int]:
    """Fetch transcripts for [{video_id, url, title?}], skipping done ones.

    Writes one ``.fetch.json`` per video into ``data/transcripts``. Fail-soft
    per video. Returns {fetched, skipped, failed}.
    """
    summary = {"fetched": 0, "skipped": 0, "failed": 0}
    interval = _min_interval()
    _SERVICE.consec_garbage = 0
    _SERVICE.broken = False   # fresh breaker per run
    for i, v in enumerate(videos):
        vid = str(v.get("video_id") or "")
        url = v.get("url") or f"https://www.youtube.com/watch?v={vid}"
        if not vid:
            continue
        dest = _file_for(vid)
        if dest.exists():
            summary["skipped"] += 1
            continue
        if i and summary["fetched"] + summary["failed"]:
            time.sleep(interval)  # pace: stay under the hosted 5 req/min limit
        status, text, detail, segs = fetch_one(url, vid)
        record = {
            "video_id": vid, "url": url, "title": v.get("title"),
            "status": status if status != STATUS_OK else "ok",
            "source": detail if status == STATUS_OK else detail,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
        if status == STATUS_OK:
            record.update({"transcript": text, "chars": len(text or ""), "segments": segs})
            summary["fetched"] += 1
            log(f"[transcripts] {i + 1}/{len(videos)} ok  {vid}  ({len(text or '')} chars, "
                f"{len(segs)} cues) via {detail}  {(v.get('title') or '')[:44]}")
        else:
            record["error"] = detail
            summary["failed"] += 1
            log(f"[transcripts] {i + 1}/{len(videos)} {status}  {vid}  {detail}")
        _write(dest, record)
    return summary


def _write(dest: Path, record: dict[str, Any]) -> None:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def load(video_id: str) -> dict[str, Any] | None:
    """Read one fetched transcript record (None if never fetched)."""
    p = _file_for(video_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_from_bulk(path: str | Path = DEFAULT_FROM) -> list[dict]:
    """Videos from a saved bulk_search file (the 93-URL list)."""
    from . import bulk_search
    data = bulk_search.load(path)
    return list(data.get("videos", []) or [])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="transcript_fetch")
    ap.add_argument("--from", dest="src", default=DEFAULT_FROM,
                    help="bulk_search JSON with the URL list (default: data/bulk_search.json)")
    ap.add_argument("--limit", type=int, default=0, help="only fetch the first N (0 = all)")
    args = ap.parse_args()
    vids = load_from_bulk(args.src)
    if args.limit:
        vids = vids[:args.limit]
    print(f"[transcripts] {len(vids)} video(s) to fetch "
          f"(~{_min_interval():.0f}s apart -> ~{len(vids) * _min_interval() / 60:.0f} min)")
    result = fetch_bulk(vids, log=print)
    print(f"[transcripts] done: {result}")
