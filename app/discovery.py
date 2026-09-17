"""Candidate + channel discovery (spec §4).

Pipeline per topic:
  1. yt-dlp ``ytsearch`` on the topic's queries  -> raw video hits
  2. group hits by channel_id                    -> candidate channels
  3. yt-dlp ``/videos`` flat playlist            -> video_count + channel meta
  4. Scrapling StealthyFetcher on ``/about``      -> joined_year + avatar (fail-soft)
  5. filter: channel exists >= 3 years (unknown year is kept, "null allowed")

Every external call is fail-soft: a missing Scrapling browser or a network
error degrades to partial metadata rather than crashing onboarding. If the
primary yt-dlp search fails entirely, ``fallback_search`` scrapes the YouTube
results page with Scrapling.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from . import config as cfg

YT_SEARCH_PREFIX = "ytsearch"
_SEARCH_LIMIT = 10
_SEARCH_TIMEOUT = 15         # per ytsearch call
_ENRICH_TIMEOUT = 15         # per-channel yt-dlp /videos call (s)
_ENRICH_MAX = 4              # enrich at most this many channels per topic
_TOPIC_BUDGET_S = 30.0       # hard wall-clock budget for one topic discovery
_JOIN_YEAR_RE = re.compile(r"Joined\s+([A-Za-z]+)\s+\d{1,2},\s+(\d{4})")
_JOIN_YEAR_RE2 = re.compile(r"(\d{4})")


@dataclass
class Candidate:
    platform_channel_id: str
    title: str
    topic: str
    url: str = ""
    subs: int | None = None
    video_count: int | None = None
    joined_year: int | None = None
    avatar_url: str | None = None
    sample_view_total: int = 0
    seen_videos: int = 0
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- yt-dlp subprocess helper (monkeypatched in tests) ----------------------
def _ytdlp_bin() -> str | None:
    return shutil.which("yt-dlp")


def ytdlp_json(args: list[str], timeout: int = 60) -> dict | None:
    """Run yt-dlp with -J and parse the JSON. Returns None on any failure."""
    exe = _ytdlp_bin()
    if not exe:
        return None
    cmd = [exe, "-J", "--no-warnings", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        _metric(False)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        _metric(False)
        return None
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        _metric(False)
        return None
    _metric(True)
    return out


def _metric(success: bool) -> None:
    try:
        from . import db
        db.metrics_bump("ytdlp", success)
    except Exception:  # noqa: BLE001 - metrics must never break discovery
        pass


# --- search -----------------------------------------------------------------
def _search_query(query: str, limit: int = _SEARCH_LIMIT,
                  timeout: int | None = None) -> list[dict]:
    info = ytdlp_json([f"{YT_SEARCH_PREFIX}{limit}:{query}", "--flat-playlist"],
                      timeout=timeout or _SEARCH_TIMEOUT)
    if not info:
        return []
    entries = info.get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


def _channel_from_hits(topic: str, query: str, hits: list[dict],
                       acc: dict[str, Candidate]) -> None:
    for h in hits:
        cid = h.get("channel_id") or h.get("uploader_id")
        if not cid:
            continue
        cu = h.get("channel_url") or ""
        if not cu and cid.startswith(("UC", "FC")):
            cu = f"https://www.youtube.com/channel/{cid}"
        cand = acc.get(cid)
        if cand is None:
            cand = Candidate(
                platform_channel_id=cid,
                title=h.get("channel") or h.get("uploader") or cid,
                topic=topic,
                url=cu,
            )
            acc[cid] = cand
        cand.seen_videos += 1
        cand.sample_view_total += int(h.get("view_count") or 0)
        if query not in cand.queries:
            cand.queries.append(query)


def _enrich_channel_meta(cand: Candidate, timeout: int = _ENRICH_TIMEOUT) -> None:
    """yt-dlp /videos tab -> video_count, subs, title, avatar fallback."""
    videos_url = (cand.url or f"https://www.youtube.com/channel/{cand.platform_channel_id}") + "/videos"
    info = ytdlp_json([videos_url, "--flat-playlist", "--playlist-items", "1-200"], timeout=timeout)
    if not info:
        return
    if info.get("title"):
        cand.title = info["title"]
    if info.get("channel_follower_count") is not None:
        cand.subs = int(info["channel_follower_count"])
    entries = info.get("entries") or []
    if info.get("playlist_count"):
        cand.video_count = int(info["playlist_count"])
    elif entries:
        cand.video_count = len(entries)
    for t in reversed(info.get("thumbnails") or []):
        if t.get("url"):
            cand.avatar_url = cand.avatar_url or t["url"]
            break


# --- Scrapling About-page enrichment ----------------------------------------
def _scrapling_available() -> bool:
    try:
        import scrapling  # noqa: F401
        return True
    except Exception:
        return False


def enrich_about(cand: Candidate) -> None:
    """Fail-soft Scrapling StealthyFetcher scrape of the channel About page."""
    if not _scrapling_available():
        return
    about_url = (cand.url or f"https://www.youtube.com/channel/{cand.platform_channel_id}") + "/about"
    try:
        from scrapling.fetchers import StealthyFetcher
        page = StealthyFetcher.fetch(about_url, headless=True, network_idle=True, timeout=45)
        text = page.get_all_text(ignore_tags=("script", "style")) if hasattr(page, "get_all_text") else str(page.html_content)
        joined = _parse_joined_year(text)
        if joined:
            cand.joined_year = joined
        av = _css_first(page, "yt-img-avatar img", "src") or _css_first(page, "img.yt-coreImage", "src")
        if av:
            cand.avatar_url = av
    except Exception:
        return


def _parse_joined_year(text: str) -> int | None:
    m = _JOIN_YEAR_RE.search(text or "")
    if m:
        try:
            return int(m.group(2))
        except ValueError:
            pass
    # "Joined Mar 5, 2019" variants already covered; bail quietly otherwise.
    return None


def _css_first(page: Any, selector: str, attr: str) -> str | None:
    try:
        el = page.css_first(selector)
        if el is not None:
            val = el.attrib.get(attr)
            return val or None
    except Exception:
        return None
    return None


# --- filtering + ranking ----------------------------------------------------
def _passes_age_filter(cand: Candidate, min_years: int) -> bool:
    if cand.joined_year is None:
        return True  # null allowed -> cannot disprove age
    this_year = datetime.now().year
    return (this_year - int(cand.joined_year)) >= min_years


def discover_topic(topic_key: str, topic_cfg: dict, min_years: int,
                   enrich_scrapling: bool = False) -> list[Candidate]:
    """Return filtered, ranked candidates for one topic (best first).

    Enrichment is capped to ``candidates`` channels and bounded by a wall-clock
    budget so onboarding stays responsive even on slow/throttled networks; any
    channel that misses enrichment is still returned with partial metadata
    (spec §4: "fail-soft: null allowed").
    """
    t0 = time.time()
    acc: dict[str, Candidate] = {}
    for q in topic_cfg.get("queries", []):
        if time.time() - t0 > _TOPIC_BUDGET_S * 0.6:
            break
        hits = _search_query(q)
        _channel_from_hits(topic_key, q, hits, acc)

    # rank raw hits first so we only spend time enriching the best candidates
    prelim = sorted(acc.values(),
                    key=lambda c: (c.seen_videos, c.sample_view_total), reverse=True)
    cap = int(topic_cfg.get("candidates", 10))
    to_enrich = prelim[:cap]
    enrich_limit = min(cap, _ENRICH_MAX)

    for cand in to_enrich[:enrich_limit]:
        if time.time() - t0 > _TOPIC_BUDGET_S:
            break
        _enrich_channel_meta(cand)
        if enrich_scrapling and (time.time() - t0) < _TOPIC_BUDGET_S:
            enrich_about(cand)

    kept = [c for c in to_enrich if _passes_age_filter(c, min_years)]
    # final rank by popularity (subs, then avg sampled views) desc
    kept.sort(key=lambda c: (
        c.subs or 0,
        (c.sample_view_total / c.seen_videos) if c.seen_videos else 0,
    ), reverse=True)
    return kept


def discover_candidates(topic_key: str, deep: bool = False) -> list[dict]:
    """Public entry: candidate dicts for a topic, capped at config candidates.

    ``deep=True`` additionally runs Scrapling About-page enrichment (joined
    year + real avatar). It is off by default so the UI loads fast; the
    dashboard "Deep enrich" toggle opts in.
    """
    c = cfg.get_config()
    topics = c.get("discovery.topics", {}) or {}
    topic_cfg = topics.get(topic_key)
    if not topic_cfg:
        return []
    min_years = int(c.get("discovery.channel_min_age_years", 3))
    cands = discover_topic(topic_key, topic_cfg, min_years, enrich_scrapling=deep)
    cap = int(topic_cfg.get("candidates", 10))
    result = [c.to_dict() for c in cands[:cap]]

    # fallback scrape path (only if yt-dlp yielded nothing)
    if not result and _scrapling_available():
        result = [c.to_dict() for c in fallback_search(topic_key, topic_cfg)[:cap]]
    return result


# --- fallback scrape (yt-dlp failed) ----------------------------------------
def fallback_search(topic_key: str, topic_cfg: dict) -> list[Candidate]:
    """Scrape youtube.com/results with Scrapling when yt-dlp search fails."""
    out: dict[str, Candidate] = {}
    if not _scrapling_available():
        return []
    try:
        from scrapling.fetchers import StealthyFetcher
    except Exception:
        return []
    for q in topic_cfg.get("queries", []):
        url = "https://www.youtube.com/results?search_query=" + re.sub(r"\s+", "+", q)
        try:
            page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=45)
            for a in page.css('a[href^="/@"]'):
                handle = a.attrib.get("href", "")
                if not handle:
                    continue
                cid = handle.lstrip("/@")
                if cid not in out:
                    out[cid] = Candidate(platform_channel_id="@" + cid,
                                         title=cid, topic=topic_key,
                                         url="https://www.youtube.com" + handle)
            for a in page.css('a[href^="/channel/"]'):
                href = a.attrib.get("href", "")
                cid = href.split("/channel/", 1)[-1].split("?")[0]
                if cid and cid not in out:
                    out[cid] = Candidate(platform_channel_id=cid, title=cid,
                                         topic=topic_key,
                                         url="https://www.youtube.com/channel/" + cid)
        except Exception:
            continue
    for cand in out.values():
        _enrich_channel_meta(cand)
    return list(out.values())
