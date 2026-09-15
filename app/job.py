"""Daily-job orchestration (spec §5).

Kept out of ``main.py`` so it is unit-testable without a running web server.
The whole sequence for one video lives in :func:`process_video`; the outer
quota/rotation loop is :func:`daily_job`. A crash leaves ``used_at`` NULL (it is
set only after a successful render), so the next run re-considers the video and
never double-produces (spec §11.7).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from . import config as cfg
from . import db, selector, downloader, analyzer, editor_ai, cutter, agent, capcut_export
from . import events, notifier

LogFn = Callable[[str], None]


def _noop(_m: str) -> None:
    return None


def topic_mode(topic_key: str) -> str:
    return cfg.get_config().topic_mode(topic_key)


def _norm_clips(result) -> list[dict]:
    """editor_ai.Pick list -> plain dicts carrying an engine tag."""
    clips = []
    for p in result.clips:
        clips.append({
            "start_s": round(p.start_s, 2), "end_s": round(p.end_s, 2),
            "hook_title": p.hook_title, "caption": p.caption, "reason": p.reason,
            "model": getattr(result, "model", None),
            "engine": "fallback_single_shot" if getattr(result, "used_fallback", False) else "single_shot",
        })
    return clips


def render_clips(video_id: str, src: Path, clips: list[dict], words, title: str,
                 channel_name: str, topic: str, prompt_source: str, log: LogFn) -> list[str]:
    """Render each clip dict to mp4, insert rows. Returns produced clip paths."""
    paths: list[str] = []
    for i, cl in enumerate(clips):
        out = cutter.output_path_for(topic, channel_name, title, i)
        path = cutter.render_clip(src, cl["start_s"], cl["end_s"], out, words=words)
        if path:
            engine = cl.get("engine", "single_shot")
            new_id = db.add_clip(video_id, cl["start_s"], cl["end_s"], str(path),
                        cl.get("caption"), engine, prompt_source, engine=engine,
                        hook_title=cl.get("hook_title"), render_mode="mp4")
            events.clip_rendered({"id": new_id, "video_id": video_id, "path": str(path),
                                  "caption": cl.get("caption"), "hook_title": cl.get("hook_title"),
                                  "engine": engine, "render_mode": "mp4",
                                  "start_s": cl["start_s"], "end_s": cl["end_s"]})
            log(f"    clip {i + 1}: {cl['start_s']:.0f}-{cl['end_s']:.0f}s "
                f"[{engine}] -> {path.name}")
            paths.append(str(path))
        else:
            log(f"    clip {i + 1}: render failed")
    return paths


def export_clips_draft(video_row: dict, src: Path, clips: list[dict], prompt_source: str,
                       log: LogFn) -> list[str]:
    """Write one CapCut draft folder per clip (feature B). Returns draft paths.

    Raises RuntimeError if capcut is disabled or the draft root is unwritable so
    the caller can fall back to mp4 mode.
    """
    results: list[str] = []
    for i, cl in enumerate(clips):
        video_row = {**video_row, "_clip_index": i}
        draft = capcut_export.export_draft(video_row, [cl], src_path=src)
        engine = cl.get("engine", "single_shot")
        db.add_clip(video_row["video_id"], cl["start_s"], cl["end_s"], None,
                    cl.get("caption"), engine, prompt_source, engine=engine,
                    hook_title=cl.get("hook_title"), render_mode="capcut", draft_path=draft)
        log(f"    draft {i + 1}: {cl['start_s']:.0f}-{cl['end_s']:.0f}s -> {Path(draft).name}")
        results.append(draft)
    return results


def _single_shot(video_id: str, channel_row: dict, src: Path, topic: str,
                 n: int, prompt_override: str | None, log: LogFn) -> list[dict]:
    mode = topic_mode(topic)
    log(f"  analyzing (mode={mode})")
    analysis = analyzer.analyze(src, topic, mode, video_id=video_id)
    db.set_video_status(video_id, "analyzed")
    log(f"  analysis: {analysis.mode} path, {len(analysis.windows)} candidate windows")
    result = editor_ai.pick_clips(src, analysis, channel_row.get("title") or "",
                                  channel_row.get("title") or topic, topic, n, prompt_override)
    if result.used_fallback:
        log(f"  editor AI -> heuristic fallback ({result.error or 'count mismatch'})")
    else:
        log(f"  editor AI selected {len(result.clips)} clip(s) via {result.model}")
    return _norm_clips(result)


def process_video(video_id: str, channel_row: dict, url: str | None, log: LogFn,
                  force_n: int | None = None,
                  prompt_override: str | None = None,
                  output_mode: str | None = None,
                  staged: bool = False) -> list[str]:
    """Download -> (agent | single-shot) -> render mp4 OR export CapCut draft.

    output_mode: 'mp4' | 'capcut' (default: config/state 'output_mode', else mp4).
    staged=True consumes a pre-staged row: skips download + transcribe (already
    cached) and logs 'prestaged'.
    """
    topic = channel_row["topic"]
    channel_name = channel_row.get("title") or topic
    c = cfg.get_config()
    prompt_source = "user" if prompt_override else "default"
    if output_mode not in ("mp4", "capcut"):
        output_mode = db.get_state("output_mode", c.get("capcut.default_mode", "mp4") or "mp4")

    vrow = db.query_one("SELECT duration_s FROM videos WHERE video_id=?", (video_id,))
    duration_s = (vrow["duration_s"] if vrow and vrow["duration_s"] else 0) or 0

    if staged:
        src = downloader.resolve_source(video_id)
    else:
        src = None
    if src and Path(src).is_file():
        log(f"  {channel_name}: using PRESTAGED source ({Path(src).name})")
        db.set_video_status(video_id, "analyzed")
        duration_s = duration_s or (analyzer.probe_duration(Path(src), c.ffprobe) or 600.0)
    else:
        if staged:
            log("  staged source missing; re-downloading")
        log(f"  downloading {video_id} ({channel_name})")
        events.job_progress("download", video_id, 0.1)
        db.set_video_status(video_id, "pending")
        src = downloader.download(video_id, url)
        if not src or not Path(src).is_file():
            log("  download failed")
            db.set_video_status(video_id, "failed")
            return []
        db.set_video_status(video_id, "downloaded")
    src = Path(src)

    if not duration_s:
        duration_s = analyzer.probe_duration(src, c.ffprobe) or 600.0
    n = force_n if force_n is not None else selector.windows_for_duration(duration_s)
    if force_n is None:
        remaining = int(c.get("job.daily_quota", 4)) - db.clips_today()
        n = selector.clamp_windows(n, max(0, remaining))
    if n < 1:
        log("  quota reached; nothing to render")
        return []

    # --- selection: tool-calling agent (default) or single-shot (fallback) ---
    clips: list[dict]
    events.job_progress("select", f"editor for {n} clip(s)", 0.4)
    if c.get("agent.enabled", True) and c.openrouter_api_key:
        log(f"  running editor AGENT for {n} clip(s)")
        clips = agent.run_agent(
            {"video_id": video_id, "duration_s": duration_s, "topic": topic},
            n, prompt_override, source=src, topic=topic, log=log)
        db.set_video_status(video_id, "analyzed")
        words = analyzer.cached_words(video_id)   # agent's get_transcript cached it
    else:
        if not c.openrouter_api_key:
            log("  no API key: using single-shot / heuristic path")
        try:
            clips = _single_shot(video_id, channel_row, src, topic, n, prompt_override, log)
        except Exception as e:  # analysis blow-up (missing model, etc.)
            log(f"  analysis error: {e}")
            db.set_video_status(video_id, "failed")
            return []
        words = analyzer.cached_words(video_id)

    if not clips:
        log("  editor produced no clips")
        db.set_video_status(video_id, "failed")
        return []

    # --- output: mp4 render OR CapCut draft (feature B) ---
    if output_mode == "capcut":
        video_row = {"video_id": video_id, "title": channel_row.get("title") or topic,
                     "duration_s": duration_s, "topic": topic}
        try:
            paths = export_clips_draft(video_row, src, clips, prompt_source, log)
        except Exception as e:
            log(f"  capcut export failed ({e}); falling back to mp4 render")
            paths = render_clips(video_id, src, clips, words, channel_row.get("title") or "",
                                 channel_name, topic, prompt_source, log)
    else:
        paths = render_clips(video_id, src, clips, words, channel_row.get("title") or "",
                             channel_name, topic, prompt_source, log)
    if paths:
        db.mark_video_used(video_id)   # used_at set ONLY after a successful render/export
    else:
        db.set_video_status(video_id, "failed")
    return paths


def ensure_custom_video_row(video_id: str, info: dict) -> dict:
    """Insert (or fetch) a channels+videos row for a user-supplied URL."""
    cid = info.get("channel_id") or f"custom-{video_id}"
    existing = db.query_one("SELECT id FROM channels WHERE platform_channel_id=?", (cid,))
    if existing:
        ch_id = existing["id"]
    else:
        ch_id = db.add_channel(cid, info.get("channel") or "Custom", info.get("_topic") or "custom",
                               None, info.get("channel_follower_count"), None, None,
                               info.get("channel_thumbnail"), )
    if not db.query_one("SELECT video_id FROM videos WHERE video_id=?", (video_id,)):
        db.insert_video(video_id, ch_id, info.get("title"), info.get("duration"),
                        info.get("view_count"), info.get("upload_date"))
    row = db.query_one("SELECT * FROM channels WHERE id=?", (ch_id,))
    return dict(row)


def daily_job(log: LogFn = _noop, output_mode: str | None = None) -> list[str]:
    """The full run-until-quota loop. Returns produced clip/draft paths (for --auto)."""
    c = cfg.get_config()
    quota = int(c.get("job.daily_quota", 4))
    max_attempts = int(c.get("job.max_attempts", 6))
    produced: list[str] = []
    attempts = 0
    tried: set[int] = set()
    log(f"[run] start — quota={quota}, clips_today={db.clips_today()}, output={output_mode or 'last-used'}")

    while db.clips_today() < quota and attempts < max_attempts:
        attempts += 1
        channel = selector.pick_next_channel(skip_channel_ids=tried)
        if not channel:
            log("[run] no channels available for rotation")
            break
        tried.add(channel["id"])
        events.job_progress("channel", channel["title"], attempts / max_attempts)
        staged = db.staged_for_channel(channel["id"])
        if staged:
            log(f"[attempt {attempts}] channel: {channel['title']} — PRESTAGED {staged['video_id']}")
            paths = process_video(staged["video_id"], channel, None, log,
                                  output_mode=output_mode, staged=True)
        else:
            log(f"[attempt {attempts}] channel: {channel['title']} ({channel['topic']})")
            video = selector.pick_video(channel)
            if not video:
                log("  no eligible video (all seen / out of duration) — next channel")
                continue
            paths = process_video(video["video_id"], channel, None, log, output_mode=output_mode)
        produced += paths
        if not paths:
            log("  channel produced 0 clips; continuing loop with next channel")

    made = db.clips_today()
    log(f"[run] done — clips_today={made}")
    events.job_progress("done", f"{made}/{quota}", 1.0)
    if made < quota:
        notifier.notify("quota_unmet", "warn", "Daily quota not met",
                        f"Produced {made}/{quota} today (channels exhausted or candidates failed).",
                        actions=[{"label": "Retry now", "action": "run_daily"},
                                 {"label": "Reduce quota today", "action": "reduce_quota"}],
                        dedupe_key=f"quota_unmet:{datetime.now().date()}")
    return produced


# --- custom single-video run (optional URL + prompt + count) ----------------
def custom_job(url_or_id: str | None, log: LogFn = _noop, clip_count: int | None = None,
               prompt_override: str | None = None, output_mode: str | None = None) -> list[str]:
    c = cfg.get_config()
    quota = int(c.get("job.daily_quota", 4))
    video_id = _extract_video_id(url_or_id) if url_or_id else None

    if not video_id:
        return daily_job(log, output_mode=output_mode)

    info = _fetch_video_info(video_id)
    ch = ensure_custom_video_row(video_id, info)
    if clip_count:
        n = int(clip_count)
    else:
        n = min(selector.windows_for_duration(info.get("duration") or 600),
                max(0, quota - db.clips_today()))
    if n < 1:
        log("[custom] quota already reached")
        return []
    return process_video(video_id, ch, None, log, force_n=n,
                         prompt_override=prompt_override, output_mode=output_mode)


def _extract_video_id(url: str) -> str | None:
    import re
    url = url.strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|/embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


def _fetch_video_info(video_id: str) -> dict:
    info = selector.discovery.ytdlp_json(
        [f"https://www.youtube.com/watch?v={video_id}", "--no-playlist",
         "--skip-download"], timeout=60) or {}
    return {
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "_topic": "custom",
    }
