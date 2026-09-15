"""ClipForge web app (spec §9).

FastAPI + Jinja2. Routes: onboarding, dashboard (live job log), clips gallery,
settings. A single background job runner (one job at a time) drives the daily
pipeline and logs into an in-memory ring buffer polled by the UI every 2 s.
Crash-safety comes from the DB, not from here (see job.py / db.py).
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config as cfg
from . import db, discovery, job as jobmod, prompts, selector, downloader, analyzer, cutter, capcut_export
from . import events, notifier, scheduler as sched_mod, suggest, health

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))


# --- job runner -------------------------------------------------------------
class JobRunner:
    """Exactly one background job at a time; ring-buffered log for the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.log: deque[str] = deque(maxlen=500)
        self.running = False
        self.started_at: float | None = None
        self.kind: str | None = None
        self.result: list[str] = []
        self.error: str | None = None

    def _log(self, msg: str) -> None:
        self.log.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def start(self, kind: str, fn) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.kind = kind
            self.error = None
            self.result = []
            self.started_at = time.time()
            self.log.clear()
            db.set_state("job_active", "1")
            db.set_state("last_run_date", datetime.now().date().isoformat())

            def runner():
                try:
                    self.result = fn(self._log) or []
                except Exception as e:  # pragma: no cover
                    self.error = str(e)
                    self._log(f"ERROR: {e}")
                finally:
                    self.running = False
                    db.set_state("job_active", "0")
                    with self._lock:
                        self._thread = None
                try:
                    _post_run_hooks(self._log)
                except Exception as e:  # pragma: no cover
                    self._log(f"post-run hook error: {e}")

            self._thread = threading.Thread(target=runner, daemon=True)
            self._thread.start()
            return True

    def status(self) -> dict:
        c = cfg.get_config()
        return {
            "running": self.running,
            "kind": self.kind,
            "error": self.error,
            "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "clips_today": db.clips_today(),
            "quota": int(c.get("job.daily_quota", 4)),
            "log": list(self.log),
        }


runner = JobRunner()


# --- app --------------------------------------------------------------------
app = FastAPI(title="ClipForge", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
cfg.ensure_dirs()
db.init_db()
db.reset_run_state()  # clear any stale job_active from a previous crash
app.mount("/output", StaticFiles(directory=str(cfg.OUTPUT_DIR)), name="output")

_CANDIDATE_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 600.0


# --- proactive hooks --------------------------------------------------------
def _log_auto(record: dict) -> None:
    try:
        cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (cfg.LOGS_DIR / "auto_actions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **record}) + "\n")
    except OSError:
        pass


def auto_review_pass(log=lambda *_a: None) -> int:
    """review.mode='auto': approve unreviewed clips older than auto_approve_hours."""
    c = cfg.get_config()
    if str(c.get("review.mode", "manual")).lower() != "auto":
        return 0
    hours = float(c.get("review.auto_approve_hours", 24) or 24)
    n = 0
    for clip in db.unreviewed_clips(older_than_hours=hours):
        if clip["render_mode"] == "capcut":
            continue  # drafts can't be auto-'kept' (no file to approve)
        db.approve_clip(clip["id"])
        try:
            newpath = cutter.move_to_approved(clip["path"])
            if newpath:
                db.execute("UPDATE clips SET path=? WHERE id=?", (str(newpath), clip["id"]))
        except OSError:
            pass
        _log_auto({"action": "auto_approve", "clip_id": clip["id"], "engine": clip["engine"]})
        n += 1
    if n:
        log(f"[auto-review] approved {n} clip(s) after {hours:.0f}h")
        events.job_progress("auto_review", f"approved {n}", 1.0)
    return n


def _post_run_hooks(log) -> None:
    auto_review_pass(log)
    health.run_all(log)
    suggest.run_all(log)


def _try_prestage() -> None:
    if runner.running or db.get_state("prestage_active") == "1":
        return
    db.set_state("prestage_active", "1")

    def _log(msg: str) -> None:
        events.job_progress("prestage", msg, 0.5)

    def _run() -> None:
        try:
            from . import prestaging
            prestaging.prestage_once(_log)
        except Exception as e:  # pragma: no cover
            notifier.notify("prestage", "warn", "Pre-staging failed", str(e)[:200])
        finally:
            db.set_state("prestage_active", "0")

    threading.Thread(target=_run, daemon=True).start()


# --- scheduler lifecycle (guard double-start) --------------------------------
@app.on_event("startup")
def _startup() -> None:
    sched_mod.configure(daily=_start_daily_job, prestage=_try_prestage)
    sched_mod.scheduler.start()
    # boot-time checks (spec: at boot)
    auto_review_pass()
    health.run_all()
    suggest.run_all()


@app.on_event("shutdown")
def _shutdown() -> None:
    sched_mod.scheduler.shutdown()


def _start_daily_job() -> bool:
    """Called by APScheduler + catch-up; returns whether it started."""
    return runner.start("daily", jobmod.daily_job)


def _ensure_restart_state() -> None:
    # surface a pending 'restart required' banner via state (read in base_ctx)
    return None


def candidates_for(topic: str, refresh: bool = False, deep: bool = False) -> list[dict]:
    now = time.time()
    hit = _CANDIDATE_CACHE.get(topic)
    if hit and not refresh and not deep and (now - hit[0]) < _CACHE_TTL:
        return hit[1]
    try:
        cands = discovery.discover_candidates(topic, deep=deep)
    except Exception:
        cands = []
    _CANDIDATE_CACHE[topic] = (now, cands)
    return cands


def _topics_meta() -> list[dict]:
    c = cfg.get_config()
    topics = c.get("discovery.topics", {}) or {}
    order = c.get("discovery.rotation_order", []) or list(topics.keys())
    out = []
    for key in order:
        cfg_t = topics.get(key, {})
        out.append({
            "key": key,
            "label": cfg_t.get("label", key.replace("_", " ").title()),
            "pick": int(cfg_t.get("pick", 0)),
            "candidates": int(cfg_t.get("candidates", 0)),
            "mode": cfg_t.get("mode", "auto"),
        })
    return out


def base_ctx(request: Request, **kw) -> dict:
    c = cfg.get_config()
    ctx = {
        "request": request,
        "quota": int(c.get("job.daily_quota", 4)),
        "clips_today": db.clips_today(),
        "has_channels": db.channel_count() > 0,
        "default_prompt": db.get_state("default_prompt", None) or prompts.DEFAULT_PROMPT,
        "version": "1.0.0",
        # proactive layer context (header bell, schedule banner)
        "unread": db.unread_count(),
        "schedule_enabled": bool(c.get("schedule.enabled", True)),
        "schedule_time": c.get("schedule.time", "06:00"),
        "next_run_countdown": sched_mod.scheduler.next_run_countdown(),
        "next_run_iso": sched_mod.scheduler.next_run_iso(),
        "restart_required": db.get_state("restart_required", None),
        "review_mode": c.get("review.mode", "manual"),
    }
    ctx.update(kw)
    return ctx


# --- routing ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    if db.channel_count() == 0:
        return RedirectResponse("/onboarding")
    return RedirectResponse("/dashboard")


@app.get("/onboarding", response_class=HTMLResponse)
def onboarding(request: Request, refresh: int = 0, deep: int = 0):
    """Renders instantly; each topic's cards are fetched from /api/candidates."""
    topics = _topics_meta()
    saved = {t["key"]: [dict(r) for r in db.channels_by_topic(t["key"], only_active=True)]
             for t in topics}
    return TEMPLATES.TemplateResponse(request, "onboarding.html", base_ctx(
        request, topics=topics, saved=saved, deep=bool(deep),
        topics_json=json.dumps({}), active="onboarding"))


@app.get("/api/candidates")
def api_candidates(topic: str, refresh: int = 0, deep: int = 0):
    """Per-topic candidate JSON so the browser loads each column async."""
    valid = {t["key"] for t in _topics_meta()}
    if topic not in valid:
        raise HTTPException(status_code=404, detail="unknown topic")
    cards = candidates_for(topic, refresh=bool(refresh), deep=bool(deep))
    return JSONResponse({"topic": topic, "cards": cards})


@app.post("/onboarding")
async def onboarding_save(request: Request):
    """Save picked channels (+ the rest of the candidate pool, deactivated).

    Enforces exact per-topic pick counts (spec §4/§11.2). All candidates shown at
    onboarding are persisted so the yield-swap suggestion has a pool to promote
    from; only picked ones are is_active=1.
    """
    payload = await request.json()
    selected: dict[str, list[dict]] = payload.get("selected", {}) or {}
    pool: dict[str, list[dict]] = payload.get("pool", {}) or {}
    required = {t["key"]: t["pick"] for t in _topics_meta()}
    errors = []
    for key, need in required.items():
        got = len(selected.get(key, []) or [])
        if got != need:
            errors.append(f"{key}: got {got}, need exactly {need}")
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    saved = active = 0
    for key, chans in selected.items():
        active_ids = {str(c.get("platform_channel_id")) for c in chans}
        # merge picked + remaining pool candidates for this topic into one set
        merged: dict[str, dict] = {}
        for ch in pool.get(key, []) or []:
            cid = str(ch.get("platform_channel_id"))
            merged[cid] = ch
        for ch in chans:
            cid = str(ch.get("platform_channel_id"))
            merged[cid] = ch
        db.execute("DELETE FROM videos WHERE channel_id IN "
                   "(SELECT id FROM channels WHERE topic=?)", (key,))
        db.execute("DELETE FROM channels WHERE topic=?", (key,))
        for cid, ch in merged.items():
            is_active = 1 if cid in active_ids else 0
            db.add_channel(
                cid, str(ch.get("title") or "unknown"), key, ch.get("subtopic"),
                _to_int(ch.get("subs")), _to_int(ch.get("video_count")),
                _to_int(ch.get("joined_year")), ch.get("avatar_url"), is_active=is_active,
            )
            saved += 1
            active += is_active
    db.set_int_state("rotation_index", 0)
    return JSONResponse({"ok": True, "saved": saved, "active": active,
                         "redirect": "/dashboard"})


def _to_int(v) -> int | None:
    try:
        return int(float(v)) if v not in (None, "", "null") else None
    except (TypeError, ValueError):
        return None


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if db.channel_count() == 0:
        return RedirectResponse("/onboarding")
    c = cfg.get_config()
    channels = [dict(r) for r in db.active_channels()]

    # last batch = clips from the most recent run day that still need a decision
    last_day = db.get_state("last_run_date", None)
    last_batch = []
    if last_day:
        for r in db.clips_by_date(last_day):
            d = dict(r)
            d["url"] = _outurl(d["path"]) if d["path"] else None
            last_batch.append(d)

    # tonight's plan = staged videos
    tonight = []
    for st in db.staged_videos():
        tonight.append({"video_id": st["video_id"], "title": st["title"],
                        "topic": st["topic"], "channel_id": st["channel_id"],
                        "channel_title": st["channel_title"], "duration_s": st["duration_s"],
                        "thumbnail": st["thumbnail"],
                        "windows": len(analyzer.cached_windows(st["video_id"]))})

    # review pending > 48h
    pending = [dict(r) for r in db.unreviewed_clips(older_than_hours=48)]

    # advisory cards: unread suggestion/health notifications (with actions)
    cards = []
    for n in db.list_notifications(limit=20):
        if n["kind"] in ("suggestion", "health", "quota_unmet") and not n["acted_at"]:
            d = _notif_dict(n)
            cards.append(d)

    return TEMPLATES.TemplateResponse(request, "dashboard.html", base_ctx(
        request, channels=channels, topics=_topics_meta(),
        default_count=selector.windows_for_duration(600),
        models={"text": c.get("openrouter.models.text"),
                "vision": c.get("openrouter.models.vision")},
        capcut=capcut_export.detect_draft_root(),
        output_mode=db.get_state("output_mode", "mp4"),
        last_batch=last_batch, tonight=tonight, pending=pending, cards=cards,
        today=datetime.now().strftime("%A, %d %B %Y"),
        has_key=bool(c.openrouter_api_key),
        active="dashboard"))


def _valid_output(mode: str | None) -> str:
    return mode if mode in ("mp4", "capcut") else "mp4"


def _current_output_mode() -> str:
    mode = db.get_state("output_mode", "mp4") or "mp4"
    # capcut only offered when the library + writable root are available
    if mode == "capcut" and not capcut_export.detect_draft_root()["writable"]:
        return "mp4"
    return mode


@app.post("/run")
async def run_daily(request: Request):
    mode = "mp4"
    try:
        body = await request.json()
        mode = _valid_output(body.get("output"))
    except Exception:
        pass
    if mode == "capcut" and not capcut_export.detect_draft_root()["writable"]:
        raise HTTPException(status_code=400, detail="CapCut draft root not writable — see Settings")
    db.set_state("output_mode", mode)  # remember last used
    if not runner.start("daily", (lambda log: jobmod.daily_job(log, output_mode=mode))):
        raise HTTPException(status_code=409, detail="A job is already running")
    return JSONResponse({"started": True, "output": mode})


@app.post("/run/custom")
async def run_custom(request: Request):
    body = await request.json()
    url = (body.get("url") or "").strip() or None
    count = _to_int(body.get("count"))
    prompt = (body.get("prompt") or "").strip() or None
    mode = _valid_output(body.get("output"))
    # an unchanged/blank prompt means "use the persisted default"
    if prompt and prompt.strip() == prompts.DEFAULT_PROMPT.strip():
        prompt = None
    if mode == "capcut" and not capcut_export.detect_draft_root()["writable"]:
        raise HTTPException(status_code=400, detail="CapCut draft root not writable — see Settings")
    db.set_state("output_mode", mode)
    fn = (lambda log: jobmod.custom_job(url, log, clip_count=count,
                                        prompt_override=prompt, output_mode=mode))
    if not runner.start("custom", fn):
        raise HTTPException(status_code=409, detail="A job is already running")
    return JSONResponse({"started": True, "output": mode})


@app.get("/run/status")
def run_status():
    return JSONResponse(runner.status())


# --- SSE live updates (feature P1, spec §3.4) -------------------------------
@app.get("/events")
async def sse(request: Request):
    sub = events.bus.subscribe()
    return StreamingResponse(events.stream(sub), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no", "Connection": "keep-alive"})


# --- notifications (feature P2, spec §3.5) ----------------------------------
@app.get("/api/notifications")
def api_notifications(unread: int = 0):
    rows = db.list_notifications(unread_only=bool(unread))
    return JSONResponse({"unread": db.unread_count(),
                         "notifications": [_notif_dict(r) for r in rows]})


def _notif_dict(r) -> dict:
    import json as _json
    d = dict(r)
    try:
        d["actions"] = _json.loads(d.pop("actions_json", None) or "[]")
    except (TypeError, _json.JSONDecodeError):
        d["actions"] = []
    return d


@app.post("/api/notifications/{notif_id}/read")
async def api_notif_read(notif_id: int, request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    db.mark_notification_read(notif_id, acted=bool(body.get("acted")))
    return JSONResponse({"ok": True})


# --- action dispatch (suggestions + health). DESTRUCTIVE -> requires confirm. -
_DESTRUCTIVE = {"update_ytdlp", "cleanup_sources", "cleanup_stale", "discard_clip",
                "swap_channel_revert"}


@app.post("/api/actions/{action}")
async def api_action(action: str, request: Request):
    body = await request.json()
    confirm = bool(body.get("confirm"))
    args = body.get("args", {}) or {}
    notif_id = body.get("notif_id")

    if action in _DESTRUCTIVE and not confirm:
        # HARD RULE: nothing destructive runs without an explicit confirm flag.
        raise HTTPException(status_code=400, detail="confirmation required")

    result: dict = {"ok": True}
    if action == "update_ytdlp":
        ok, msg = health.update_ytdlp(lambda m: events.job_progress("update", m, 0.5))
        result = {"ok": ok, "message": msg, "restart_required": ok}
    elif action == "swap_channel":
        a_id, b_id = int(args["deactivate_id"]), int(args["activate_id"])
        db.swap_channels(a_id, b_id)
        db.set_int_state("rotation_index", 0)  # rotation re-derives from active set
        events.schedule_changed(swapped=True)
        result = {"ok": True, "deactivated": a_id, "activated": b_id}
    elif action == "apply_prompt_tune":
        line = str(args.get("line", ""))
        _apply_topic_prompt(args.get("topic"), line)
        _log_auto({"action": "apply_prompt_tune", "topic": args.get("topic"), "line": line})
        result = {"ok": True, "applied": True}
    elif action in ("cleanup_sources", "cleanup_stale"):
        ids = args.get("video_ids")
        freed = _cleanup_sources(ids)
        _log_auto({"action": action, "video_ids": ids or "stale", "freed_bytes": freed})
        result = {"ok": True, "freed_gb": round(freed / 1e9, 2), "count": len(ids or []) or None}
    elif action == "run_daily":
        if not runner.start("daily", jobmod.daily_job):
            raise HTTPException(status_code=409, detail="A job is already running")
        result = {"ok": True, "started": True}
    elif action == "reduce_quota":
        c = cfg.get_config()
        new_q = max(1, int(c.get("job.daily_quota", 4)) - 1)
        c.raw.setdefault("job", {})["daily_quota"] = new_q
        db.set_state("ov.job.daily_quota", str(new_q))
        result = {"ok": True, "quota": new_q}
    elif action == "prestage_retry":
        _try_prestage()
        result = {"ok": True, "started": True}
    elif action == "dismiss":
        result = {"ok": True}
    elif action == "nav":
        result = {"ok": True, "to": args.get("to", "/")}
    else:
        raise HTTPException(status_code=404, detail="unknown action")

    if notif_id:
        db.mark_notification_read(int(notif_id), acted=True)
    return JSONResponse(result)


def _apply_topic_prompt(topic: str, line: str) -> None:
    if not topic or not line:
        return
    cur = db.get_state(f"topic_prompt:{topic}", "") or ""
    db.set_state(f"topic_prompt:{topic}", (cur + ("\n" if cur else "") + line))


def _cleanup_sources(video_ids: list | None) -> int:
    """Delete resolved source files for the given (or all stale reviewed) videos."""
    if not video_ids:
        # stale reviewed sources: > cleanup_days, all clips reviewed
        c = cfg.get_config()
        cutoff = (datetime.now() - timedelta(days=int(c.get("review.cleanup_days", 7) or 7))).isoformat(timespec="seconds")
        video_ids = [v["video_id"] for v in db.query(
            "SELECT DISTINCT v.video_id FROM videos v JOIN clips c ON c.video_id=v.video_id "
            "WHERE v.used_at IS NOT NULL AND v.used_at<=? "
            "AND NOT EXISTS (SELECT 1 FROM clips x WHERE x.video_id=v.video_id AND x.revised_at IS NULL AND x.status='pending')",
            (cutoff,))]
    freed = 0
    for vid in video_ids or []:
        src = downloader.resolve_source(vid)
        if src and Path(src).is_file():
            freed += Path(src).stat().st_size
            try:
                Path(src).unlink()
                Path(src).with_suffix(".16k.wav").unlink(missing_ok=True)
            except OSError:
                pass
    return freed


# --- clip review decisions (Keep / Discard / Fill) --------------------------
@app.post("/api/clips/{clip_id}/keep")
def api_clip_keep(clip_id: int):
    clip = db.get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="no such clip")
    if clip["status"] == "discarded":
        raise HTTPException(status_code=409, detail="clip discarded")
    db.approve_clip(clip_id)
    newpath = cutter.move_to_approved(clip["path"])
    if newpath:
        db.execute("UPDATE clips SET path=? WHERE id=?", (str(newpath), clip_id))
    return JSONResponse({"ok": True, "id": clip_id, "status": "approved"})


@app.post("/api/clips/{clip_id}/discard")
async def api_clip_discard(clip_id: int, request: Request):
    body = await request.json()
    if not bool(body.get("confirm")):  # HARD RULE: discard needs explicit confirm
        raise HTTPException(status_code=400, detail="confirmation required")
    clip = db.get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="no such clip")
    if clip["path"]:
        try:
            Path(clip["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    db.discard_clip(clip_id)   # video stays 'used'; quota is NOT refunded
    return JSONResponse({"ok": True, "id": clip_id, "status": "discarded",
                         "quota": int(cfg.get_config().get("job.daily_quota", 4))})


@app.get("/api/fill-candidate")
def api_fill_candidate():
    """A staged video available to fill a slot, if quota not yet met."""
    quota = int(cfg.get_config().get("job.daily_quota", 4))
    if db.clips_today() >= quota:
        return JSONResponse({"available": False, "reason": "quota met"})
    for st in db.staged_videos():
        return JSONResponse({"available": True, "video_id": st["video_id"],
                             "channel_id": st["channel_id"]})
    return JSONResponse({"available": False, "reason": "no staged videos — pre-staging will run tonight"})


@app.post("/api/fill-slot")
async def api_fill_slot(request: Request):
    body = await request.json() if await request.body() else {}
    video_id = body.get("video_id")
    quota = int(cfg.get_config().get("job.daily_quota", 4))
    if db.clips_today() >= quota:
        raise HTTPException(status_code=400, detail="quota already met")
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id required")
    vrow = db.video_with_channel(video_id)
    if not vrow or vrow["status"] != "staged":
        raise HTTPException(status_code=400, detail="video not staged")
    channel = {"id": vrow["channel_id"], "title": vrow["channel_title"], "topic": vrow["topic"]}

    def _run(log):
        return jobmod.process_video(video_id, channel, None, log, staged=True,
                                    output_mode=db.get_state("output_mode", "mp4"))

    if not runner.start("fill", _run):
        raise HTTPException(status_code=409, detail="A job is already running")
    return JSONResponse({"started": True, "video_id": video_id})


@app.post("/api/channels/{channel_id}/swap")
async def api_swap_channel(channel_id: int, request: Request):
    """One-click [Swap] from a yield suggestion. Activates a stored candidate."""
    body = await request.json() if await request.body() else {}
    activate_id = _to_int(body.get("activate_id"))
    if not activate_id:
        raise HTTPException(status_code=400, detail="activate_id required")
    old = db.query_one("SELECT * FROM channels WHERE id=?", (channel_id,))
    new = db.query_one("SELECT * FROM channels WHERE id=?", (activate_id,))
    if not old or not new:
        raise HTTPException(status_code=404, detail="unknown channel")
    db.swap_channels(int(channel_id), int(activate_id))
    db.set_int_state("rotation_index", 0)
    events.schedule_changed(swapped=True)
    return JSONResponse({"ok": True, "deactivated": old["title"], "activated": new["title"]})


@app.get("/api/prestaging")
def api_prestaging():
    """Tonight's plan: staged videos + candidate window counts."""
    out = []
    for st in db.staged_videos():
        windows = len(analyzer.cached_windows(st["video_id"])) if hasattr(analyzer, "cached_windows") else 0
        out.append({"video_id": st["video_id"], "title": st["title"], "topic": st["topic"],
                    "channel_id": st["channel_id"], "channel_title": st["channel_title"],
                    "duration_s": st["duration_s"], "thumbnail": st["thumbnail"],
                    "windows": windows})
    return JSONResponse({"staged": out})


@app.post("/api/prestaging/swap")
async def api_prestage_swap(request: Request):
    """[Swap] on tonight's plan: replace a staged video with the next-best candidate."""
    body = await request.json()
    channel_id = _to_int(body.get("channel_id"))
    drop_id = str(body.get("video_id") or "")
    if not channel_id:
        raise HTTPException(status_code=400, detail="channel_id required")
    ch = db.query_one("SELECT * FROM channels WHERE id=?", (channel_id,))
    if not ch:
        raise HTTPException(status_code=404, detail="unknown channel")
    if drop_id:
        _drop_staged(drop_id)
    # next-best not-yet-known video for that channel
    picked = selector.pick_video(dict(ch))
    if picked:
        from . import prestaging as _ps
        _ps._stage_channel(dict(ch), lambda *_a: None)
    return JSONResponse({"ok": True, "dropped": drop_id or None})


def _drop_staged(video_id: str) -> None:
    src = downloader.resolve_source(video_id)
    if src:
        try:
            Path(src).unlink(missing_ok=True)
        except OSError:
            pass
    db.delete_video(video_id)


# --- schedule toggle --------------------------------------------------------
@app.post("/api/schedule")
async def api_schedule(request: Request):
    body = await request.json()
    enabled = bool(body.get("enabled"))
    c = cfg.get_config()
    c.raw.setdefault("schedule", {})["enabled"] = enabled
    db.set_state("ov.schedule.enabled", "true" if enabled else "false")
    sched_mod.scheduler.apply_schedule()
    events.schedule_changed(enabled=enabled, next_run=sched_mod.scheduler.next_run_iso())
    return JSONResponse({"ok": True, "enabled": enabled,
                         "next_run": sched_mod.scheduler.next_run_iso()})


@app.get("/clips", response_class=HTMLResponse)
@app.get("/review", response_class=HTMLResponse)
def clips(request: Request):
    """Gallery grouped by day + in-place review editor (feature C)."""
    rows = [dict(r) for r in db.current_clips()]
    vsrc = {}
    for r in rows:
        vid = r["video_id"]
        if vid not in vsrc:
            vsrc[vid] = downloader.resolve_source(vid) is not None
        r["has_source"] = vsrc[vid]
        r["duration_url"] = f"/media/source/{vid}"
    # videos eligible for a brand-new manual clip
    eligible = [dict(v) for v in db.query(
        "SELECT video_id, title, duration_s FROM videos "
        "WHERE status IN ('done','analyzed') ORDER BY used_at DESC LIMIT 200")]
    for v in eligible:
        v["has_source"] = downloader.resolve_source(v["video_id"]) is not None
    groups: dict[str, list[dict]] = {}
    for r in rows:
        day = (r.get("created_at") or "")[:10]
        groups.setdefault(day, []).append(r)
    return TEMPLATES.TemplateResponse(request, "clips.html", base_ctx(
        request, groups=groups, total=len(rows), eligible=eligible,
        clip_max=int(cfg.get_config().get("clip.length_s", 60)),
        capcut=capcut_export.detect_draft_root(),
        output_mode=db.get_state("output_mode", "mp4"), active="clips"))


# --- review media + manual clips (feature C) -------------------------------
def _parse_range(header: str | None, total: int) -> tuple[int, int] | None:
    """Parse a single 'bytes=a-b' / 'bytes=a-' / 'bytes=-n' range. None = full."""
    if not header or not header.strip().lower().startswith("bytes="):
        return None
    spec = header[6:].split(",")[0].strip()
    if "-" not in spec:
        return None
    s, _, e = spec.partition("-")
    try:
        if s == "" and e == "":
            return None
        if s == "":                      # suffix: last n bytes
            n = int(e)
            start, end = max(0, total - n), total - 1
        else:
            start = int(s)
            end = int(e) if e else total - 1
    except ValueError:
        return None
    end = min(end, total - 1)
    if start < 0 or start > end or start >= total:
        raise HTTPException(status_code=416, detail="range not satisfiable")
    return start, end


def _stream_file(path: Path, start: int, end: int):
    chunk = 64 * 1024
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = fh.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@app.get("/media/source/{video_id}")
def media_source(video_id: str, request: Request):
    """Stream the downloaded source with manual HTTP Range (206) so seeking works."""
    src = downloader.resolve_source(video_id)
    if not src or not src.is_file():
        raise HTTPException(status_code=404, detail="source not downloaded")
    total = src.stat().st_size
    ctype = mimetypes.guess_type(str(src))[0] or "video/mp4"
    rng = _parse_range(request.headers.get("range"), total)
    if rng is None:
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(total)}
        return StreamingResponse(_stream_file(src, 0, total - 1),
                                 media_type=ctype, headers=headers)
    start, end = rng
    length = end - start + 1
    headers = {"Accept-Ranges": "bytes",
               "Content-Range": f"bytes {start}-{end}/{total}",
               "Content-Length": str(length)}
    return StreamingResponse(_stream_file(src, start, end),
                             status_code=206, media_type=ctype, headers=headers)


@app.get("/api/videos")
def api_videos():
    """Videos a manual clip can be cut from (status done/analyzed with a source)."""
    rows = db.query("SELECT video_id, title, duration_s, status FROM videos "
                    "WHERE status IN ('done','analyzed') ORDER BY used_at DESC LIMIT 200")
    out = []
    for r in rows:
        if downloader.resolve_source(r["video_id"]):
            out.append({"video_id": r["video_id"], "title": r["title"],
                        "duration_s": r["duration_s"], "status": r["status"]})
    return JSONResponse({"videos": out})


def _overlaps_any(video_id: str, s: float, e: float, exclude_id: int | None) -> bool:
    for r in db.video_clips_active(video_id, exclude_id):
        os_, oe = float(r["start_s"]), float(r["end_s"])
        if not (e <= os_ or s >= oe):
            return True
    return False


@app.post("/api/clips")
async def api_create_clip(request: Request):
    """Create or revise a clip by re-rendering a window of its source (feature C)."""
    if not cfg.get_config().ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg unavailable")
    body = await request.json()
    video_id = str(body.get("video_id") or "")
    try:
        s = round(float(body.get("start_s")), 2)
        e = round(float(body.get("end_s")), 2)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="start_s/end_s required")
    caption = str(body.get("caption") or "")[:150]
    hook = str(body.get("hook_title") or "")[:40]
    mode = (body.get("mode") or "new").lower()
    clip_id = body.get("clip_id")
    clip_id = int(clip_id) if clip_id not in (None, "", "null") else None

    vrow = db.query_one("SELECT * FROM videos WHERE video_id=?", (video_id,))
    if not vrow:
        raise HTTPException(status_code=404, detail="unknown video_id")
    dur = float(vrow["duration_s"] or 0) or 0
    errors = []
    if e - s < 5:
        errors.append("clip must be at least 5s")
    if e - s > 60:
        errors.append("clip must be at most 60s")
    if s < 0 or (dur and e > dur + 0.5):
        errors.append("window outside video bounds")
    if _overlaps_any(video_id, s, e, clip_id if mode == "replace" else None):
        errors.append("overlaps another clip of this video")
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    src = downloader.resolve_source(video_id)
    if not src:
        raise HTTPException(status_code=404, detail="source file missing; re-download first")
    # decide output mode: explicit body > existing clip (on replace) > current default
    existing = db.get_clip(clip_id) if clip_id else None
    render_mode = str(body.get("render_mode") or "").lower() or \
        (existing["render_mode"] if existing and existing["render_mode"] else
         db.get_state("output_mode", "mp4"))
    if render_mode not in ("mp4", "capcut"):
        render_mode = "mp4"

    if mode == "replace" and clip_id and db.get_clip(clip_id):
        db.mark_clip_revised(clip_id)
        engine, parent = "review", clip_id
    else:
        engine, parent = "manual", None

    clip = {"start_s": s, "end_s": e, "caption": caption, "hook_title": hook, "engine": engine}

    if render_mode == "capcut":
        report = capcut_export.detect_draft_root()
        if not report["available"]:
            raise HTTPException(status_code=503, detail=report["error"] or "CapCut export unavailable")
        if not report["writable"] or not report["draft_root"]:
            raise HTTPException(status_code=503,
                                detail=report["error"] or "draft root unwritable — mp4 mode still works")
        title = vrow["title"] or "video"
        try:
            draft = capcut_export.export_draft(
                {"video_id": video_id, "title": title, "duration_s": dur, "_clip_index": 0},
                [clip], src_path=src)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"draft export failed: {e}")
        # new draft succeeded; now remove the superseded folder (no orphans)
        if existing and existing["render_mode"] == "capcut" and existing["draft_path"] != draft:
            capcut_export.remove_draft(existing["draft_path"])
        new_id = db.add_clip(video_id, s, e, None, caption, engine, "user",
                             engine=engine, parent_clip_id=parent, hook_title=hook,
                             render_mode="capcut", draft_path=draft)
    else:
        if not cfg.get_config().ffmpeg:
            raise HTTPException(status_code=503, detail="ffmpeg unavailable")
        words = analyzer.cached_words(video_id)
        out = cutter.review_path_for(video_id)
        path = cutter.render_clip(src, s, e, out, words=words)
        if not path:
            raise HTTPException(status_code=500, detail="render failed")
        new_id = db.add_clip(video_id, s, e, str(path), caption, engine, "user",
                             engine=engine, parent_clip_id=parent, hook_title=hook,
                             render_mode="mp4")

    row = db.get_clip(new_id)
    log_line = f"review: video {video_id} {s:.1f}-{e:.1f}s [{engine}/{render_mode}]"
    (cfg.LOGS_DIR / "review.log").open("a", encoding="utf-8").write(
        f"{datetime.now().isoformat(timespec='seconds')} {log_line}\n")
    return JSONResponse({"clip": {k: row[k] for k in row.keys()},
                         "url": _outurl(str(row["path"])) if row["path"] else None,
                         "engine": engine, "render_mode": render_mode})


def _agent_ctx(c) -> dict:
    return {
        "enabled": bool(c.get("agent.enabled", True)),
        "model": c.get("agent.model", "google/gemini-2.5-flash"),
        "max_steps": int(c.get("agent.max_steps", 8)),
        "vision_enabled": bool(c.get("agent.vision_enabled", False)),
    }


@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request):
    c = cfg.get_config()
    return TEMPLATES.TemplateResponse(request, "settings.html", base_ctx(
        request,
        cfg_models={"text": c.get("openrouter.models.text"),
                    "vision": c.get("openrouter.models.vision")},
        whisper_model=c.get("whisper.model"),
        clip_length=c.get("clip.length_s", 60),
        face_tracking=c.get("clip.face_tracking"),
        agent=_agent_ctx(c),
        proactive=_proactive_ctx(c),
        capcut=capcut_export.detect_draft_root(),
        capcut_version=capcut_export.VERSION_NOTE,
        output_mode=db.get_state("output_mode", "mp4"),
        has_key=bool(c.openrouter_api_key),
        test_result=None,
        active="settings"))


@app.post("/settings/capcut_test")
def settings_capcut_test():
    ok, msg = capcut_export.test_write_draft()
    return JSONResponse({"ok": ok, "message": msg})


@app.post("/settings/output")
async def settings_output(request: Request):
    """Persist the last-used output mode so it survives reloads (radio choice)."""
    body = await request.json()
    out = str(body.get("output") or "").lower()
    if out not in ("mp4", "capcut"):
        raise HTTPException(status_code=400, detail="output must be mp4 or capcut")
    db.set_state("output_mode", out)
    return JSONResponse({"ok": True, "output": out})


@app.post("/settings")
def settings_save(
    request: Request,
    text_model: str = Form(...), vision_model: str = Form(...),
    whisper_model: str = Form(...), daily_quota: str = Form("4"),
    clip_length: str = Form("60"), face_tracking: str = Form("auto"),
    default_prompt: str = Form(""),
    agent_model: str = Form("google/gemini-2.5-flash"),
    agent_enabled: str = Form("off"), agent_vision: str = Form("off"),
    schedule_time: str = Form("06:00"), schedule_enabled: str = Form("off"),
    prestaging_enabled: str = Form("off"), hours_before: str = Form("10"),
    review_mode: str = Form("manual"), auto_approve_hours: str = Form("24"),
    cleanup_days: str = Form("7"),
    webhook_url: str = Form(""), telegram_bot_token: str = Form(""),
    telegram_chat_id: str = Form(""),
):
    _persist_settings(text_model, vision_model, whisper_model, daily_quota,
                      clip_length, face_tracking, default_prompt,
                      agent_model=agent_model, agent_enabled=agent_enabled,
                      agent_vision=agent_vision,
                      schedule_time=schedule_time, schedule_enabled=schedule_enabled,
                      prestaging_enabled=prestaging_enabled, hours_before=hours_before,
                      review_mode=review_mode, auto_approve_hours=auto_approve_hours,
                      cleanup_days=cleanup_days, webhook_url=webhook_url,
                      telegram_bot_token=telegram_bot_token,
                      telegram_chat_id=telegram_chat_id)
    c = cfg.get_config()
    return TEMPLATES.TemplateResponse(request, "settings.html", base_ctx(
        request, cfg_models={"text": c.get("openrouter.models.text"),
                             "vision": c.get("openrouter.models.vision")},
        whisper_model=c.get("whisper.model"), face_tracking=c.get("clip.face_tracking"),
        clip_length=c.get("clip.length_s"), agent=_agent_ctx(c), proactive=_proactive_ctx(c),
        has_key=bool(c.openrouter_api_key),
        test_result="Settings saved.", active="settings"))


def _proactive_ctx(c) -> dict:
    return {
        "schedule_enabled": bool(c.get("schedule.enabled", True)),
        "time": c.get("schedule.time", "06:00"),
        "catch_up": bool(c.get("schedule.catch_up", True)),
        "prestage_enabled": bool(c.get("prestaging.enabled", True)),
        "hours_before": int(c.get("prestaging.hours_before", 10)),
        "review_mode": c.get("review.mode", "manual"),
        "auto_approve_hours": int(c.get("review.auto_approve_hours", 24)),
        "cleanup_days": int(c.get("review.cleanup_days", 7)),
        "webhook_url": c.get("notifications.webhook_url", "") or "",
        "telegram_bot_token": c.get("notifications.telegram_bot_token", "") or "",
        "telegram_chat_id": c.get("notifications.telegram_chat_id", "") or "",
    }


@app.post("/settings/test_key")
def settings_test_key():
    from .editor_ai import test_key
    ok, msg = test_key()
    return JSONResponse({"ok": ok, "message": msg})


@app.post("/settings/test_notification")
def settings_test_notification():
    ok, msg = notifier.test_message()
    return JSONResponse({"ok": ok, "message": msg})


def _persist_settings(text_model, vision_model, whisper_model, daily_quota,
                      clip_length, face_tracking, default_prompt,
                      agent_model="google/gemini-2.5-flash", agent_enabled="on",
                      agent_vision="", schedule_time="06:00", schedule_enabled="off",
                      prestaging_enabled="off", hours_before="10", review_mode="manual",
                      auto_approve_hours="24", cleanup_days="7", webhook_url="",
                      telegram_bot_token="", telegram_chat_id="") -> None:
    """Write model/quota/clip/prompt/agent/schedule/review/notification overrides
    into config + DB state (§11.8). config.yaml stays the source of defaults;
    runtime overrides win at read time and persist across restarts."""
    c = cfg.get_config()
    enabled = "on" if (agent_enabled in ("on", "true", "1")) else "off"
    vision = "on" if (agent_vision in ("on", "true", "1")) else "off"
    sched_on = "on" if (schedule_enabled in ("on", "true", "1")) else "off"
    pre_on = "on" if (prestaging_enabled in ("on", "true", "1")) else "off"
    c.raw.setdefault("openrouter", {}).setdefault("models", {})
    c.raw["openrouter"]["models"]["text"] = text_model
    c.raw["openrouter"]["models"]["vision"] = vision_model
    c.raw.setdefault("whisper", {})["model"] = whisper_model
    c.raw.setdefault("job", {})["daily_quota"] = int(float(daily_quota or 4))
    c.raw.setdefault("clip", {})["length_s"] = int(float(clip_length or 60))
    c.raw["clip"]["face_tracking"] = "off" if face_tracking == "off" else "auto"
    ag = c.raw.setdefault("agent", {})
    ag["model"] = agent_model or "google/gemini-2.5-flash"
    ag["enabled"] = enabled == "on"
    ag["vision_enabled"] = vision == "on"
    sc = c.raw.setdefault("schedule", {})
    sc["enabled"] = sched_on == "on"
    sc["time"] = schedule_time or "06:00"
    ps = c.raw.setdefault("prestaging", {})
    ps["enabled"] = pre_on == "on"
    ps["hours_before"] = int(float(hours_before or 10))
    rv = c.raw.setdefault("review", {})
    rv["mode"] = "auto" if review_mode == "auto" else "manual"
    rv["auto_approve_hours"] = int(float(auto_approve_hours or 24))
    rv["cleanup_days"] = int(float(cleanup_days or 7))
    nt = c.raw.setdefault("notifications", {})
    nt["webhook_url"] = webhook_url.strip()
    nt["telegram_bot_token"] = telegram_bot_token.strip()
    nt["telegram_chat_id"] = telegram_chat_id.strip()
    overrides = {
        "ov.openrouter.models.text": text_model,
        "ov.openrouter.models.vision": vision_model,
        "ov.whisper.model": whisper_model,
        "ov.job.daily_quota": str(int(float(daily_quota or 4))),
        "ov.clip.length_s": str(int(float(clip_length or 60))),
        "ov.clip.face_tracking": "off" if face_tracking == "off" else "auto",
        "ov.agent.model": ag["model"],
        "ov.agent.enabled": "true" if ag["enabled"] else "false",
        "ov.agent.vision_enabled": "true" if ag["vision_enabled"] else "false",
        "ov.schedule.enabled": "true" if sc["enabled"] else "false",
        "ov.schedule.time": sc["time"],
        "ov.prestaging.enabled": "true" if ps["enabled"] else "false",
        "ov.prestaging.hours_before": str(ps["hours_before"]),
        "ov.review.mode": rv["mode"],
        "ov.review.auto_approve_hours": str(rv["auto_approve_hours"]),
        "ov.review.cleanup_days": str(rv["cleanup_days"]),
        "ov.notifications.webhook_url": nt["webhook_url"],
        "ov.notifications.telegram_bot_token": nt["telegram_bot_token"],
        "ov.notifications.telegram_chat_id": nt["telegram_chat_id"],
    }
    for k, v in overrides.items():
        db.set_state(k, str(v))
    if default_prompt.strip() and default_prompt.strip() != prompts.DEFAULT_PROMPT:
        db.set_state("default_prompt", default_prompt.strip())
    else:
        db.set_state("default_prompt", "")
    sched_mod.scheduler.apply_schedule()   # honour new time / enabled immediately


def apply_state_overlays() -> None:
    """Re-apply persisted settings overrides onto the live config (on startup)."""
    c = cfg.get_config()
    mapping = {
        "ov.openrouter.models.text": ["openrouter", "models", "text"],
        "ov.openrouter.models.vision": ["openrouter", "models", "vision"],
        "ov.whisper.model": ["whisper", "model"],
        "ov.job.daily_quota": ["job", "daily_quota"],
        "ov.clip.length_s": ["clip", "length_s"],
        "ov.clip.face_tracking": ["clip", "face_tracking"],
        "ov.agent.model": ["agent", "model"],
        "ov.agent.enabled": ["agent", "enabled"],
        "ov.agent.vision_enabled": ["agent", "vision_enabled"],
        "ov.schedule.enabled": ["schedule", "enabled"],
        "ov.schedule.time": ["schedule", "time"],
        "ov.prestaging.enabled": ["prestaging", "enabled"],
        "ov.prestaging.hours_before": ["prestaging", "hours_before"],
        "ov.review.mode": ["review", "mode"],
        "ov.review.auto_approve_hours": ["review", "auto_approve_hours"],
        "ov.review.cleanup_days": ["review", "cleanup_days"],
        "ov.notifications.webhook_url": ["notifications", "webhook_url"],
        "ov.notifications.telegram_bot_token": ["notifications", "telegram_bot_token"],
        "ov.notifications.telegram_chat_id": ["notifications", "telegram_chat_id"],
    }
    for key, path in mapping.items():
        val = db.get_state(key)
        if val is None:
            continue
        node = c.raw
        for p in path[:-1]:
            node = node.setdefault(p, {})
        last = path[-1]
        if last in ("daily_quota", "length_s", "hours_before", "auto_approve_hours",
                    "cleanup_days"):
            try:
                node[last] = int(float(val))
            except (TypeError, ValueError):
                pass
        elif last in ("enabled", "vision_enabled"):
            node[last] = val in ("true", "1", "on", "True")
        else:
            node[last] = val


apply_state_overlays()


# --- template filters -------------------------------------------------------
TEMPLATES.env.filters["human"] = lambda n: f"{(n or 0) / 1_000_000:.1f}M" if (n or 0) >= 1_000_000 else (f"{(n or 0) / 1000:.0f}K" if (n or 0) >= 1000 else str(n or 0))
TEMPLATES.env.filters["duration"] = lambda s: f"{int((s or 0) // 60)}:{int((s or 0) % 60):02d}"
TEMPLATES.env.filters["relpath"] = lambda p: Path(p).name if p else ""


def _outurl(p) -> str:
    """Map a stored absolute output path to the /output/ static URL."""
    if not p:
        return ""
    try:
        rel = Path(p).resolve().relative_to(Path(cfg.OUTPUT_DIR).resolve())
        return "/output/" + rel.as_posix()
    except (ValueError, OSError):
        return ""


TEMPLATES.env.filters["outurl"] = _outurl


# --- CLI entry: `python -m app.main --auto [--output capcut|mp4]` ------------
def _run_auto(output_mode: str | None = None) -> int:
    db.init_db()
    db.reset_run_state()
    if output_mode in ("mp4", "capcut"):
        db.set_state("output_mode", output_mode)
    else:
        output_mode = db.get_state("output_mode", "mp4")  # default: last used

    def log(msg: str) -> None:
        print(msg, flush=True)

    produced = jobmod.daily_job(log, output_mode=output_mode)
    kind = "CapCut drafts" if output_mode == "capcut" else "clips"
    print(f"\nProduced {kind}:")
    for p in produced:
        print(" ", p)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="clipforge")
    ap.add_argument("--auto", action="store_true", help="run daily job headless and exit")
    ap.add_argument("--output", choices=["mp4", "capcut"], default=None,
                    help="output mode (default: last used)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.auto:
        raise SystemExit(_run_auto(args.output))
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port)
