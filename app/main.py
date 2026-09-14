"""ClipForge web app (spec §9).

FastAPI + Jinja2. Routes: onboarding, dashboard (live job log), clips gallery,
settings. A single background job runner (one job at a time) drives the daily
pipeline and logs into an in-memory ring buffer polled by the UI every 2 s.
Crash-safety comes from the DB, not from here (see job.py / db.py).
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config as cfg
from . import db, discovery, job as jobmod, prompts, selector

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
    saved = {t["key"]: [dict(r) for r in db.channels_by_topic(t["key"])] for t in topics}
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
    """Save picked channels. Enforces exact per-topic pick counts (spec §4/§11.2)."""
    payload = await request.json()
    selected: dict[str, list[dict]] = payload.get("selected", {}) or {}
    required = {t["key"]: t["pick"] for t in _topics_meta()}
    errors = []
    for key, need in required.items():
        got = len(selected.get(key, []) or [])
        if got != need:
            errors.append(f"{key}: got {got}, need exactly {need}")
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    saved = 0
    for key, chans in selected.items():
        # wipe previous picks for this topic so counts stay exact on re-save
        db.execute("DELETE FROM videos WHERE channel_id IN "
                   "(SELECT id FROM channels WHERE topic=?)", (key,))
        db.execute("DELETE FROM channels WHERE topic=?", (key,))
        for ch in chans:
            db.add_channel(
                str(ch.get("platform_channel_id")), str(ch.get("title") or "unknown"),
                key, ch.get("subtopic"), _to_int(ch.get("subs")),
                _to_int(ch.get("video_count")), _to_int(ch.get("joined_year")),
                ch.get("avatar_url"),
            )
            saved += 1
    db.set_int_state("rotation_index", 0)
    return JSONResponse({"ok": True, "saved": saved, "redirect": "/dashboard"})


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
    channels = [dict(r) for r in db.all_channels()]
    return TEMPLATES.TemplateResponse(request, "dashboard.html", base_ctx(
        request, channels=channels, topics=_topics_meta(),
        default_count=selector.windows_for_duration(600),
        models={
            "text": c.get("openrouter.models.text"),
            "vision": c.get("openrouter.models.vision"),
        },
        has_key=bool(c.openrouter_api_key),
        active="dashboard"))


@app.post("/run")
def run_daily():
    if not runner.start("daily", jobmod.daily_job):
        raise HTTPException(status_code=409, detail="A job is already running")
    return JSONResponse({"started": True})


@app.post("/run/custom")
async def run_custom(request: Request):
    body = await request.json()
    url = (body.get("url") or "").strip() or None
    count = _to_int(body.get("count"))
    prompt = (body.get("prompt") or "").strip() or None
    # an unchanged/blank prompt means "use the persisted default"
    if prompt and prompt.strip() == prompts.DEFAULT_PROMPT.strip():
        prompt = None
    fn = (lambda log: jobmod.custom_job(url, log, clip_count=count,
                                        prompt_override=prompt))
    if not runner.start("custom", fn):
        raise HTTPException(status_code=409, detail="A job is already running")
    return JSONResponse({"started": True})


@app.get("/run/status")
def run_status():
    return JSONResponse(runner.status())


@app.get("/clips", response_class=HTMLResponse)
def clips(request: Request):
    rows = [dict(r) for r in db.all_clips()]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        day = (r.get("created_at") or "")[:10]
        groups.setdefault(day, []).append(r)
    return TEMPLATES.TemplateResponse(request, "clips.html", base_ctx(
        request, groups=groups, total=len(rows), active="clips"))


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
        has_key=bool(c.openrouter_api_key),
        test_result=None,
        active="settings"))


@app.post("/settings")
def settings_save(
    request: Request,
    text_model: str = Form(...), vision_model: str = Form(...),
    whisper_model: str = Form(...), daily_quota: str = Form("4"),
    clip_length: str = Form("60"), face_tracking: str = Form("auto"),
    default_prompt: str = Form(""),
):
    _persist_settings(text_model, vision_model, whisper_model, daily_quota,
                      clip_length, face_tracking, default_prompt)
    return TEMPLATES.TemplateResponse(request, "settings.html", base_ctx(
        request, cfg_models={"text": text_model, "vision": vision_model},
        whisper_model=whisper_model, face_tracking=face_tracking,
        clip_length=clip_length,
        has_key=bool(cfg.get_config().openrouter_api_key),
        test_result="Settings saved.", active="settings"))


@app.post("/settings/test_key")
def settings_test_key():
    from .editor_ai import test_key
    ok, msg = test_key()
    return JSONResponse({"ok": ok, "message": msg})


def _persist_settings(text_model, vision_model, whisper_model, daily_quota,
                      clip_length, face_tracking, default_prompt) -> None:
    """Write model/quota/clip/prompt overrides into the DB state layer (spec §11.8).

    config.yaml stays the source of defaults; runtime overrides win at read time
    via cfg.get_config().raw overlays stored in state.
    """
    c = cfg.get_config()
    c.raw.setdefault("openrouter", {}).setdefault("models", {})
    c.raw["openrouter"]["models"]["text"] = text_model
    c.raw["openrouter"]["models"]["vision"] = vision_model
    c.raw.setdefault("whisper", {})["model"] = whisper_model
    c.raw.setdefault("job", {})["daily_quota"] = int(float(daily_quota or 4))
    c.raw.setdefault("clip", {})["length_s"] = int(float(clip_length or 60))
    c.raw["clip"]["face_tracking"] = "off" if face_tracking == "off" else "auto"
    # persist to state so overrides survive a restart
    db.set_state("ov.openrouter.models.text", text_model)
    db.set_state("ov.openrouter.models.vision", vision_model)
    db.set_state("ov.whisper.model", whisper_model)
    db.set_state("ov.job.daily_quota", str(int(float(daily_quota or 4))))
    db.set_state("ov.clip.length_s", str(int(float(clip_length or 60))))
    db.set_state("ov.clip.face_tracking", "off" if face_tracking == "off" else "auto")
    if default_prompt.strip() and default_prompt.strip() != prompts.DEFAULT_PROMPT:
        db.set_state("default_prompt", default_prompt.strip())
    else:
        db.set_state("default_prompt", "")


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
    }
    for key, path in mapping.items():
        val = db.get_state(key)
        if val is None:
            continue
        node = c.raw
        for p in path[:-1]:
            node = node.setdefault(p, {})
        if path[-1] in ("daily_quota", "length_s"):
            try:
                node[path[-1]] = int(float(val))
            except (TypeError, ValueError):
                pass
        else:
            node[path[-1]] = val


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


# --- CLI entry: `python -m app.main --auto` (spec §10) ----------------------
def _run_auto() -> int:
    db.init_db()
    db.reset_run_state()
    produced: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)

    produced = jobmod.daily_job(log)
    print("\nProduced clips:")
    for p in produced:
        print(" ", p)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="clipforge")
    ap.add_argument("--auto", action="store_true", help="run daily job headless and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.auto:
        raise SystemExit(_run_auto())
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port)
