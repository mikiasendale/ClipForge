"""APScheduler wiring (feature P0, spec §3.1).

A single AsyncIOScheduler is started with the FastAPI app. It runs:
  * the daily job at ``schedule.time`` (cron), and
  * a pre-staging pass at ``schedule.time - prestaging.hours_before`` (cron).

Callbacks return instantly — they only hand work to ``main.runner`` (a guarded
worker thread) — so the event loop is never blocked. A module-level singleton +
``started`` flag guards against double-start under uvicorn --reload / repeated
app import. ``catch_up`` runs the job once at boot if today's scheduled time
already passed and no run happened today (``state.last_run_date`` prevents
doubles).
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from . import config as cfg, db, events

daily_fn: Callable[[], None] = lambda: None
prestage_fn: Callable[[], None] = lambda: None
_scheduled_job_id = "clipforge_daily"
_prestage_job_id = "clipforge_prestage"


def configure(daily: Callable[[], None], prestage: Callable[[], None]) -> None:
    global daily_fn, prestage_fn
    daily_fn, prestage_fn = daily, prestage


def parse_time(hhmm: str) -> tuple[int, int]:
    try:
        h, m = str(hhmm).strip().split(":")
        return max(0, min(23, int(h))), max(0, min(59, int(m)))
    except (ValueError, AttributeError):
        return 6, 0


class AppScheduler:
    def __init__(self) -> None:
        self.sched = None
        self.started = False

    # --- lifecycle -----------------------------------------------------
    def start(self) -> bool:
        """Start once. Returns True if newly started."""
        if self.started:
            return False
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
        except Exception:  # pragma: no cover
            return False
        self.sched = AsyncIOScheduler(timezone=None)  # local tz
        self.sched.start()
        self.started = True
        self.apply_schedule()
        self.catch_up()
        return True

    def shutdown(self) -> None:
        if self.sched is not None:
            try:
                self.sched.shutdown(wait=False)
            except Exception:  # pragma: no cover
                pass
        self.started = False
        self.sched = None

    # --- jobs ----------------------------------------------------------
    def apply_schedule(self) -> None:
        """(Re)register/remove jobs from current config + emit change event."""
        if self.sched is None:
            return
        for jid in (_scheduled_job_id, _prestage_job_id):
            if self.sched.get_job(jid):
                self.sched.remove_job(jid)
        c = cfg.get_config()
        if not c.get("schedule.enabled", True):
            events.schedule_changed(enabled=False, next_run=None)
            return
        hour, minute = parse_time(c.get("schedule.time", "06:00"))
        self.sched.add_job(self._fire_daily, "cron", hour=hour, minute=minute,
                           id=_scheduled_job_id, replace_existing=True, coalesce=True,
                           misfire_grace_time=3600)
        if c.get("prestaging.enabled", True):
            hb = int(c.get("prestaging.hours_before", 10) or 0)
            ph = (hour - (hb % 24)) % 24
            self.sched.add_job(self._fire_prestage, "cron", hour=ph, minute=minute,
                               id=_prestage_job_id, replace_existing=True, coalesce=True,
                               misfire_grace_time=3600)
        events.schedule_changed(enabled=True, time=f"{hour:02d}:{minute:02d}",
                                next_run=self.next_run_iso())

    def _fire_daily(self) -> None:
        try:
            daily_fn()
        except Exception:  # pragma: no cover
            pass

    def _fire_prestage(self) -> None:
        try:
            prestage_fn()
        except Exception:  # pragma: no cover
            pass

    # --- helpers for the briefing -------------------------------------
    def next_run_dt(self) -> Optional[datetime]:
        if self.sched is None:
            return None
        job = self.sched.get_job(_scheduled_job_id)
        return job.next_run_time if job else None

    def next_run_iso(self) -> Optional[str]:
        dt = self.next_run_dt()
        return dt.isoformat(timespec="seconds") if dt else None

    def next_run_countdown(self) -> Optional[str]:
        dt = self.next_run_dt()
        if not dt:
            return None
        delta = dt - datetime.now(dt.tzinfo) if dt.tzinfo else dt - datetime.now()
        total = int(delta.total_seconds())
        if total <= 0:
            return "now"
        h, m = total // 3600, (total % 3600) // 60
        return f"{h}h {m}m" if h else f"{m}m"

    # --- catch-up ------------------------------------------------------
    def catch_up(self) -> None:
        """Run once at boot if today's scheduled time passed with no run today."""
        c = cfg.get_config()
        if not (c.get("schedule.enabled", True) and c.get("schedule.catch_up", True)):
            return
        today = datetime.now().date().isoformat()
        if db.get_state("last_run_date", None) == today:
            return  # already ran today
        hour, minute = parse_time(c.get("schedule.time", "06:00"))
        sched_today = datetime.now().replace(hour=hour, minute=minute,
                                             second=0, microsecond=0)
        if datetime.now() >= sched_today:
            db.set_state("last_run_date", today)  # claim before starting to avoid races
            self._fire_daily()


scheduler = AppScheduler()
