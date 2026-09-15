"""Notification fan-out (feature P2, spec §3.5).

``notify()`` is the single entry point used by health checks + suggestions +
jobs. It: (1) skips if the same ``dedupe_key`` fired within its window,
(2) inserts the row, (3) broadcasts an SSE ``notification_new``, and
(4) optionally posts to a webhook / Telegram. All external sends are best-effort
and never raise to the caller — a broken webhook must not break a job.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from . import config as cfg, db, events

# severity -> dedupe window (hours) when no explicit window given
_DEFAULT_WINDOW = {"info": 24.0, "warn": 24.0, "error": 24.0, "suggestion": 24 * 7}


def notify(kind: str, severity: str, title: str, body: str = "",
           actions: list[dict] | None = None, dedupe_key: str | None = None,
           within_hours: float | None = None, db_path: Any = None) -> int | None:
    """Create + broadcast a notification. Returns row id, or None if deduped."""
    actions = actions or []
    actions_json = json.dumps(actions) if actions else None
    window = within_hours if within_hours is not None else _DEFAULT_WINDOW.get(severity, 24.0)

    if dedupe_key and db.dedupe_seen(dedupe_key, window, db_path):
        return None  # already surfaced within its window -> stay quiet

    nid = db.insert_notification(kind, severity, title, body, actions_json,
                                 dedupe_key, db_path)
    if nid is None:
        return None
    events.notification_new(nid, severity, title, kind)
    _dispatch_external(severity, title, body, kind)
    return nid


def _dispatch_external(severity: str, title: str, body: str, kind: str) -> None:
    c = cfg.get_config()
    n = c.get("notifications", {}) or {}
    payload = {"kind": kind, "severity": severity, "title": title, "body": body,
               "source": "clipforge"}
    url = (n.get("webhook_url") or "").strip()
    if url:
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:  # noqa: BLE001 - best effort only
            pass
    token = (n.get("telegram_bot_token") or "").strip()
    chat = (n.get("telegram_chat_id") or "").strip()
    if token and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": f"[{severity}] {title}\n{body}"},
                          timeout=10)
        except Exception:  # noqa: BLE001
            pass


def test_message() -> tuple[bool, str]:
    """[Test] buttons in settings: fire a visible test notification."""
    nid = notify("test", "info", "ClipForge test notification",
                 "If you can see this in the bell, notifications are working.",
                 dedupe_key=None)
    return (nid is not None, f"notification id {nid}")
