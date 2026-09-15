"""In-process pub/sub bus backing the SSE endpoint (feature P1, spec §3.4).

Deliberately thread-safe: the daily job and pre-staging run in worker threads
but publish events to browser SSE streams living on the asyncio loop. Each
subscriber owns a ``SimpleQueue``; the SSE generator drains it via
``run_in_executor`` so publishers never touch the event loop directly (no loop
capture, no cross-thread async hazards).
"""
from __future__ import annotations

import asyncio
import json
import threading
import queue
from typing import Any, Iterator


class _Subscriber:
    __slots__ = ("q",)

    def __init__(self) -> None:
        self.q: queue.SimpleQueue = queue.SimpleQueue()


class EventBus:
    def __init__(self) -> None:
        self._subs: set[_Subscriber] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> _Subscriber:
        sub = _Subscriber()
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        """Fire-and-forget. Safe to call from any thread."""
        msg = {"event": event, "data": data}
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            try:
                s.q.put(msg)
            except Exception:  # pragma: no cover
                pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


bus = EventBus()


def sse_format(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_comment(text: str = "ping") -> str:
    return f": {text}\n\n"


async def stream(sub: _Subscriber, heartbeat_s: float = 15.0) -> Iterator[str]:
    """Async generator yielding SSE chunks; heartbeats when idle.

    Blocks on the subscriber queue in the default threadpool (a local app has
    only a handful of tabs), and emits an SSE comment every ``heartbeat_s`` to
    keep proxies from closing the connection.
    """
    loop = asyncio.get_running_loop()
    try:
        while True:
            fut = loop.run_in_executor(None, sub.q.get)
            try:
                msg: dict = await asyncio.wait_for(asyncio.shield(fut), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                yield sse_comment()
                continue
            yield sse_format(msg["event"], msg["data"])
    finally:
        bus.unsubscribe(sub)


# --- convenience publishers -------------------------------------------------
def job_progress(stage: str, detail: str = "", pct: float = 0.0) -> None:
    bus.publish("job_progress", {"stage": stage, "detail": detail, "pct": round(pct, 3)})


def notification_new(notif_id: int, severity: str, title: str, kind: str) -> None:
    bus.publish("notification_new", {"id": notif_id, "severity": severity,
                                     "title": title, "kind": kind})


def clip_rendered(clip: dict[str, Any]) -> None:
    bus.publish("clip_rendered", clip)


def schedule_changed(**data: Any) -> None:
    bus.publish("schedule_changed", data)
