"""Bulk search + curation for one query (e.g. "steve jobs interview").

Uses the existing discovery pipeline (yt-dlp ``ytsearch``) to load up to
``limit`` results, then guarantees a curated list:

  * only videos LONGER than ``min_hours`` (default 1 hour) qualify
  * no two videos with matching OR similar titles (normalized
    SequenceMatcher ratio >= ``threshold`` keeps the higher-viewed one)
  * no two videos with the same length (same duration kept once, higher
    views preferred)

Results are saved to a JSON file (rich, reloadable) plus a plain ``.txt`` of
URLs (one per line) so the list can be used again via :func:`load`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import discovery

DEFAULT_QUERY = "steve jobs interview"
DEFAULT_LIMIT = 200
DEFAULT_MIN_HOURS = 1
DEFAULT_THRESHOLD = 0.85
DEFAULT_OUT = "data/bulk_search.json"


# --- normalization + similarity ---------------------------------------------
def norm_title(title: str) -> str:
    """Lowercase, collapse punctuation/spacing -> comparable form."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def similarity(a: str, b: str) -> float:
    """0..1 ratio between two normalized titles (1 = identical)."""
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


# --- curation steps (pure, testable) ---------------------------------------
def filter_duration(entries: list[dict], min_s: float) -> tuple[list[dict], int]:
    """Keep only entries with duration > min_s. Returns (kept, dropped_count)."""
    kept, dropped = [], 0
    for e in entries:
        dur = e.get("duration")
        if dur is None:            # unknown length -> cannot guarantee the rule
            dropped += 1
            continue
        if float(dur) > min_s:
            kept.append(e)
        else:
            dropped += 1
    return kept, dropped


def dedupe_titles(entries: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Drop matching/similar titles; keep the higher-viewed of each cluster."""
    ranked = sorted(entries, key=lambda e: int(e.get("view_count") or 0), reverse=True)
    kept: list[dict] = []
    for e in ranked:
        if any(similarity(e.get("title", ""), k.get("title", "")) >= threshold for k in kept):
            continue
        kept.append(e)
    return kept


def dedupe_durations(entries: list[dict]) -> list[dict]:
    """One video per exact duration; the higher-viewed one wins."""
    by_dur: dict[int, dict] = {}
    for e in sorted(entries, key=lambda x: int(x.get("view_count") or 0), reverse=True):
        dur = int(float(e["duration"]))
        if dur not in by_dur:
            by_dur[dur] = e
    return sorted(by_dur.values(), key=lambda e: int(e.get("view_count") or 0), reverse=True)


def curate(entries: list[dict], limit: int = DEFAULT_LIMIT, min_s: float = 3600.0,
           threshold: float = DEFAULT_THRESHOLD, query: str | None = None,
           require_terms: bool = False) -> list[dict]:
    """Full guarantee: duration filter -> title dedupe -> duration dedupe -> cap.

    With ``require_terms`` (and a query), entries whose title AND channel
    mention none of the query's words are dropped as off-topic noise.
    """
    kept = entries
    if require_terms and query:
        terms = [w for w in norm_title(query).split() if w]
        kept = [e for e in kept if _mentions(e, terms)]
    kept, _ = filter_duration(kept, min_s)
    kept = dedupe_titles(kept, threshold)
    kept = dedupe_durations(kept)
    return kept[:limit]


def _mentions(entry: dict, terms: list[str]) -> bool:
    hay = norm_title(f"{entry.get('title', '')} {entry.get('channel', '')}")
    return any(t in hay for t in terms)


# --- persistence ------------------------------------------------------------
def save(entries: list[dict], query: str, out: str | Path = DEFAULT_OUT) -> dict[str, Path]:
    """Write JSON (reloadable) + TXT (one URL per line). Returns both paths."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(entries),
        "rules": {"min_duration_s": 3600, "unique_titles": True,
                  "unique_durations": True},
        "videos": [
            {"url": e.get("url"), "video_id": e.get("id"), "title": e.get("title"),
             "duration_s": e.get("duration"), "views": e.get("view_count"),
             "channel": e.get("channel")}
            for e in entries
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(e.get("url") or "" for e in entries) + "\n", encoding="utf-8")
    return {"json": out, "txt": txt}


def load(path: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    """Reload a previously saved search file."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# --- entry point ------------------------------------------------------------
def run(query: str = DEFAULT_QUERY, limit: int = DEFAULT_LIMIT,
        min_hours: float = DEFAULT_MIN_HOURS, out: str | Path = DEFAULT_OUT,
        require_terms: bool = True, log: Callable = print) -> list[dict]:
    log(f"[bulk] searching ytsearch{limit}:{query!r}")
    # large result pages take longer than the per-topic discovery budget
    entries = discovery._search_query(query, limit=limit, timeout=180)
    log(f"[bulk] {len(entries)} raw results")
    kept = curate(entries, limit=limit, min_s=min_hours * 3600,
                  query=query, require_terms=require_terms)
    log(f"[bulk] {len(kept)} curated (>{min_hours}h, unique titles, unique lengths)")
    paths = save(kept, query, out)
    log(f"[bulk] saved {paths['json']} + {paths['txt']}")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="bulk_search")
    ap.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-require-terms", action="store_true",
                    help="keep off-topic results (default: title/channel must mention the query)")
    args = ap.parse_args()
    rows = run(args.query, args.limit, args.min_hours, args.out,
               require_terms=not args.no_require_terms)
    for r in rows[:20]:
        m = int((r.get("duration") or 0) // 60)
        views = int(r.get("view_count") or r.get("views") or 0)
        print(f"  {m:>4} min  {views:>9,}  {r.get('title')}")
    if len(rows) > 20:
        print(f"  … +{len(rows) - 20} more")
