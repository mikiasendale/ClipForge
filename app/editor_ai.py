"""Editor AI via OpenRouter (spec §7).

* text path   -> title/channel/duration + transcript with [mm:ss] markers
* vision path -> per-candidate-window 5 keyframes (512w base64) + timestamps

Robust JSON parsing (strip fences, extract first ``{...}``), validation (exactly
N clips, <= max_len, non-overlapping, within bounds and >=2s from ends), one
retry, then a *fail-soft* fall back to the analyzer's top heuristic windows.
Every call is logged (model, latency, token usage) to ``data/logs/``.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from . import config as cfg
from . import prompts
from .analyzer import Analysis

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Pick:
    start_s: float
    end_s: float
    hook_title: str
    caption: str
    reason: str
    source: str  # 'ai' | 'heuristic'


@dataclass
class EditorResult:
    clips: list[Pick]
    model: str
    used_fallback: bool
    error: str | None = None


def _log(record: dict[str, Any]) -> None:
    try:
        cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (cfg.LOGS_DIR / "openrouter.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **record}) + "\n")
    except Exception:
        pass


# --- OpenRouter transport (monkeypatched in tests) --------------------------
def call_openrouter(messages: list[dict], model: str) -> tuple[str, dict, float]:
    """Return (assistant_text, usage, latency_ms). Raises on hard failure."""
    data, latency = _post_chat(messages, model)
    usage = data.get("usage", {}) or {}
    text = data["choices"][0]["message"]["content"]
    return text, usage, latency


def call_openrouter_tools(messages: list[dict], tools: list[dict],
                          model: str) -> tuple[dict, dict, float]:
    """Tool-calling variant: return (assistant_message_dict, usage, latency_ms).

    assistant_message_dict is the raw choice.message, i.e. {role, content,
    tool_calls?}. Raises on any transport/HTTP error.
    """
    data, latency = _post_chat(messages, model, tools=tools, tool_choice="auto")
    usage = data.get("usage", {}) or {}
    message = data["choices"][0]["message"]
    return message, usage, latency


def _post_chat(messages: list[dict], model: str,
               tools: list[dict] | None = None,
               tool_choice: str | None = None) -> tuple[dict, float]:
    c = cfg.get_config()
    key = c.openrouter_api_key
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    base = c.get("openrouter.base_url", "https://openrouter.ai/api/v1")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(c.get("openrouter.temperature", 0.3)),
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:8000",
        "X-Title": "ClipForge",
    }
    t0 = time.time()
    resp = requests.post(f"{base}/chat/completions", json=body, headers=headers,
                         timeout=float(c.get("openrouter.timeout_s", 120)))
    latency = (time.time() - t0) * 1000.0
    resp.raise_for_status()
    return resp.json(), latency


# --- JSON extraction + validation ------------------------------------------
def extract_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    # prefer an object that contains "clips"
    m = re.search(r'\{[^{}]*"clips".*\}', cleaned, re.DOTALL)
    candidate = m.group(0) if m else None
    if candidate is None:
        mo = _JSON_OBJ_RE.search(cleaned)
        candidate = mo.group(0) if mo else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # last resort: try the whole cleaned blob
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _valid_clip(d: dict, max_len: float) -> bool:
    try:
        s = float(d["start_s"]); e = float(d["end_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return s >= 0 and e > s and (e - s) <= max_len + 0.5


def validate_clips(clips: list[dict], n: int, duration: float,
                   max_len: float, edge_pad: float = 2.0) -> list[dict]:
    """Keep in-bounds, non-overlapping, <=max_len clips; cap to n; enforce count."""
    good: list[dict] = []
    for d in clips:
        if not _valid_clip(d, max_len):
            continue
        s = float(d["start_s"]); e = float(d["end_s"])
        if s < edge_pad or e > duration - edge_pad:
            continue
        if any(not (e <= gs or s >= ge) for gs, ge in
               ((float(g["start_s"]), float(g["end_s"])) for g in good)):
            continue
        good.append(d)
        if len(good) >= n:
            break
    return good


# --- keyframe extraction for vision path ------------------------------------
def _extract_keyframes(video: Path, start: float, end: float, count: int,
                       ffmpeg: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    span = max(0.1, end - start)
    for i in range(count):
        t = start + span * (i + 0.5) / count
        fp = out_dir / f"frame_{int(start)}_{i}.jpg"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-ss", f"{t:.2f}", "-i", str(video),
                 "-frames:v", "1", "-vf", "scale=512:-1", str(fp)],
                capture_output=True, text=True, timeout=60,
            )
            if fp.is_file():
                frames.append(fp)
        except Exception:
            continue
    return frames


def _img_data_url(fp: Path) -> str:
    b = fp.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")


# --- message builders -------------------------------------------------------
def build_text_messages(analysis: Analysis, title: str, channel: str, n: int,
                        prompt: str) -> list[dict]:
    header = prompts.TEXT_HEADER.format(
        title=title, channel=channel, duration_s=round(analysis.duration_s),
        transcript=analysis.transcript_text(),
    )
    task = prompts.format_prompt(prompt, n)
    return [{"role": "user", "content": header + task}]


def build_vision_messages(analysis: Analysis, title: str, channel: str, n: int,
                          prompt: str, video: Path, ffmpeg: str, out_dir: Path) -> list[dict]:
    header = prompts.VISION_HEADER.format(
        title=title, channel=channel, duration_s=round(analysis.duration_s))
    task = prompts.format_prompt(prompt, n)
    content: list[dict] = [{"type": "text", "text": header}]
    for widx, win in enumerate(analysis.windows):
        stamp = f"[candidate {widx + 1}: {win.start:.0f}s-{win.end:.0f}s]"
        content.append({"type": "text", "text": stamp})
        for fp in _extract_keyframes(video, win.start, win.end, 5, ffmpeg,
                                     out_dir / f"cand{widx}"):
            content.append({"type": "image_url", "image_url": {"url": _img_data_url(fp)}})
    content.append({"type": "text", "text": "TASK\n" + task})
    return [{"role": "user", "content": content}]


# --- heuristic fallback -----------------------------------------------------
def _heuristic_picks(analysis: Analysis, n: int, max_len: float,
                     duration: float) -> list[Pick]:
    picks: list[Pick] = []
    for w in analysis.windows:
        s, e = w.start, w.end
        if s < 2.0:
            s = 2.0
        if e > duration - 2.0:
            e = duration - 2.0
        if e - s > max_len:
            e = s + max_len
        if e - s <= 1.0:
            continue
        picks.append(Pick(round(s, 2), round(e, 2),
                          "You won't believe this", "Wait for it",
                          w.reason or "heuristic", "heuristic"))
        if len(picks) >= n:
            break
    return picks


# --- main entry -------------------------------------------------------------
def pick_clips(video: Path, analysis: Analysis, title: str, channel: str,
               topic: str, n: int, prompt_override: str | None = None) -> EditorResult:
    c = cfg.get_config()
    max_len = float(c.get("clip.length_s", 60))
    prompt = prompt_override or _saved_prompt(c)
    model_text = str(c.get("openrouter.models.text", "google/gemini-2.5-flash"))
    model_vision = str(c.get("openrouter.models.vision", "google/gemini-2.5-flash"))

    if analysis.mode == "speech" and analysis.has_transcript:
        messages = build_text_messages(analysis, title, channel, n, prompt)
        model = model_text
    else:
        out_dir = cfg.FRAMES_DIR / datetime.now().strftime("%Y%m%d%H%M%S")
        messages = build_vision_messages(analysis, title, channel, n, prompt,
                                         video, c.ffmpeg or "ffmpeg", out_dir)
        model = model_vision

    clips: list[dict] = []
    last_err: str | None = None
    try:
        for attempt in range(2):  # initial + 1 retry (spec §7)
            msg = messages if attempt == 0 else _with_retry(messages)
            text, usage, latency = call_openrouter(msg, model)
            _log({"model": model, "path": analysis.mode, "attempt": attempt,
                  "latency_ms": round(latency, 1), "usage": usage})
            parsed = extract_json(text)
            raw = (parsed or {}).get("clips", []) if isinstance(parsed, dict) else []
            clips = validate_clips(raw, n, analysis.duration_s, max_len)
            if len(clips) == n:
                break
        used_fallback = len(clips) != n
        if not used_fallback:
            picks = [Pick(float(d["start_s"]), float(d["end_s"]),
                          str(d.get("hook_title", ""))[:40],
                          str(d.get("caption", ""))[:150],
                          str(d.get("reason", "")), "ai") for d in clips]
            return EditorResult(picks, model, False)
    except Exception as e:  # network / auth / schema failure -> fail-soft
        last_err = str(e)
        _log({"model": model, "path": analysis.mode, "error": last_err})

    picks = _heuristic_picks(analysis, n, max_len, analysis.duration_s)
    return EditorResult(picks, model, True, last_err)


def _with_retry(messages: list[dict]) -> list[dict]:
    out = [dict(m) for m in messages]
    last = out[-1]
    if isinstance(last.get("content"), str):
        last["content"] = last["content"] + prompts.RETRY_SUFFIX
    elif isinstance(last.get("content"), list):
        last["content"] = last["content"] + [
            {"type": "text", "text": prompts.RETRY_SUFFIX}]
    return out


def _saved_prompt(c: cfg.Config) -> str:
    from . import db
    return db.get_state("default_prompt", None) or prompts.DEFAULT_PROMPT


def test_key() -> tuple[bool, str]:
    """Used by the settings 'test key' button (spec §9)."""
    c = cfg.get_config()
    model = str(c.get("openrouter.models.text", "google/gemini-2.5-flash"))
    try:
        text, _u, _l = call_openrouter(
            [{"role": "user", "content": "Reply with the single word: OK"}], model)
        return (True, text.strip()[:120])
    except Exception as e:
        return (False, str(e))
