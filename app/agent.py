"""Tool-calling editor agent (feature A).

Replaces the fixed analyzer→editor handoff with a loop where the LLM calls
local analysis tools (transcript, scene list, audio energy, optional keyframes)
and finishes by calling the terminal ``propose_clips`` tool. Works with
text-only OpenRouter models; vision is opt-in.

Grounding: the model must call tools rather than invent timestamps.
Crash-safety/determinism are untouched — this only *selects* windows; the caller
(job.py) still renders and only then sets ``used_at``.

Fallback ladder (spec A1):
  1. API error / no tool call -> retry the pass once.
  2. loop exhausts max_steps without a clean terminal -> reuse the last
     ``propose_clips`` attempt if it yields >=1 valid window (engine
     "agent_last_attempt").
  3. still nothing -> run the OLD single-shot path (analyzer candidates +
     editor_ai) and return those with engine "fallback_single_shot".

Every iteration is logged to ``data/logs/agent_{video_id}.jsonl``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from . import analyzer, editor_ai, prompts

LogFn = Callable[[str], None]

_EDGE_PAD = 2.0
_CHUNK_CHARS = 4000
_TS_RE = re.compile(r"\[(\d+):(\d+)\]")

# --- tool schemas (OpenAI-compatible function tools) ------------------------
TOOLS_TEXT = [
    {"type": "function", "function": {
        "name": "get_transcript",
        "description": "Transcript as [mm:ss] word text, chunked. Call with "
                       "chunk_index=0 first; it reports total_chunks.",
        "parameters": {"type": "object", "properties": {
            "chunk_index": {"type": "integer", "minimum": 0}},
            "required": ["chunk_index"]}}},
    {"type": "function", "function": {
        "name": "get_scene_list",
        "description": "PySceneDetect scene boundaries as a [mm:ss] list.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_audio_energy",
        "description": "Per-0.5s normalized RMS (0-1) in a range as 'mm:ss=0.42' "
                       "lines. Ranges over 600s are rejected.",
        "parameters": {"type": "object", "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"}},
            "required": ["start_s", "end_s"]}}},
    {"type": "function", "function": {
        "name": "propose_clips",
        "description": "TERMINAL. Call exactly once with your final clip windows.",
        "parameters": {"type": "object", "properties": {
            "clips": {"type": "array", "items": {"type": "object", "properties": {
                "start_s": {"type": "number"}, "end_s": {"type": "number"},
                "hook_title": {"type": "string"}, "caption": {"type": "string"},
                "reason": {"type": "string"}},
                "required": ["start_s", "end_s"]}}},
            "required": ["clips"]}}},
]

TOOLS_VISION = [{
    "type": "function", "function": {
        "name": "get_keyframes",
        "description": "Up to 5 base64 JPEG keyframes (512w) with timestamps over a "
                       "range, for visual scoring.",
        "parameters": {"type": "object", "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "n": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["start_s", "end_s"]}}}]


def _now_ts(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _parse_ts(s: str) -> float:
    m = _TS_RE.match(s) or re.match(r"(\d+):(\d+)", s)
    if not m:
        return float(s)
    return int(m.group(1)) * 60 + int(m.group(2))


def _log_line(video_id: str, record: dict) -> None:
    try:
        cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
        with (cfg.LOGS_DIR / f"agent_{video_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# --- material context -------------------------------------------------------
class Material:
    def __init__(self, video_row: dict, source: Path, topic: str, duration_s: float):
        self.video_id = video_row["video_id"]
        self.source = Path(source)
        self.topic = topic
        self.duration_s = float(duration_s or 0) or analyzer.probe_duration(
            self.source, cfg.get_config().ffprobe) or 600.0
        self._words: list[analyzer.Word] | None = None

    @property
    def words(self) -> list[analyzer.Word]:
        if self._words is None:
            self._words = analyzer.ensure_words(self.video_id, self.source)
        return self._words


# --- tool implementations ---------------------------------------------------
def _tool_transcript(mat: Material, args: dict) -> dict:
    words = mat.words
    full = analyzer.Analysis("speech", mat.duration_s, words).transcript_text()
    chunks = _chunk(full, _CHUNK_CHARS)
    idx = int(args.get("chunk_index", 0) or 0)
    idx = max(0, min(idx, len(chunks) - 1)) if chunks else 0
    return {"total_chunks": len(chunks), "chunk_index": idx,
            "text": chunks[idx] if chunks else "(no transcript)"}


def _chunk(text: str, size: int) -> list[str]:
    if not text:
        return []
    out, cur, n = [], "", 0
    for word in text.split(" "):
        if n + len(word) + 1 > size and cur:
            out.append(cur)
            cur, n = word, len(word)
        else:
            cur = (cur + " " + word).strip()
            n = len(cur)
    if cur:
        out.append(cur)
    return out


def _tool_scenes(mat: Material, args: dict) -> dict:
    scenes = analyzer.ensure_scenes(mat.video_id, mat.source)
    return {"scene_count": len(scenes), "scenes": [_now_ts(s) for s in scenes[:200]]}


def _tool_energy(mat: Material, args: dict) -> dict:
    start = float(args.get("start_s", 0) or 0)
    end = float(args.get("end_s", 0) or 0)
    if end - start > 600:
        return {"error": "range too large: max 600s"}
    step = float(cfg.get_config().get("analyzer.visual_peak_step_s", 0.5))
    times, rms, _ = analyzer.ensure_energy(mat.video_id, mat.source, step)
    lines = [f"{_now_ts(t)}={r:.2f}" for t, r in zip(times, rms) if start <= t <= end]
    return {"samples": len(lines), "energy": "\n".join(lines[:1200])}


def _tool_keyframes(mat: Material, args: dict) -> dict:
    start = float(args.get("start_s", 0) or 0)
    end = min(float(args.get("end_s", 0) or 0), mat.duration_s)
    n = max(1, min(5, int(args.get("n", 5) or 5)))
    ffmpeg = cfg.get_config().ffmpeg or "ffmpeg"
    out_dir = cfg.FRAMES_DIR / "agent" / mat.video_id
    frames = editor_ai._extract_keyframes(mat.source, start, end, n, ffmpeg, out_dir)
    return {"window": f"{_now_ts(start)}-{_now_ts(end)}",
            "images": [{"t": _now_ts(start + (end - start) * (i + 0.5) / n),
                        "b64": editor_ai._img_data_url(fp)} for i, fp in enumerate(frames)]}


# --- propose_clips validation ----------------------------------------------
def validate_proposed(clips: list[dict], n_clips: int, duration: float,
                      max_len: float = 60.0, edge: float = _EDGE_PAD) -> tuple[list[dict], list[str]]:
    """Return (valid_clips, errors). valid_clips sorted, normalized, deduped."""
    errors: list[str] = []
    out: list[dict] = []
    if not isinstance(clips, list):
        return [], ["clips must be an array"]
    for i, d in enumerate(clips):
        if not isinstance(d, dict) or "start_s" not in d or "end_s" not in d:
            errors.append(f"clip[{i}]: need start_s and end_s"); continue
        try:
            s = round(float(d["start_s"]), 2)
            e = round(float(d["end_s"]), 2)
        except (TypeError, ValueError):
            errors.append(f"clip[{i}]: non-numeric bounds"); continue
        if e <= s:
            errors.append(f"clip[{i}]: end must be > start"); continue
        if e - s > max_len + 0.5:
            errors.append(f"clip[{i}]: {e - s:.1f}s > {max_len:.0f}s max"); continue
        if s < edge:
            errors.append(f"clip[{i}]: starts {s:.1f}s, must be >= {edge}s from start"); continue
        if e > duration - edge:
            errors.append(f"clip[{i}]: ends {e:.1f}s, must be <= {duration - edge:.1f}s"); continue
        if any(not (e <= os_ or s >= oe) for os_, oe in
               ((float(g["start_s"]), float(g["end_s"])) for g in out)):
            errors.append(f"clip[{i}]: overlaps an earlier clip"); continue
        out.append({"start_s": s, "end_s": e,
                    "hook_title": str(d.get("hook_title", ""))[:40],
                    "caption": str(d.get("caption", ""))[:150],
                    "reason": str(d.get("reason", ""))})
    if not out:
        if not errors:
            errors.append("no clips provided")
        return [], errors
    if len(out) > n_clips:
        errors.append(f"{len(out)} clips given, expected {n_clips}; taking the first {n_clips}")
        out = out[:n_clips]
    elif len(out) < 1:
        errors.append(f"expected exactly {n_clips}, got 0")
    out.sort(key=lambda c: c["start_s"])
    return out, errors


# --- the loop ---------------------------------------------------------------
def _system_prompt(n_clips: int, override: str | None) -> str:
    template = override or _saved_default_prompt()
    base = prompts.format_prompt(template, n_clips)
    return (
        base
        + f"\n\nYou are the editor. The video is {n_clips} clip(s) to select. Use the "
          "tools to inspect the material, then call propose_clips exactly once when "
          "you have decided. Do not guess timestamps — ground them in tool output."
    )


def _saved_default_prompt() -> str:
    from . import db
    return db.get_state("default_prompt", None) or prompts.DEFAULT_PROMPT


def _clean_assistant(message: dict) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def _run_loop(mat: Material, n_clips: int, prompt: str, log: LogFn) -> tuple[list[dict] | None, str, list]:
    c = cfg.get_config()
    model = str(c.get("agent.model", "google/gemini-2.5-flash"))
    max_steps = int(c.get("agent.max_steps", 8))
    vision = bool(c.get("agent.vision_enabled", False))
    tools = TOOLS_TEXT + (TOOLS_VISION if vision else [])
    messages: list[dict] = [{"role": "system", "content": prompt}]

    last_propose: list[dict] = []
    tool_names: set[str] = set()

    for step in range(max_steps):
        message, usage, latency = editor_ai.call_openrouter_tools(messages, tools, model)
        tool_calls = message.get("tool_calls") or []
        _log_line(mat.video_id, {"step": step, "model": model, "latency_ms": round(latency, 1),
                                 "usage": usage, "tool_calls": [tc["function"]["name"] for tc in tool_calls]})
        messages.append(_clean_assistant(message))
        if not tool_calls:
            messages.append({"role": "user", "content":
                             "Call a tool to inspect the material, or call propose_clips to finish."})
            log(f"      agent step {step}: no tool call, nudged")
            continue
        terminal_ok: list[dict] | None = None
        for tc in tool_calls:
            name = tc["function"]["name"]
            tool_names.add(name)
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch(name, args, mat, n_clips)
            _log_line(mat.video_id, {"step": step, "tool": name, "args": args, "result": _short(result)})
            if name == "propose_clips":
                ok, errs = result["__valid__"], result["__errors__"]
                last_propose = result["__raw__"]
                if ok:
                    terminal_ok = ok
                    result_payload = {"accepted": True, "clips": ok}
                else:
                    result_payload = {"accepted": False, "errors": errs}
            else:
                result_payload = result
            messages.append({"role": "tool", "tool_call_id": tc.get("id", name),
                             "name": name, "content": json.dumps(result_payload, default=str)})
        if terminal_ok is not None:
            return terminal_ok, "agent", sorted(tool_names)
    # loop exhausted
    valid, _ = validate_proposed(last_propose, n_clips, mat.duration_s)
    if valid:
        return valid, "agent_last_attempt", sorted(tool_names)
    return None, "none", sorted(tool_names)


def _dispatch(name: str, args: dict, mat: Material, n_clips: int) -> dict:
    if name == "get_transcript":
        return _tool_transcript(mat, args)
    if name == "get_scene_list":
        return _tool_scenes(mat, args)
    if name == "get_audio_energy":
        return _tool_energy(mat, args)
    if name == "get_keyframes":
        return _tool_keyframes(mat, args)
    if name == "propose_clips":
        raw = args.get("clips", []) if isinstance(args, dict) else []
        valid, errs = validate_proposed(raw, n_clips, mat.duration_s)
        return {"__raw__": raw, "__valid__": valid, "__errors__": errs}
    return {"error": f"unknown tool {name}"}


def _short(result: dict) -> dict:
    """Trim big tool outputs (base64/transcript) before writing to the jsonl log."""
    r = {k: v for k, v in result.items() if not k.startswith("__")}
    if "text" in r and isinstance(r["text"], str) and len(r["text"]) > 200:
        r["text"] = r["text"][:200] + f"... (+{len(result['text']) - 200} chars)"
    if "images" in r:
        r["images"] = f"{len(r['images'])} keyframe(s)"
    if "energy" in r and isinstance(r["energy"], str) and len(r["energy"]) > 200:
        r["energy"] = r["energy"][:200] + "...(truncated)"
    return r


# --- fallback: the OLD single-shot path -------------------------------------
def _fallback_single_shot(mat: Material, n_clips: int, override: str | None,
                          log: LogFn) -> list[dict]:
    mode = cfg.get_config().topic_mode(mat.topic)
    log("      agent -> running single-shot fallback")
    analysis = analyzer.analyze(mat.source, mat.topic, mode, video_id=mat.video_id)
    title = ""
    channel = ""
    result = editor_ai.pick_clips(mat.source, analysis, title, channel,
                                  mat.topic, n_clips, override)
    clips = [{"start_s": round(p.start_s, 2), "end_s": round(p.end_s, 2),
              "hook_title": p.hook_title, "caption": p.caption, "reason": p.reason,
              "engine": "fallback_single_shot"} for p in result.clips]
    return clips


# --- public entry -----------------------------------------------------------
def run_agent(video_row: dict, n_clips: int, user_prompt_override: str | None = None,
              *, source: str | Path | None = None, topic: str | None = None,
              log: LogFn = lambda *_a: None) -> list[dict]:
    """Drive the editor agent. Returns clip dicts with an ``engine`` key."""
    source = Path(source) if source else None
    if source is None:
        from . import downloader
        source = downloader.resolve_source(video_row["video_id"])
    topic = topic or video_row.get("topic") or "football"
    duration = video_row.get("duration_s") or 0.0
    mat = Material(video_row, source, topic, duration)
    prompt = _system_prompt(n_clips, user_prompt_override)

    attempts = 0
    while attempts < 2:  # pass + one retry on API error (spec A1 fallback 1)
        attempts += 1
        try:
            clips, engine, seen = _run_loop(mat, n_clips, prompt, log)
        except Exception as e:
            log(f"      agent API error (attempt {attempts}): {e}")
            _log_line(mat.video_id, {"error": str(e), "attempt": attempts})
            continue
        if clips is not None:
            log(f"      agent done via {engine} (tools used: {', '.join(seen) or 'none'})")
            for cl in clips:
                cl.setdefault("engine", engine)
            return clips

    # exhausted both passes without a usable terminal -> single-shot fallback
    _log_line(mat.video_id, {"engine": "fallback_single_shot", "reason": "no clean propose_clips"})
    return _fallback_single_shot(mat, n_clips, user_prompt_override, log)
