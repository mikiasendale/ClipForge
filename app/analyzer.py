"""Highlight detection (spec §6).

Two paths, chosen by the topic's ``mode`` (speech | visual | auto):

speech/auto
    faster-whisper word timestamps. If the transcript has < 50 words and mode
    is ``auto``, fall through to the visual path. Otherwise rank sliding 60s
    windows (step 15s) by keyword density (exclamations, superlatives, numbers,
    topic words) -> top ``speech_top_candidates``.

visual
    PySceneDetect ContentDetector scenes + ffmpeg 16 kHz mono wav -> RMS energy
    per 0.5 s -> smoothed. peak = audio_energy + scene_change_density. Build
    ``visual_candidates`` 60 s windows spread >= ``visual_min_gap_s`` apart
    around the top peaks.

All heavy imports are lazy so the module loads even when a dependency is
missing (the pipeline then degrades gracefully).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg

SUPERLATIVES = {
    "best", "greatest", "amazing", "incredible", "unbelievable", "insane", "wild",
    "world-class", "magical", "legendary", "perfect", "brilliant", "phenomenal",
    "goalll", "knockout", "knocked", "clutch", "record", "historic", "momentum",
}
TOPIC_WORDS = {
    "football": {"goal", "shot", "pass", "keeper", "striker", "penalty", "header",
                 "counter", "dribble", "assist", "match", "score", "freekick", "cross"},
    "boxing": {"punch", "jab", "hook", "uppercut", "combo", "knockout", "ko",
               "round", "footwork", "guard", "counter", "chin", "southpaw", "clinch"},
    "cats_silent": {"cat", "kitten", "jump", "pounce", "chase", "play"},
    "cats_compilations": {"cat", "kitten", "funny", "jump", "fail", "chase"},
}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Window:
    start: float
    end: float
    score: float
    reason: str = ""


@dataclass
class Analysis:
    mode: str                       # 'speech' | 'visual'
    duration_s: float
    words: list[Word] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)

    @property
    def has_transcript(self) -> bool:
        return len(self.words) >= 50

    def transcript_text(self) -> str:
        """Transcript with [mm:ss] markers (spec §7 text path)."""
        parts = []
        last_marker = -10
        for w in self.words:
            m = int(w.start // 60)
            s = int(w.start % 60)
            if w.start - last_marker >= 5.0 or not parts:
                parts.append(f"[{m:02d}:{s:02d}]")
                last_marker = w.start
            parts.append(w.text.strip())
        return " ".join(parts)


# --- media probes -----------------------------------------------------------
def probe_duration(video: Path, ffprobe: str | None) -> float:
    if not ffprobe:
        return 0.0
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=30,
        )
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def extract_wav(video: Path, ffprobe_unused: str | None, ffmpeg: str) -> Path | None:
    """ffmpeg -> 16 kHz mono pcm wav next to the video (spec §6)."""
    out = video.with_suffix(".16k.wav")
    if out.exists():
        return out
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video), "-ac", "1", "-ar", "16000",
             "-vn", "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True, timeout=600,
        )
        return out if out.exists() else None
    except Exception:
        return None


# --- speech path ------------------------------------------------------------
def transcribe(video: Path, model_name: str, compute_type: str) -> list[Word]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, compute_type=compute_type)
    segments, _info = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(float(w.start), float(w.end), (w.word or "").strip()))
    return words


def _keyword_hits(text: str, topic: str) -> int:
    low = text.lower()
    score = 0
    score += low.count("!")
    score += sum(1 for s in SUPERLATIVES if s in low)
    score += sum(low.count(ch) for ch in "0123456789")
    score += sum(1 for w in TOPIC_WORDS.get(topic, set()) if w in low)
    return score


def speech_windows(words: list[Word], duration_s: float, topic: str,
                   window_s: float, step_s: float, top_k: int) -> list[Window]:
    if not words:
        return []
    windows: list[Window] = []
    start_t = 0.0
    guard = 0
    while start_t + 1.0 < duration_s and guard < 10000:
        guard += 1
        end_t = min(start_t + window_s, duration_s)
        chunk = " ".join(w.text for w in words if start_t <= w.start < end_t)
        score = float(_keyword_hits(chunk, topic))
        windows.append(Window(round(start_t, 2), round(end_t, 2), score,
                              reason="keyword_density"))
        start_t += step_s
    windows.sort(key=lambda w: w.score, reverse=True)
    return windows[:top_k]


# --- visual path ------------------------------------------------------------
def detect_scenes(video: Path) -> list[float]:
    """Return scene-start times (seconds) via PySceneDetect ContentDetector."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(str(video), ContentDetector())
        return [float(s[0].get_seconds()) for s in scenes] if scenes else [0.0]
    except Exception:
        return [0.0]


def rms_profile(wav: Path, step_s: float) -> tuple[np.ndarray, int]:
    """RMS energy per ``step_s`` over the wav. Returns (profile, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(str(wav), dtype="float32")
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    frame = max(1, int(sr * step_s))
    usable = (len(data) // frame) * frame
    data = data[:usable]
    if usable == 0:
        return np.zeros(1, dtype="float32"), sr
    frames = data.reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms, sr


def _smooth(arr: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(arr) < k:
        return arr
    kernel = np.ones(k, dtype="float32") / k
    return np.convolve(arr, kernel, mode="same")


def visual_windows(video: Path, ffmpeg: str, duration_s: float,
                   window_s: float, step_s: float, min_gap: float,
                   top_k: int) -> list[Window]:
    scene_starts = detect_scenes(video)
    wav = extract_wav(video, None, ffmpeg)
    if wav:
        rms, _sr = rms_profile(wav, step_s)
        rms = _smooth(rms, k=3)
    else:
        rms = np.zeros(max(1, int(duration_s / step_s)), dtype="float32")

    n = len(rms)
    if rms.max() > 0:
        rms_n = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        rms_n = rms
    # scene-change density per step bucket
    density = np.zeros(n, dtype="float32")
    for s in scene_starts:
        b = min(int(s / step_s), n - 1)
        if 0 <= b < n:
            density[b] += 1.0
    if density.max() > 0:
        density = density / density.max()

    peak = rms_n + density
    # pick spread-out peaks greedily
    chosen: list[Window] = []
    order = np.argsort(peak)[::-1]
    for i in order:
        t = float(i) * step_s
        start = max(0.0, min(t - window_s / 2, duration_s - window_s))
        end = min(duration_s, start + window_s)
        if any(abs(start - c.start) < min_gap for c in chosen):
            continue
        chosen.append(Window(round(start, 2), round(end, 2), float(peak[i]),
                             reason="audio_energy+scene_density"))
        if len(chosen) >= top_k:
            break
    # keep chronological for downstream keyframe extraction
    chosen.sort(key=lambda w: w.start)
    return chosen


# --- orchestrator -----------------------------------------------------------
def analyze(video: Path, topic: str, mode: str) -> Analysis:
    c = cfg.get_config()
    ffprobe = c.ffprobe
    ffmpeg = c.ffmpeg or "ffmpeg"
    a = c.get("analyzer", {}) or {}
    window_s = float(a.get("window_s", 60))
    duration_s = probe_duration(video, ffprobe)
    if duration_s <= 0:
        duration_s = 600.0  # safe fallback if probe unavailable

    words: list[Word] = []
    use_mode = mode
    if mode in ("speech", "auto"):
        words = transcribe(video, str(c.get("whisper.model", "small")),
                           str(c.get("whisper.compute_type", "int8")))
        min_words = int(a.get("min_words_for_speech", 50))
        if len(words) < min_words:
            if mode == "auto":
                use_mode = "visual"
            elif mode == "speech":
                # speech requested but (almost) silent: still give heuristic windows
                use_mode = "speech"

    if use_mode == "visual":
        windows = visual_windows(
            video, ffmpeg, duration_s, window_s,
            float(a.get("visual_peak_step_s", 0.5)),
            float(a.get("visual_min_gap_s", 90)),
            int(a.get("visual_candidates", 6)),
        )
        return Analysis("visual", duration_s, words, windows)

    windows = speech_windows(
        words, duration_s, topic, window_s,
        float(a.get("speech_step_s", 15)),
        int(a.get("speech_top_candidates", 6)),
    )
    return Analysis("speech", duration_s, words, windows)
