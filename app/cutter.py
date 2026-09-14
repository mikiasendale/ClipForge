"""Clip rendering with raw ffmpeg subprocess (spec §8) — no moviepy.

Target 1080x1920 @30fps. From a 1280x720 source the 9:16 crop is
``crop=404:720:438:0`` scaled to ``1080:1920``; the crop width is computed
generally as ``height * 9/16`` so any input works.

Face tracking (``clip.face_tracking`` != ``off``): sample faces every 0.5 s
(OpenCV YuNet if an .onnx is bundled, else the Haar cascade). If faces appear
in >``face_min_ratio`` of samples, render 2 s chunks each cropped on a moving
average centroid (3 s window, clamped) and concat them; otherwise center crop.

Captions: whisper word timestamps -> ASS (uppercase, ~54px, white + 4px black
outline, <=3 words/line, lower third). Burned via ``-vf ass=``.
Audio: ``loudnorm=I=-14``. Output: ``data/output/YYYY-MM-DD/...``.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config as cfg


@dataclass
class SourceDims:
    width: int = 1280
    height: int = 720


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --- probes -----------------------------------------------------------------
def probe_dims(src: Path, ffprobe: str | None) -> SourceDims:
    if not ffprobe:
        return SourceDims()
    try:
        proc = _run([ffprobe, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height",
                     "-of", "json", str(src)], timeout=30)
        st = (proc.stdout and json.loads(proc.stdout).get("streams") or [{}])[0]
        return SourceDims(int(st.get("width") or 1280), int(st.get("height") or 720))
    except Exception:
        return SourceDims()


# --- geometry ---------------------------------------------------------------
def _even(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 0 else n - 1


def crop_width(dims: SourceDims) -> int:
    cw = _even(dims.height * 9 / 16)
    return max(2, min(cw, dims.width))


def center_crop(dims: SourceDims, cw: int) -> int:
    x = (dims.width - cw) // 2
    return _even(max(0, min(x, dims.width - cw)))


# --- ASS captions -----------------------------------------------------------
def build_ass(words, s: float, e: float, out_path: Path, font: str,
              fontsize: int, wpl: int, W: int, H: int) -> bool:
    """Write an ASS subtitle for words in [s,e]. Returns True if events written."""
    import re as _re
    inwin = sorted([w for w in words if s <= w.start < e], key=lambda w: w.start)
    if not inwin:
        return False
    lines: list[tuple[float, float, str]] = []
    for i in range(0, len(inwin), max(1, wpl)):
        grp = inwin[i:i + wpl]
        lines.append((grp[0].start, grp[-1].end,
                      " ".join(w.text for w in grp).upper()))

    def ts(t: float) -> str:
        t = max(0.0, t - s)
        h = int(t // 3600); m = int((t % 3600) // 60); sec = t % 60
        return f"{h}:{m:02d}:{sec:05.2f}"

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,0,0,1,4,1,2,60,60," + str(int(H * 0.28)) + ",1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for a, b, text in lines:
        safe = _re.sub(r"\{", "(", text)
        events.append(f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{safe}")
    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return True


# --- face detection ---------------------------------------------------------
def _make_detector():
    try:
        import cv2  # noqa
    except Exception:
        return None
    import glob
    root = cfg.PROJECT_ROOT
    yunet = None
    for pat in ("tools/**/*.onnx", "app/**/*.onnx", "data/**/*.onnx"):
        hits = glob.glob(str(root / pat), recursive=True)
        if hits:
            yunet = hits[0]
            break
    if yunet and hasattr(cv2, "FaceDetectorYN"):
        try:
            det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.5, 0.3, 0.2)
            return ("yunet", det)
        except Exception:
            pass
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cascade.empty():
            return ("haar", cascade)
    except Exception:
        pass
    return None


def _detect_face(detector, frame):
    kind, det = detector
    import cv2
    if kind == "yunet":
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        try:
            _, faces = det.detect(frame)
        except Exception:
            return None
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        return (best[0] + best[2] / 2, best[1] + best[3] / 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (x + fw / 2, y + fh / 2)


def face_trajectory(src: Path, s: float, e: float, dims: SourceDims,
                    step_s: float) -> list[tuple[float, float | None]]:
    """Return [(t, center_x or None)] sampled every step_s across [s,e]."""
    detector = _make_detector()
    if detector is None:
        return []
    try:
        import cv2
    except Exception:
        return []
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return []
    traj: list[tuple[float, float | None]] = []
    t = s
    while t <= e + 0.001:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        except Exception:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            t += step_s
            traj.append((round(t, 2), None))
            continue
        c = None
        try:
            face = _detect_face(detector, frame)
            if face:
                c = float(face[0])
        except Exception:
            c = None
        traj.append((round(t, 2), c))
        t += step_s
    cap.release()
    return traj


def _smooth_centroids(traj: list[tuple[float, float | None]], window_s: float,
                      fallback: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    vals = [v for _, v in traj if v is not None]
    for t, v in traj:
        neigh = [vv for tt, vv in traj if vv is not None and abs(tt - t) <= window_s / 2]
        if neigh:
            out.append((t, sum(neigh) / len(neigh)))
        else:
            out.append((t, v if v is not None else (sum(vals) / len(vals) if vals else fallback)))
    return out


# --- ffmpeg filter builders -------------------------------------------------
def _base_vf(cw: int, x: int, dims: SourceDims, W: int, H: int, fps: int,
             ass: Path | None) -> str:
    parts = [
        f"crop={cw}:{dims.height}:{x}:0",
        f"scale={W}:{H}",
        f"fps={fps}",
        "setsar=1",
    ]
    if ass:
        parts.append(f"ass=filename='{ass.as_posix()}'")
    parts.append("format=yuv420p")
    return ",".join(parts)


def _encode_tail(out_path: Path, crf: int, preset: str, abitrate: str) -> list[str]:
    return [
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-c:a", "aac", "-b:a", abitrate, "-movflags", "+faststart",
        "-y", str(out_path),
    ]


def _loudnorm(clip_cfg) -> list[str]:
    i = clip_cfg.get("loudnorm_i", -14)
    return ["-af", f"loudnorm=I={i}:TP=-1.5:LRA=11"]


def render_single(src: Path, s: float, e: float, out_path: Path, ass: Path | None) -> Path | None:
    """One-pass center-crop render (used when face tracking is off / not useful)."""
    c = cfg.get_config()
    ffmpeg = c.ffmpeg
    if not ffmpeg:
        return None
    clip_cfg = c.get("clip", {}) or {}
    dims = probe_dims(src, c.ffprobe)
    cw = crop_width(dims)
    x = center_crop(dims, cw)
    W = int(clip_cfg.get("target_width", 1080)); H = int(clip_cfg.get("target_height", 1920))
    fps = int(clip_cfg.get("fps", 30))
    vf = _base_vf(cw, x, dims, W, H, fps, ass)
    cmd = [ffmpeg, "-y", "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", str(src)]
    cmd += ["-vf", vf]
    cmd += _loudnorm(clip_cfg)
    cmd += _encode_tail(out_path, int(clip_cfg.get("crf", 20)),
                        str(clip_cfg.get("preset", "veryfast")),
                        str(clip_cfg.get("audio_bitrate", "128k")))
    try:
        proc = _run(cmd, timeout=1800)
    except Exception:
        return None
    return out_path if out_path.is_file() and proc.returncode == 0 else None


# --- tracked render (chunk + concat) ----------------------------------------
def render_tracked(src: Path, s: float, e: float, centroids: list[tuple[float, float]],
                   out_path: Path, ass: Path | None) -> Path | None:
    c = cfg.get_config()
    ffmpeg = c.ffmpeg
    if not ffmpeg:
        return None
    clip_cfg = c.get("clip", {}) or {}
    dims = probe_dims(src, c.ffprobe)
    cw = crop_width(dims)
    W = int(clip_cfg.get("target_width", 1080)); H = int(clip_cfg.get("target_height", 1920))
    fps = int(clip_cfg.get("fps", 30))
    chunk = float(clip_cfg.get("face_chunk_s", 2.0))
    x_center = center_crop(dims, cw)

    # per-chunk smoothed crop x
    bounds: list[tuple[float, float, int]] = []
    t = s
    while t < e - 0.05:
        ce = min(t + chunk, e)
        neigh = [cx for ct, cx in centroids if t <= ct < ce]
        if neigh:
            cx = sum(neigh) / len(neigh)
            x = _even(max(0, min(int(cx - cw / 2), dims.width - cw)))
        else:
            x = x_center
        bounds.append((t, ce, x))
        t = ce

    tmp = out_path.parent / f".tmp_{out_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    chunk_files: list[Path] = []
    for i, (cs, ce, x) in enumerate(bounds):
        cf = tmp / f"chunk_{i:03d}.mp4"
        vf = _base_vf(cw, x, dims, W, H, fps, ass=None)  # captions burned in final pass
        cmd = [ffmpeg, "-y", "-ss", f"{cs:.2f}", "-to", f"{ce:.2f}", "-i", str(src),
               "-an", "-vf", vf] + _encode_tail(cf, int(clip_cfg.get("crf", 20)),
                                                 str(clip_cfg.get("preset", "veryfast")),
                                                 "128k")
        # strip audio codec from tail for video-only chunk
        cmd = _drop_audio_args(cmd)
        try:
            _run(cmd, timeout=300)
        except Exception:
            return None
        if cf.is_file():
            chunk_files.append(cf)
    if not chunk_files:
        return None

    listfile = tmp / "concat.txt"
    listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in chunk_files), encoding="utf-8")
    joined = tmp / "joined.mp4"
    try:
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
              "-c", "copy", str(joined)], timeout=300)
    except Exception:
        return None
    if not joined.is_file():
        return None

    # final pass: add windowed audio from src + burn captions + loudnorm
    vf_final = "format=yuv420p"
    if ass:
        vf_final = f"ass=filename='{ass.as_posix()}',format=yuv420p"
    cmd = [ffmpeg, "-y",
           "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", str(src),   # 0: audio source (windowed)
           "-i", str(joined),                                       # 1: tracked video
           "-map", "1:v:0", "-map", "0:a:0",
           "-vf", vf_final] + _loudnorm(clip_cfg)
    cmd += ["-c:v", "libx264", "-crf", str(clip_cfg.get("crf", 20)),
            "-preset", str(clip_cfg.get("preset", "veryfast")),
            "-c:a", "aac", "-b:a", str(clip_cfg.get("audio_bitrate", "128k")),
            "-shortest", "-movflags", "+faststart", str(out_path)]
    try:
        _run(cmd, timeout=600)
    except Exception:
        _cleanup_tmp(tmp)
        return None
    _cleanup_tmp(tmp)
    return out_path if out_path.is_file() else None


def _cleanup_tmp(tmp: Path) -> None:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def _drop_audio_args(cmd: list[str]) -> list[str]:
    out: list[str] = []
    skip = 0
    for tok in cmd:
        if skip:
            skip -= 1
            continue
        if tok in ("-c:a", "-b:a"):
            skip = 1
            continue
        out.append(tok)
    return out


# --- public API -------------------------------------------------------------
def output_path_for(topic: str, channel: str, title: str, idx: int) -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    safe = lambda t: "".join(ch for ch in (t or "") if ch.isalnum() or ch in ("-", "_", " "))[:60]
    name = f"{safe(topic)}_{safe(channel)}_{safe(title)}_{idx:02d}.mp4"
    d = cfg.OUTPUT_DIR / day
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def render_clip(src: Path, s: float, e: float, out_path: Path, words=None) -> Path | None:
    """Render one 9:16 clip. Chooses tracked vs center crop per config (§8)."""
    c = cfg.get_config()
    clip_cfg = c.get("clip", {}) or {}
    mode = str(clip_cfg.get("face_tracking", "auto"))

    ass: Path | None = None
    if words:
        ass_path = out_path.with_suffix(".ass")
        ok = build_ass(words, s, e, ass_path,
                       str(clip_cfg.get("caption_font", "Arial Black")),
                       int(clip_cfg.get("caption_fontsize", 54)),
                       int(clip_cfg.get("caption_words_per_line", 3)),
                       int(clip_cfg.get("target_width", 1080)),
                       int(clip_cfg.get("target_height", 1920)))
        ass = ass_path if ok else None

    if mode == "off":
        return render_single(src, s, e, out_path, ass)

    dims = probe_dims(src, c.ffprobe)
    cw = crop_width(dims)
    step = 0.5
    traj = face_trajectory(src, s, e, dims, step)
    hits = sum(1 for _, cx in traj if cx is not None)
    ratio = hits / len(traj) if traj else 0.0
    min_ratio = float(clip_cfg.get("face_min_ratio", 0.30))
    if not traj or ratio < min_ratio:
        return render_single(src, s, e, out_path, ass)  # fall back to center crop

    window_s = float(clip_cfg.get("face_smooth_window_s", 3.0))
    centroids = _smooth_centroids(traj, window_s, center_crop(dims, cw) + cw / 2)
    return render_tracked(src, s, e, centroids, out_path, ass)
