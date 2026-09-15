"""CapCut / JianYing draft export (feature B) — an ALTERNATIVE output mode.

Instead of rendering an mp4, we hand the agent's chosen windows to a CapCut
desktop *project* (a "draft"): the source mp4 is referenced and TRIMMED in the
timeline (never pre-cut), center-cropped to 9:16, with a static per-clip caption
text segment. The user opens the draft in CapCut, tweaks, and exports manually.

Honest limitations (surfaced in the UI + README):
  * pyJianYingDraft writes a reverse-engineered format; a CapCut auto-update can
    break draft loading. mp4 mode is fully independent of this module.
  * Face-tracking crop and word-synced captions are mp4-only; drafts get a
    center crop + one static caption per clip.
  * Final export is always manual (no open-source path to render CapCut drafts).
"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config as cfg

# --- import guard (B1): missing/broken lib disables capcut, never crashes ---
try:
    import pyJianYingDraft as _jd
    from pyJianYingDraft import (  # type: ignore
        TrackType, TrackSpec, Timerange, CropSettings, ClipSettings,
        VideoMaterial, VideoSegment, TextSegment, TextStyle, TextBorder, SEC,
    )
    CAPCUT_AVAILABLE = True
    IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover - exercised via monkeypatch in tests
    _jd = None
    CAPCUT_AVAILABLE = False
    IMPORT_ERROR = str(e)

VERSION_NOTE = "pyJianYingDraft 0.3.0 (reverse-engineered CapCut draft format)"


# --- draft root detection (B2) ----------------------------------------------
def _windows_candidates() -> list[Path]:
    local = os.environ.get("LOCALAPPDATA") or ""
    cands: list[Path] = []
    if local:
        base = Path(local)
        cands += [
            base / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft",
            base / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
        ]
        # generic scan: LOCALAPPDATA/*/User Data/Projects/com.lveditor.draft
        try:
            for d in base.iterdir():
                if d.is_dir() and re.search(r"jianying|capcut", d.name, re.I):
                    cands.append(d / "User Data" / "Projects" / "com.lveditor.draft")
        except OSError:
            pass
    return cands


def is_windows() -> bool:
    return CAPCUT_AVAILABLE and bool(getattr(_jd, "ISWIN", False)) or os.name == "nt"


def detect_draft_root() -> dict[str, Any]:
    """Resolve + validate the draft root. Returns a report dict for the UI."""
    c = cfg.get_config()
    explicit = str(c.get("capcut.draft_root", "auto") or "auto")
    detected = None
    source = None
    if explicit and explicit != "auto":
        detected = Path(explicit)
        source = "config"
    elif is_windows():
        for cand in _windows_candidates():
            if cand.exists() and cand.is_dir():
                detected, source = cand, "auto"
                break
    result = {
        "available": CAPCUT_AVAILABLE,
        "import_error": IMPORT_ERROR,
        "draft_root": str(detected) if detected else None,
        "source": source,
        "writable": False,
        "error": None,
        "windows": is_windows(),
    }
    if not CAPCUT_AVAILABLE:
        result["error"] = "CapCut export unavailable — pip install failed"
        return result
    if detected is None:
        if is_windows():
            result["error"] = ("No JianYing/CapCut draft root found. Launch CapCut "
                               "once (so it creates its Projects folder) or set "
                               "capcut.draft_root in config.yaml.")
        else:
            result["error"] = ("Set capcut.draft_root in config.yaml to your CapCut "
                               "drafts folder (e.g. ~/Movies/JianyingPro/.../com.lveditor.draft). "
                               "No auto-detect path on this OS.")
        return result
    result["writable"] = _is_writable(detected)
    if not result["writable"]:
        result["error"] = f"draft root not writable: {detected}"
    return result


def _is_writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".clipforge_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def test_write_draft() -> tuple[bool, str]:
    """[Test] button: create + delete a dummy draft in the resolved root."""
    report = detect_draft_root()
    if not report["available"]:
        return False, report["error"] or "capcut unavailable"
    root = report["draft_root"]
    if not root:
        return False, "no draft root detected — set capcut.draft_root in config.yaml"
    if not report["writable"]:
        return False, f"draft root not writable: {root}"
    try:
        df = _jd.DraftFolder(root)
        sf = df.create_draft(f"ClipForge_test_{datetime.now():%Y%m%d%H%M%S}", 1080, 1920, 30)
        sf.save()
        folder = Path(sf.save_path).parent
        shutil.rmtree(folder, ignore_errors=True)
        return True, f"OK — wrote and removed a test draft in {root}"
    except Exception as e:  # pragma: no cover
        return False, f"write test failed: {e}"


def _draft_root_for_write() -> str:
    report = detect_draft_root()
    if not report["available"]:
        raise RuntimeError(report["error"] or "capcut unavailable")
    if not report["draft_root"] or not report["writable"]:
        raise RuntimeError(report["error"] or "draft root unwritable")
    return report["draft_root"]


def _unique_name(folder: Path, base: str) -> str:
    name = base
    n = 2
    while (folder / name).exists():
        name = f"{base}_{n}"
        n += 1
    return name


def _short_title(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (title or "video")).strip("_")
    return s[:28] or "video"


# --- crop math --------------------------------------------------------------
def _center_crop(src_w: int, src_h: int) -> "CropSettings":
    tgt = 9 / 16
    cur = (src_w / src_h) if src_h else 1.0
    if cur > tgt:                       # too wide -> trim sides
        keep = tgt / cur
        lo, hi = (1 - keep) / 2, (1 - keep) / 2 + keep
        return CropSettings(upper_left_x=lo, upper_right_x=hi,
                            lower_left_x=lo, lower_right_x=hi)
    if cur < tgt:                       # too tall -> trim top/bottom
        keep = cur / tgt
        lo, hi = (1 - keep) / 2, (1 - keep) / 2 + keep
        return CropSettings(upper_left_y=lo, upper_right_y=lo,
                            lower_left_y=hi, lower_right_y=hi)
    return CropSettings()


# --- main export ------------------------------------------------------------
def export_draft(video_row: dict, clips: list[dict], dest_name: str | None = None,
                 *, src_path: str | Path | None = None) -> str:
    """Write a CapCut draft folder for `clips` and return its path.

    Raises RuntimeError if capcut is disabled / draft root is unwritable / the
    source file is missing. Never overwrites an existing draft (unique suffix).
    """
    if not CAPCUT_AVAILABLE:
        raise RuntimeError("CapCut export unavailable — pip install failed")
    root = Path(_draft_root_for_write())
    from . import downloader
    src = Path(src_path) if src_path else downloader.resolve_source(video_row["video_id"])
    if not src or not Path(src).is_file():
        raise RuntimeError(f"source file missing for {video_row.get('video_id')}")

    date = datetime.now().strftime("%Y-%m-%d")
    if dest_name is None:
        idx = int(video_row.get("_clip_index", 0))
        dest_name = f"ClipForge_{date}_{_short_title(video_row.get('title') or '')}_{idx:02d}"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", dest_name)[:80]
    name = _unique_name(root, base)

    c = cfg.get_config()
    W = int(c.get("clip.target_width", 1080))
    H = int(c.get("clip.target_height", 1920))
    FPS = int(c.get("clip.fps", 30))

    df = _jd.DraftFolder(str(root))
    sf = df.create_draft(name, W, H, FPS, allow_replace=False)

    # probe once, reuse for every segment
    probe = _jd.VideoMaterial(str(src))
    crop = _center_crop(probe.width, probe.height)
    sf.append_track(TrackSpec(TrackType.video, name="clips"))
    sf.append_track(TrackSpec(TrackType.text, name="captions"))

    target_t = 0
    for cl in clips:
        s = float(cl["start_s"])
        e = float(cl["end_s"])
        dur = max(0.5, e - s)
        vm = VideoMaterial(str(src), crop_settings=crop)
        seg = VideoSegment(vm, Timerange(target_t, int(dur * SEC)),
                           source_timerange=Timerange(int(s * SEC), int(dur * SEC)))
        sf.add_segment(seg, track="clips")
        caption = (cl.get("caption") or cl.get("hook_title") or "").strip()
        if caption:
            text = TextSegment(caption[:150], Timerange(target_t, int(dur * SEC)),
                               style=TextStyle(size=15, bold=True, color=(1.0, 1.0, 1.0)),
                               border=TextBorder(color=(0.0, 0.0, 0.0), width=30.0),
                               clip_settings=ClipSettings(transform_y=-0.35))
            sf.add_segment(text, track="captions")
        target_t += int(dur * SEC)

    sf.save()
    return str(Path(sf.save_path).parent)


def remove_draft(draft_path: str | None) -> bool:
    """Delete a superseded draft folder (used by /review replace to avoid orphans)."""
    if not draft_path:
        return False
    p = Path(draft_path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        return not p.exists()
    return False
