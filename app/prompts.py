"""Editor-AI prompts + response schemas.

``DEFAULT_PROMPT`` (spec §7) is overridable per-run from the UI and persisted
in settings; the placeholder in the settings/dashboard forms is this exact
string so users can see what they're replacing.
"""
from __future__ import annotations

import json

# {N} is formatted at call time. Keep the instruction terse & JSON-only.
DEFAULT_PROMPT = (
    "You are a viral short-form editor. From the material, select exactly {N} "
    "non-overlapping moments of at most 60 seconds each that are the most gripping "
    "(peak action/emotional payoff), each starting ≥2s from video ends. For each "
    "return: start_s, end_s, hook_title (≤40 chars), caption (≤150 chars, TikTok "
    "style, 2 emojis max), reason (≤15 words). Respond with ONLY valid JSON: "
    '{"clips":[{"start_s":..,"end_s":..,"hook_title":"..","caption":"..","reason":".."}]}'
)

# Appended on the single retry (spec §7).
RETRY_SUFFIX = " Return ONLY the JSON object."

# Text-path context header placed before the transcript.
TEXT_HEADER = (
    "MATERIAL\nTitle: {title}\nChannel: {channel}\nDuration_s: {duration_s}\n"
    "Transcript (word timestamps as [mm:ss]):\n{transcript}\n\nTASK\n"
)

# Vision-path header placed before the base64 image blocks.
VISION_HEADER = (
    "MATERIAL\nTitle: {title}\nChannel: {channel}\nDuration_s: {duration_s}\n"
    "Below are candidate windows; each shows 5 evenly-spaced keyframes with their "
    "timestamps. Score every window for short-form virality and pick the best "
    "non-overlapping moments.\n\nTASK\n"
)

# The canonical clip object, used for validation + docs.
CLIP_SCHEMA = {
    "type": "object",
    "required": ["start_s", "end_s", "hook_title", "caption", "reason"],
    "properties": {
        "start_s": {"type": "number", "minimum": 0},
        "end_s": {"type": "number", "minimum": 0},
        "hook_title": {"type": "string", "maxLength": 80},
        "caption": {"type": "string", "maxLength": 300},
        "reason": {"type": "string"},
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["clips"],
    "properties": {"clips": {"type": "array", "items": CLIP_SCHEMA}},
}


def format_prompt(template: str, n: int) -> str:
    """Fill {N} safely (ignore other braces that may appear in a user prompt)."""
    try:
        return template.replace("{N}", str(n))
    except Exception:  # pragma: no cover - defensive
        return template


def schema_example() -> str:
    return json.dumps(RESPONSE_SCHEMA, indent=2)
