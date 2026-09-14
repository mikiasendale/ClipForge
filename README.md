# ClipForge

Autonomous **old-YouTube → TikTok** clip pipeline. Finds vintage channels in your
chosen topics, downloads videos, detects the best moments (local transcription +
scene/audio analysis), picks highlights with an OpenRouter LLM (text or vision),
and renders 9:16 captioned clips — up to a daily quota, never reusing a video.

Runs entirely locally. The only external service is the **OpenRouter API**.

---

## Requirements

- **Python 3.12** (pinned; bootstrap installs it if missing)
- **ffmpeg / ffprobe** (bootstrap installs it; not a pip package)
- An **OpenRouter API key**

No Node.js, no Docker, no cloud beyond OpenRouter.

---

## Quick start

### Windows
```bat
bootstrap.bat
run.bat
```
### Linux / macOS
```bash
./bootstrap.sh
./run.sh
```

Then put your key in `.env`:
```
OPENROUTER_API_KEY=sk-or-...
```

`run.*` starts a local server at **http://127.0.0.1:8000** and opens your browser.

Headless daily job (no UI):
```bash
./run.sh --auto        # or: run.bat --auto
```

---

## Using the app

1. **Onboarding** (`/onboarding`) — each of the 4 topic sections loads candidate
   channels asynchronously. Tick exactly the required count
   (Football 3, Boxing 3, Cats/Silent 2, Cats/Compilations 2). **Save** enables only
   when every quota is met. Use *Deep enrich* to fill subscriber counts / joined
   year / avatars via Scrapling.
2. **Dashboard** (`/dashboard`) — **Run daily job** (rotation across your picked
   channels until the daily quota is hit) or **Run custom…** (optional single URL,
   clip count, prompt override). A live job log polls the background thread.
3. **Clips** (`/clips`) — finished clips grouped by day; play them in the browser.
4. **Settings** (`/settings`) — model names, whisper model, daily quota, clip
   length, face tracking, and the persisted default editor prompt. *Test key*
   pings OpenRouter.

---

## How a run works (`app/job.py`)

`quota=4`. While `clips_today < quota` and `attempts < 6`:

1. next channel by **rotation** (`state.rotation_index`, interleaved by topic)
2. list channel `/videos`, exclude every `video_id` already in the DB, keep
   3–30 min, score by normalized mid-high views, pick the best
3. download 720p H.264 → `data/downloads/`
4. **select clips** — either the **tool-calling agent** (`app/agent.py`, default)
   or the single-shot analyzer→editor path (`agent.enabled: false`)
5. `N = min(4, max(1, round(duration_min/5)))`, clamped to quota remaining
6. cutter renders each window → `clips` rows
7. `used_at` set **only after a successful render**; failed videos get
   `status='failed'`

### Editor agent (`app/agent.py`) — feature A

The LLM drives its own analysis instead of a fixed handoff. It calls local tools
and finishes with the terminal `propose_clips` tool. **Works with text-only
OpenRouter models** (vision is opt-in via `agent.vision_enabled`).

- **tools**: `get_transcript(chunk_index)` (chunked `[mm:ss]`, ~4000 chars),
  `get_scene_list()`, `get_audio_energy(start_s,end_s)` (`mm:ss=0.42` lines,
  >600 s rejected), `get_keyframes(...)` (only if vision enabled),
  `propose_clips(clips)` — terminal; validated for ≤60 s, ≥2 s from ends,
  non-overlapping, in-bounds; invalid proposals return errors so the model
  self-corrects.
- **system prompt** = `DEFAULT_PROMPT` (+ grounding instruction), override-able.
- **on-demand whisper**: transcription runs inside `get_transcript` and is cached
  to `data/transcripts/{video_id}.json` (scenes + audio energy cached too), so
  re-renders and retries never re-transcribe.
- **fallback ladder**: API error → retry once → last valid `propose_clips`
  → the OLD single-shot path (`engine="fallback_single_shot"`). The chosen engine
  is logged per clip and written to `data/logs/agent_{video_id}.jsonl`.

### Review & edit (`/review`) — feature C

Each finished clip has an **Adjust** editor over the **source** video:
- `GET /media/source/{video_id}` streams the download with **manual HTTP Range
  (206/416)** so seeking works.
- Two range sliders set in/out with live timecodes + preview; save is blocked
  (inline error) if length <5 s, >60 s, or overlapping another clip of the video.
- Caption (≤150, live counter) + hook title.
- **Save** → `POST /api/clips` re-runs the cutter (same face-tracking + caption
  settings), re-rendering in seconds (never re-transcribes; reuses the cache).
  `mode:"replace"` marks the old row `revised_at` and links `parent_clip_id`;
  `mode:"new"` adds a manual clip. Superseded versions are hidden from the
  gallery; manual clips don't touch `used_at`.
```yaml
agent: { enabled: true, model: "google/gemini-2.5-flash", max_steps: 8, vision_enabled: false }
```

### Highlight detection (`app/analyzer.py`)
- **speech/auto**: faster-whisper word timestamps. Sliding 60s windows (step 15s)
  ranked by keyword density → top 6. `<50` words in *auto* mode → visual path.
- **visual**: PySceneDetect scenes + ffmpeg 16 kHz mono wav → RMS energy per 0.5 s;
  peak = energy + scene density; 6 windows spread ≥90 s apart.

### Cutter (`app/cutter.py`)
9:16 target 1080×1920 @30fps (a 1280×720 source → `crop=404:720:438:0`).
Segment-wise **face-tracking crop** when faces are in >30% of 0.5s samples,
else center crop. ASS captions (uppercase, ≤3 words/line, lower third, white with
4px black outline). `loudnorm=I=-14`. Output → `data/output/YYYY-MM-DD/…`.

---

## Layout

```
app/  main.py db.py config.py discovery.py selector.py
      downloader.py analyzer.py editor_ai.py cutter.py prompts.py job.py agent.py
      templates/ static/
data/ clipforge.db  downloads/  output/  logs/  frames/  transcripts/  (gitignored)
tests/ bootstrap.sh bootstrap.bat run.sh run.bat config.yaml .env.example
```

## Tests

```bash
.venv/bin/python -m pytest          # 51 tests (network + whisper + OpenRouter mocked; real ffmpeg renders)
```

---

## Manual acceptance checklist (spec §11)

| # | Step | Expected |
|---|------|----------|
| 1 | Fresh clone → `bootstrap` → `run` → open `/onboarding` | 4 sections render; candidates load per-topic (network permitting) |
| 2 | POST `/onboarding` with wrong pick counts | **400**, per-topic error message (also tested) |
| 3 | Daily run on a ~10-min speech video | exactly **2 clips**, each ≤61s, 9:16 (1080×1920), captions burned |
| 4 | Run the daily job twice | second run picks a **different** video (rotation advanced) |
| 5 | Silent cat video, `auto`/`visual` mode | goes through the **vision path** (no transcript) yet returns 60s windows |
| 6 | Custom run with a prompt override | new clip rows have `prompt_source='user'` |
| 7 | Kill the server mid-job, restart, run again | **no duplicate** videos or clips (`used_at` set only after render) |
| 8 | Edit `config.yaml` (e.g. model name) | reflected without code changes |

### Agent + review (features A / C)

| # | Step | Expected |
|---|------|----------|
| A1 | `agent.enabled=true`, text-only model, ~10-min video | 2 clips ≤61s; `data/logs/agent_*.jsonl` shows ≥1 tool call before `propose_clips` |
| A2 | `agent.model` = nonexistent model | run completes via fallback; log says `fallback_single_shot` |
| A3 | `propose_clips` with overlapping windows | model gets a validation error and self-corrects, or falls back — never renders invalid clips |
| C1 | `GET /media/source/{id}` with a `Range` header | **206** + correct `Content-Range`; seeking works in Chromium |
| C2 | Adjust a clip's out-point, save | new file rendered; old row `revised_at` set; gallery shows the new version only |
| C3 | Create a manual clip from a `done` video, >60s | rejected (400); ≤60s accepted, `engine=manual` |

---

## Troubleshooting

- **No candidates on onboarding** — network/YouTube throttling; click *Refresh
  search*, or *Deep enrich* once Scrapling's browser is installed
  (`.venv/bin/scrapling install`).
- **Editor AI fell back to heuristics** — check `OPENROUTER_API_KEY`, then
  `data/logs/openrouter.log` (model, latency, tokens, errors).
- **ffmpeg not found** — re-run bootstrap; the launcher also checks `tools/`.

## Security notes

- `.env` (secrets) is gitignored; only `.env.example` is committed.
- Server binds `127.0.0.1` only.
- All downloaded/rendered media stays under `data/` (gitignored).
