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
4. analyze → `N = min(4, max(1, round(duration_min/5)))`, clamped to quota remaining
5. editor AI selects N windows (text path from transcript, or vision path from
   keyframes for silent clips) → cutter renders → `clips` rows
6. `used_at` set **only after a successful render**; failed videos get
   `status='failed'`

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
      downloader.py analyzer.py editor_ai.py cutter.py prompts.py job.py
      templates/ static/
data/ clipforge.db  downloads/  output/  logs/  frames/   (gitignored)
tests/ bootstrap.sh bootstrap.bat run.sh run.bat config.yaml .env.example
```

## Tests

```bash
.venv/bin/python -m pytest          # 40 tests (network + whisper mocked; real ffmpeg renders)
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
