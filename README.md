# whisper-transcribe

GPU-accelerated transcription + Discord TL;DW bot. Five content flows:

1. **Videos with speech** → whisperX (faster-whisper + wav2vec2 alignment + pyannote diarization) → LLM summary, plus a 4th *Community Reaction* embed pulling top YouTube comments
2. **Videos without speech** (music videos, silent gameplay, ASMR) → frame extraction + VLM descriptions → LLM summary
3. **Web articles** → Crawl4AI / FlareSolverr → LLM summary (triggered by replying `tldr` or `summarize` to a Discord message containing a URL). Reddit + HackerNews URLs get a structured fetch (post + linked article + top comments).
4. **AI litmus test** → regex stylistic scan + Wayback / AdSense / author-byline metadata + ambiguous-case LLM qualitative read → forensic signals report (no verdict). Triggered by replying `litmus` to a URL message.
5. **Images** (screenshots, memes, documents, photos) → EasyOCR text extraction + VLM scene description → LLM summary. Triggered by replying `tldr` to a Discord message with image attachments — OCR handles faithful transcription of any visible text, the vision model describes the scene, and the LLM combines both into a concise summary.

React SPA + HTTP API for the whisper service; Discord bot for hands-off summarisation.

## Quick Start

### Prerequisites

- **NVIDIA GPU** with current drivers + nvidia-container-toolkit (whisper service requires CUDA; CPU-only runs work but slowly)
- **Docker Engine** with `docker compose` v2
- **An OpenAI-compatible LLM endpoint**. The defaults assume **Ollama** running on the host (`ollama serve` + `ollama pull llama3.1` is enough). See `LLM_API_URL` below for alternatives.

### Five-minute path (Ollama on host)

```bash
git clone https://github.com/erfianugrah/whisper-transcribe.git
cd whisper-transcribe

# Optional: enable diarization (Discord bot still works without it)
echo "HF_TOKEN=hf_..." > .env

# Optional: enable the Discord bot (skip if you only want the whisper API/UI)
cp bot/.env.example bot/.env
$EDITOR bot/.env   # set DISCORD_TOKEN

make build && make up
# Whisper UI:        http://localhost:7860
# Whisper API ping:  curl http://localhost:7860/api/status
```

The defaults point `LLM_API_URL` and `LLM_VISION_API_URL` at
`http://model_proxy:11434/v1` on the external `llmc` network owned by
[llm-compose](https://github.com/erfianugrah/llm-compose). Bring
`llm-compose` up first (`cd ~/llm-compose && make up`) — this stack
declares the network `external: true` and refuses to start with a clean
"network llmc not found" error otherwise.

To use a different LLM provider (host-side Ollama, llama.cpp, vLLM,
hosted APIs), override `LLM_API_URL` / `LLM_MODEL` / `LLM_VISION_API_URL`
/ `LLM_VISION_MODEL` in `bot/.env` — see `.env.example` for examples.

### Run without llm-compose (transcription only)

If you only need the transcription API/UI (e.g. driving `:7860` from a
tool or agent) and don't want to run the GPU LLM stack at all, use the
standalone overlay instead of `make up`:

```bash
make up-standalone     # valkey + whisper + whisper-live, no llm-compose
make down-standalone   # tear it down
```

This layers `compose.standalone.yaml`, which redefines `llmc` as a
self-managed bridge (under a distinct name, `whisper-standalone-llmc`,
so it never collides with llm-compose's real `llmc`). The co-deployed
default (`make up`) is unchanged — when llm-compose is running, whisper
still reaches `model_proxy` over the external `llmc` as before.

In standalone mode the LLM-dependent extras (bot summaries, `/api/describe`
VLM frame description, scene synthesis) can't reach `model_proxy` and
degrade gracefully; core transcription is fully functional. Point the
`LLM_*_API_URL` vars at any reachable OpenAI-compatible endpoint if you
want those extras to work standalone.

## Features

### Whisper service
- **WhisperX pipeline**: faster-whisper transcription → wav2vec2 word alignment → pyannote speaker diarization
- **Models**: tiny, base, small, medium, large (v3), turbo (large-v3-turbo)
- **Output formats**: txt, srt, vtt, json (with word-level timestamps)
- **Speaker diarization**: identifies who is speaking (requires HF_TOKEN)
- **VLM fallback**: silent / music videos summarised via frame descriptions
- **YouTube download**: yt-dlp + deno for JS-challenged streams (YouTube Music)
- **Idle VRAM management**: models auto-unload after 5min idle (configurable via MODEL_IDLE_TIMEOUT)
- **MCP integration**: tools for OpenCode LLM workflow (download → transcribe → summarize)

### Discord bot
- **Auto-summarise videos** when their URL is posted (YouTube + 17 other platforms; URL-shape-aware so Reddit/Twitter text posts don't auto-trigger)
- **YouTube comments** — top 100 fetched via yt-dlp, filtered (creator-hearted/pinned prioritised), summarised as a 4th "Community Reaction" embed
- **`tldr` reply trigger** on web URLs → scrape + summarise. Reddit + HackerNews get structured fetches (linked article + OP body + top comments)
- **`tldr` reply trigger** on image attachments → EasyOCR + VLM describe + LLM summary. Up to 4 images per message; per-image cap 32MB. Screenshots, documents, memes, photos all supported.
- **`litmus` reply trigger** → AI-litmus forensic report (LLM-tic phrases, em-dash density, Wayback domain age, AdSense, author byline; LLM qualitative read on ambiguous middle range)
- **Silent-video fallback** — VLM frame descriptions when speech density is low
- **Voice-channel live transcription**: `/transcribe-join` streams the live voice call to whisper-live and posts a running transcript — **one stream per speaker**, attributed by display name and timestamped — into a dedicated per-call thread; `/transcribe-leave` stops and posts an **LLM summary** of the call. Per-speaker streams **auto-reconnect** if whisper-live blips; `/transcribe-optout` lets a participant exclude themselves. Transparently handles Discord's mandatory DAVE end-to-end voice encryption (discord.py 2.7.1+). See [docs/discord-bot-guide.md](docs/discord-bot-guide.md#voice-channel-live-transcription).
- **Slash commands**: `/summarize`, `/transcribe`, `/status`, `/find`, `/config`, `/serverconfig`, `/transcribe-join`, `/transcribe-leave`, `/transcribe-cleanup`, `/transcribe-optout`
- **Per-channel + per-guild config**: model override, VLM toggle, YT-comments toggle, diarize default, summary archive channel
- **Speaker rename**: button on diarized summaries lets users label SPEAKER_xx → real names
- **Rate limiting**: per-user sliding window + global queue cap

## HTTP API (whisper service)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | GPU info, service status, VLM availability |
| `/api/yt-download` | POST | Download audio (and optionally video) from any yt-dlp-supported URL |
| `/api/jobs` | POST | Submit a transcription job (async, queued) — **preferred** |
| `/api/jobs/{id}` | GET | Job status + result |
| `/api/jobs/{id}` | DELETE | Cancel a queued job (running jobs can't be cancelled) |
| `/api/image` | POST | OCR + VLM describe a single uploaded image (multipart `file` field). Returns `{ocr, description, width, height, model, bytes}`. Used by the bot's image-summary reply trigger. |
| `/api/queue` | GET | Queue depth, active jobs, recent terminal jobs |
| `/api/transcribe` | POST | Transcribe (deprecated sync wrapper around `/api/jobs`) |
| `/api/describe` | POST | Extract frames + describe via VLM (silent-video fallback) |
| `/api/cleanup` | POST | Remove a yt-dlp temp file (best-effort) |

### POST /api/yt-download

```json
{"url": "https://youtube.com/watch?v=...", "playlist": false}
```

Returns: `{"filename": "...", "title": "...", "duration": 123}`

### POST /api/jobs (preferred)

Async job submission. All consumers (bot, MCP, web SPA, ad-hoc curl)
share a single Valkey-backed FIFO — no more 409 races on concurrent
submits. Submitter gets a `job_id` immediately and polls
`GET /api/jobs/{id}` until terminal.

```json
{
  "file_path": "/tmp/yt-dlp-xxx/id.wav",
  "model": "turbo",
  "language": "Auto-detect",
  "format": "txt",
  "diarize": false,
  "translate": "auto",
  "cleanup": true,
  "consumer": "my-script"
}
```

Returns 202: `{"job_id": "wbx_a1b2c3d4", "status": "queued", "position": 3}`

**`translate`** (default `"auto"`):
- `"auto"` — server runs a 30s LID pre-pass and translates non-English
  sources to English. Good default for LLM-summarisation use cases
  (CS-FLEURS: Whisper translates code-switched audio cleanly while ASR
  CER doubles).
- `true` — force `task=translate` regardless of source.
- `false` — preserve source language. wav2vec2 alignment runs only when
  the detected language has a default aligner in whisperX (~40 of
  Whisper's 100 supported languages); others gracefully skip alignment
  and return segment-level timestamps only.

### GET /api/jobs/{id}

Status snapshot. Shape varies by state:

```json
// queued
{"status": "queued", "position": 2, "submitted_at": "2026-05-12T13:57:00Z"}

// running
{"status": "running", "started_at": "..."}

// done
{"status": "done", "result": {"status": "Done -- 4321 segments",
                              "transcript": "...", "subtitle_file": null,
                              "cached": false},
 "completed_at": "..."}

// failed
{"status": "failed", "error": "CUDA OOM", "permanent": false, "completed_at": "..."}
```

### POST /api/transcribe (legacy)

**Deprecated** — prefer `/api/jobs`. Returns 202 + `job_id` by default
(same shape as `/api/jobs`). Pass `"wait": true` in the body for the
legacy sync behaviour where the call blocks until completion and the
response is the result inline.

```json
{
  "file_path": "/tmp/yt-dlp-xxx/id.wav",
  "model": "turbo",
  "wait": true,
  "cleanup": true
}
```

When `wait=true` returns `{"status": "Done -- ...", "transcript": "...",
"subtitle_file": "..."}` exactly as before. When `wait=false` (default),
returns 202 + `{"job_id": "...", ...}` — caller polls `/api/jobs/{id}`.

If Valkey is unreachable, this endpoint falls through to a single-slot
lock-based path and returns 409 when busy (pre-queue behaviour).

## MCP Server (optional)

The HTTP API is fully usable as-is. If you also want OpenCode (or any other
MCP client) to call whisper directly via tools, the companion MCP server
ships in [llm-compose](https://github.com/erfianugrah/llm-compose) at
`mcp/whisper-server.py`. It's an OpenAI-tool-shape wrapper around the same
`/api/*` endpoints; nothing whisper-specific lives in it.

**Tools exposed:**
- `whisper_status` — service health check
- `yt_download` — download YouTube audio
- `whisper_transcribe` — transcribe local file
- `yt_transcribe` — download + transcribe (one-shot for summaries)
- `yt_transcribe_playlist` — process entire playlists

To wire up: copy `whisper-server.py` from llm-compose into your MCP server
directory and register it in your OpenCode config. Or call the HTTP API
directly — same surface area.

## Environment Variables

### Whisper service
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token for pyannote diarization models |
| `DEBUG_MODE` | `1` | Verbose logging |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds before unloading models from VRAM |
| `LLM_VISION_MODEL` | `Qwen3-VL-2B-Instruct-Q8_0` | VLM used by `/api/describe` |
| `VLM_FRAME_CONCURRENCY` | `4` | Parallel frame description requests |
| `ALIGN_MODEL_CACHE_SIZE` | `4` | LRU cap for wav2vec2 alignment models |
| `YT_DLP_COOKIES_FILE` | — | Optional cookies.txt path for age-gated / bot-flagged videos |
| `VALKEY_URL` | `redis://valkey:6379/0` | Job queue backing store |
| `TRANSCRIPT_CACHE_TTL` | `604800` | Shared transcript cache TTL (seconds; default 7d) |
| `JOB_TTL` | `3600` | How long terminal job hashes stick around (seconds) |
| `JOB_RECENT_LIMIT` | `100` | Cap on `jobs:recent` list (for `/api/queue`) |
| `WORKER_CONCURRENCY` | `1` | Number of parallel transcription workers (single GPU → 1) |
| `IMAGE_MAX_BYTES` | `33554432` | Server-side per-image upload cap on `/api/image` (32MB) |
| `IMAGE_VLM_PROMPT` | (default) | Override the VLM prompt used for single-image describe calls |
| `VLM_OCR_ENABLED` | `1` | Whether EasyOCR runs (set `0` to skip OCR on `/api/image` and `/api/describe`) |
| `VLM_OCR_LANGUAGES` | `en` | CSV of EasyOCR language codes (e.g. `en,fr,de`) |

### whisper-live (streaming) service
Used by the SPA Live tab (microphone **or** system/OBS audio via screen-share), the standalone `live-tap/` CLI, and the Discord voice bot (all stream raw 16 kHz mono PCM to `/ws-stream`).
| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_MODEL` | `large-v3` | faster-whisper model for the streaming path |
| `LIVE_COMPUTE_TYPE` | `float16` | Compute type (`float16` on GPU) |
| `LIVE_MAX_STREAMS` | `4` | Concurrent streaming sessions (shared by SPA + every voice speaker) |
| `LIVE_PROCESS_INTERVAL` | `0.4` | Seconds between inference passes on the growing buffer (lower = snappier) |
| `LIVE_MIN_CHUNK_S` | `0.5` | Minimum audio buffered before the first inference |
| `LIVE_TAIL_SILENCE_S` | `0.4` | Trailing silence that marks end-of-utterance (commit boundary) |
| `LIVE_BEAM_SIZE` | `1` | Greedy decode (beam=1) ~halves per-pass latency vs beam=5 |
| `LIVE_HALLUCINATION_SILENCE_S` | `2.0` | Skip silent gaps longer than this where whisper hallucinates continuation text (`0` disables) |

### Discord bot
| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_TOKEN` | — | **Required**. Bot token from Discord developer portal |
| `WHISPER_API_URL` | `http://whisper:7860` | Whisper service location (compose-provided) |
| `SCRAPER_API_URL` | `http://crawl4ai:11235` | Crawl4AI location |
| `FLARESOLVERR_API_URL` | `http://flaresolverr:8191/v1` | FlareSolverr location |
| `LLM_API_URL` | `http://model_proxy:11434/v1` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `Qwen3.5-4B-Q8_0` | Default summary model |
| `EXA_API_KEY` | — | Optional. Web search for terminology (proper-noun spelling) |
| `VLM_ENABLED` | `1` | Enable VLM fallback for silent videos |
| `YT_COMMENTS_ENABLED` | `1` | Fetch + summarise top YouTube comments (4th "Community Reaction" embed) |
| `YT_COMMENTS_MAX` | `100` | Cap on comments yt-dlp pulls per video (top + replies) |
| `YT_COMMENT_MIN_CHARS` | `40` | Drop comments shorter than this as noise (lol, first, emoji-only) |
| `YT_COMMENT_SUMMARY_TOP_N` | `30` | After filter+rank, top-N comments fed to the LLM |
| `BRIEF_SENTENCES` | `3-5` | Length of the brief TL;DW paragraph (videos) |
| `REDDIT_BRIEF_SENTENCES` | `4-6` | Length of the brief on Reddit/HN posts |
| `WEB_BRIEF_SENTENCES` | `3-5` | Length of the brief on plain web articles |
| `CHAPTERS_TARGET` | `4-10` | Target number of chapters per video |
| `CHAPTERS_MAX` | `15` | Hard upper bound on chapter count (anti-overchaptering) |
| `CHAPTERS_STATIC_TARGET` | `2-5` | Target chapters for static-shot content (music videos, ASMR) |
| `CHAPTER_HEADING_WORDS` | `3-7` | Word count for chapter headings |
| `CHAPTER_BODY_SENTENCES` | `1-2` | Sentence count per chapter body |
| `YT_COMMENTS_SENTENCES` | `4-7` | Length of the Community Reaction summary |
| `SECTIONS_BODY_SENTENCES` | `2-3` | Sentence count per section (web sections embed) |
| `REDDIT_TOP_COMMENTS` | `10` | Top-N Reddit comments to summarise per post |
| `REDDIT_REPLY_DEPTH` | `1` | Reddit reply tree depth |
| `HN_TOP_COMMENTS` | `10` | Top-N HackerNews comments to summarise per post |
| `HN_REPLY_DEPTH` | `1` | HackerNews reply tree depth |
| `LITMUS_SKIP_LLM_BELOW` | `2` | Litmus aggregate score below this skips the LLM qualitative read (clearly clean) |
| `LITMUS_SKIP_LLM_ABOVE` | `8` | Above this skips the LLM (clearly LLM-style; signals already strong) |
| `LITMUS_EXCERPT_CHARS` | `8000` | Hard cap on article excerpt sent to the litmus LLM call |
| `WAYBACK_TIMEOUT` | `8` | Seconds to wait on archive.org's /available endpoint |
| `MAX_JOBS_PER_USER_PER_HOUR` | `20` | Per-user sliding-window rate limit |
| `MAX_QUEUE_SIZE` | `40` | Global queue cap |
| `JOB_POLL_INTERVAL` | `3` | Seconds between `/api/jobs/{id}` polls while a transcription runs |
| `IMAGE_MAX_ATTACHMENTS` | `4` | Max image attachments per `tldr` reply |
| `IMAGE_MAX_BYTES_PER_ATTACHMENT` | `33554432` | Per-attachment byte cap the bot will forward (32MB) |
| `IMAGE_API_TIMEOUT` | `180` | Per-image timeout on the `/api/image` call (OCR is fast; VLM is the long pole) |
| `IMAGE_OCR_VERBATIM_MIN_CHARS` | `80` | Minimum total OCR chars across attachments before the verbatim "Text in image" embed + key-points pass fire |

#### Voice-channel live transcription (Discord bot)
| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_TRANSCRIBE_ENABLED` | — | Set to `1` to enable `/transcribe-join` + `/transcribe-leave`. Unset = feature off (commands not registered). |
| `VOICE_TRANSCRIPT_CHANNEL_ID` | `0` | Channel ID where per-call transcript threads are created. `0`/unset = the channel the command was invoked in. |
| `VOICE_MAX_SPEAKERS` | `4` | Max simultaneous per-speaker streams (each uses one whisper-live slot; capped by `LIVE_MAX_STREAMS`). |
| `VOICE_SPEAKER_IDLE_S` | `45` | Close a speaker's stream after this many seconds of silence, freeing a slot (re-opens on next speech). |
| `VOICE_MAX_RECONNECTS` | `5` | Consecutive whisper-live reconnect attempts (exp. backoff) before a speaker's stream is abandoned; it re-opens fresh on their next utterance. |
| `VOICE_SEND_INTERVAL` | `0.15` | How often buffered PCM is flushed to whisper-live (lower = snappier, more requests). |
| `VOICE_MAX_SILENCE_S` | `2.0` | Cap on reconstructed silence injected for a pause (gives whisper-live its end-of-utterance boundary). |
| `VOICE_TRANSCRIPT_RETENTION_DAYS` | `7` | Auto-purge deletes per-call transcript threads older than this. `0` disables the background purge (also off unless `VOICE_TRANSCRIPT_CHANNEL_ID` is set). |
| `VOICE_CLEANUP_INTERVAL_H` | `6` | How often the auto-purge loop runs (hours; floored at 5 min). |

Latency is also governed by the whisper-live streaming knobs (`LIVE_PROCESS_INTERVAL`, `LIVE_MIN_CHUNK_S`, `LIVE_TAIL_SILENCE_S` — see the Whisper service table); these are tuned low in `compose.yaml` for snappy live output.

See `.env.example` and `bot/.env.example` for the full list.

## Architecture

```
                       ┌─ React SPA (:7860/)
[whisper service]──────┤                                          GPU (~32GB)
  whisperX + alignment │
  + diarization +      ├─ HTTP API (:7860/api/*)
  yt-dlp + ffmpeg      │     │
                       │     └─ /api/jobs ─── enqueue ──┐
                       │                                ▼
                       │                          [valkey] ←─── queue: jobs + transcript cache
                       │                                ▲
                       │     ┌─ async worker pool ──────┘
                       │     │     BLPOP → run → store result
                       └─ /api/describe ──┐
                                          │
                                          ▼  HTTP
                              [model_proxy] (llm-compose, external network)
                                          │
                                          ▼
                              [Discord bot] ──── outbound websocket ──→ Discord
                                          ▲
                       ┌──────────────────┘
[crawl4ai] (Playwright)│  /md
[flaresolverr]         ┘  CF-challenge fallback
```

Job queue: all consumers (Discord bot, MCP server, web SPA, ad-hoc curl)
submit through `POST /api/jobs` and poll `GET /api/jobs/{id}`. The
async worker pool inside the whisper container `BLPOP`s from Valkey,
runs whisperx, writes the result back. Persistent (AOF) so a crash
mid-transcription doesn't lose state — recovered jobs run first on
restart. Transcript cache (sha1 of file content + decode settings) means
re-summarising the same video skips whisper entirely.

The bot is the only component talking to Discord. Discord connections are
**outbound only** — no port forwarding, public IP, or DNS needed. Other Discord
users reach the bot through Discord's gateway, which pushes events down the
already-open websocket.

Whisper + LLM together fit in 32GB VRAM (turbo ~6GB + LLM ~16-20GB +
alignment ~360MB). Crawl4AI and FlareSolverr each run their own Chromium
(~400MB image each, no GPU).

## Docker Compose services

| Service | Image | Purpose |
|---------|-------|---------|
| `whisper` | built locally | Transcription + VLM frame description |
| `whisper-live` | built locally | Low-latency streaming ASR (SPA Live tab — mic or system/OBS — + `live-tap/` CLI + Discord voice bot) |
| `bot` | built locally | Discord interface |
| `valkey` | `valkey/valkey:9-alpine` | Job queue + transcript cache (AOF-persisted) |
| `crawl4ai` | `unclecode/crawl4ai:0.7.4` | Article scraper (readability → Markdown) |
| `flaresolverr` | `ghcr.io/flaresolverr/flaresolverr:v3.4.6` | Cloudflare-challenge fallback |

Plus a standalone CLI (not a service): [`live-tap/`](live-tap/README.md) —
`desktop_tap.py` captures OBS / Windows-sink / mic audio and prints the live
transcript to stdout, so you can pipe it into any LLM / research step without
running the Discord bot. It only depends on the `whisper-live` `/ws-stream`
contract.

## Documentation

| Doc | Contents |
|-----|----------|
| [live-tap/README.md](live-tap/README.md) | Standalone audio → transcript tap (OBS / desktop / mic → stdout), virtual-cable routing, `make live-tap` |
| [docs/discord-bot-guide.md](docs/discord-bot-guide.md) | Full Discord bot guide — setup, every interaction path, slash commands, voice-channel live transcription, per-user/channel/server config, rate limits, diagnostics |
| [docs/design/multilingual.md](docs/design/multilingual.md) | Multilingual design — language ID pre-pass + translate-to-English flow |
| [docs/design/global-queue.md](docs/design/global-queue.md) | Global job-queue design notes |
| [docs/plans/](docs/plans/) | Implementation plans (live transcription, Discord voice transcription) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution + local-dev notes |
| [AGENTS.md](AGENTS.md) | Build/deploy/test cheat-sheet (Makefile targets, conventions, footguns) |
