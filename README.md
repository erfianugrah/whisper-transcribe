# whisper-transcribe

GPU-accelerated transcription + Discord TL;DW bot. Four content flows:

1. **Videos with speech** → whisperX (faster-whisper + wav2vec2 alignment + pyannote diarization) → LLM summary, plus a 4th *Community Reaction* embed pulling top YouTube comments
2. **Videos without speech** (music videos, silent gameplay, ASMR) → frame extraction + VLM descriptions → LLM summary
3. **Web articles** → Crawl4AI / FlareSolverr → LLM summary (triggered by replying `tldr` or `summarize` to a Discord message containing a URL). Reddit + HackerNews URLs get a structured fetch (post + linked article + top comments).
4. **AI litmus test** → regex stylistic scan + Wayback / AdSense / author-byline metadata + ambiguous-case LLM qualitative read → forensic signals report (no verdict). Triggered by replying `litmus` to a URL message.

Gradio UI + HTTP API for the whisper service; Discord bot for hands-off summarisation.

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

The bot defaults to `llama3.1` via Ollama on the host; change `LLM_MODEL` /
`LLM_API_URL` in `bot/.env` to point at any OpenAI-compatible service
(see `.env.example` for examples covering vLLM, llama.cpp server, OpenAI,
and llm-compose).

### Running alongside [llm-compose](https://github.com/erfianugrah/llm-compose)

If you already run llm-compose on the same host (for hot-swapping local
GGUFs), bring it up first, then layer in the overlay so this stack joins
its `model_proxy`:

```bash
docker compose -f compose.yaml -f compose.llm-compose.yaml up -d
```

Defaults in the overlay point `LLM_API_URL` at `http://model_proxy:11434/v1`
and pick reasonable `LLM_MODEL` / `LLM_VISION_MODEL` presets that exist in
llm-compose. Override anything you want in `.env`.

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
- **`litmus` reply trigger** → AI-litmus forensic report (LLM-tic phrases, em-dash density, Wayback domain age, AdSense, author byline; LLM qualitative read on ambiguous middle range)
- **Silent-video fallback** — VLM frame descriptions when speech density is low
- **Slash commands**: `/summarize`, `/transcribe`, `/status`, `/find`, `/config`, `/serverconfig`
- **Per-channel + per-guild config**: model override, VLM toggle, YT-comments toggle, diarize default, summary archive channel
- **Speaker rename**: button on diarized summaries lets users label SPEAKER_xx → real names
- **Rate limiting**: per-user sliding window + global queue cap

## HTTP API (whisper service)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | GPU info, service status, VLM availability |
| `/api/yt-download` | POST | Download audio (and optionally video) from any yt-dlp-supported URL |
| `/api/transcribe` | POST | Transcribe a local file |
| `/api/describe` | POST | Extract frames + describe via VLM (silent-video fallback) |
| `/api/cleanup` | POST | Remove a yt-dlp temp file (best-effort) |

### POST /api/yt-download

```json
{"url": "https://youtube.com/watch?v=...", "playlist": false}
```

Returns: `{"filename": "...", "title": "...", "duration": 123}`

### POST /api/transcribe

```json
{
  "file_path": "/tmp/yt-dlp-xxx/id.wav",
  "model": "turbo",
  "language": "Auto-detect",
  "format": "txt",
  "diarize": false,
  "cleanup": true
}
```

Returns: `{"status": "Done -- ...", "transcript": "...", "subtitle_file": "..."}`

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

See `.env.example` and `bot/.env.example` for the full list.

## Architecture

```
                       ┌─ Gradio UI (:7860)
[whisper service]──────┤                                          GPU (~32GB)
  whisperX + alignment │
  + diarization +      ├─ HTTP API (:7860/api/*)
  yt-dlp + ffmpeg      │
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
| `bot` | built locally | Discord interface |
| `crawl4ai` | `unclecode/crawl4ai:0.7.4` | Article scraper (readability → Markdown) |
| `flaresolverr` | `ghcr.io/flaresolverr/flaresolverr:v3.4.6` | Cloudflare-challenge fallback |
