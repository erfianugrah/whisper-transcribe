# whisper-transcribe

GPU-accelerated transcription service using [whisperX](https://github.com/m-bain/whisperX) (faster-whisper + wav2vec2 alignment + pyannote diarization). Gradio UI + HTTP API for programmatic access.

## Quick Start

```bash
# Requires: NVIDIA GPU, Docker with nvidia-container-toolkit
cp .env.example .env  # Set HF_TOKEN for diarization
docker compose up -d --build
# UI: http://localhost:7860
# API: http://localhost:7860/api/status
```

## Features

- **WhisperX pipeline**: faster-whisper transcription → wav2vec2 word alignment → pyannote speaker diarization
- **Models**: tiny, base, small, medium, large (v3), turbo (large-v3-turbo)
- **Output formats**: txt, srt, vtt, json (with word-level timestamps)
- **Speaker diarization**: identifies who is speaking (requires HF_TOKEN)
- **YouTube download**: yt-dlp integration for direct URL transcription
- **Idle VRAM management**: models auto-unload after 5min idle (configurable via MODEL_IDLE_TIMEOUT)
- **MCP integration**: tools for OpenCode LLM workflow (download → transcribe → summarize)

## HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | GPU info, service status |
| `/api/yt-download` | POST | Download audio from YouTube URL |
| `/api/transcribe` | POST | Transcribe a local file |

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

## MCP Server

The companion MCP server lives at `~/llm-compose/mcp/whisper-server.py` and is registered globally in `~/.config/opencode/opencode.json`.

**Tools:**
- `whisper_status` — service health check
- `yt_download` — download YouTube audio
- `whisper_transcribe` — transcribe local file
- `yt_transcribe` — download + transcribe (one-shot for summaries)
- `yt_transcribe_playlist` — process entire playlists

**Usage from OpenCode:**
> "Transcribe and summarize this video: https://youtube.com/watch?v=..."

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token for pyannote diarization models |
| `DEBUG_MODE` | `1` | Verbose logging |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds before unloading models from VRAM |

## Architecture

```
Gradio UI (:7860)          HTTP API (:7860/api/*)
       \                        /
        ├─ whisperX (GPU, ~6GB VRAM for turbo)
        ├─ wav2vec2 alignment
        ├─ pyannote diarization
        └─ yt-dlp (downloads to /tmp, auto-cleanup)
```

Runs independently alongside llm-compose. Both fit in 32GB VRAM (turbo ~6GB + LLM model ~16-20GB).

## Docker Compose

```yaml
services:
  whisper:
    build: .
    ports: ["7860:7860"]
    volumes:
      - ./uploads:/data
      - ./app.py:/app/app.py:ro        # Live reload without rebuild
      - model-cache:/root/.cache/huggingface
      - /mnt/d/Videos:/media:ro         # Local media for UI dropdown
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
```
