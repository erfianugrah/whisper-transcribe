# Live Transcription — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `whisper-live` sidecar service (faster-whisper streaming) so the Discord bot can transcribe currently-airing live streams (summary posted when the stream ends), and the Gradio UI can transcribe microphone input in real time.

**Architecture:** A new `whisper-live/` directory holds a self-contained Starlette service (port 7861). **All media plumbing (yt-dlp + ffmpeg) lives in this service** — the bot container is `python:3.13-slim` with no yt-dlp/ffmpeg and delegates everything over HTTP/WebSocket, exactly like it already delegates downloads to the `whisper` service. The bot sends a URL to whisper-live and receives transcript segments. Live-stream detection re-uses the existing `NotAVideoError` re-route idiom: a new `IsLiveError` raised at the top of `process()` flips `job.kind` to `"live"` and the worker re-dispatches to `process_live()`. The existing `whisper` (WhisperX batch) service is unchanged except for a new Gradio Live tab.

**Tech Stack:** faster-whisper (CTranslate2), starlette + uvicorn, yt-dlp + ffmpeg (in the whisper-live container only), aiohttp WebSocket client (bot, already present), numpy.

---

## Architecture rationale (read before starting)

Three corrections over the naive design, each load-bearing:

1. **The bot has no yt-dlp/ffmpeg.** `bot/Dockerfile` is `python:3.13-slim`; the bot never shells out — it POSTs to `whisper`'s `/api/yt-download`. So the `yt-dlp | ffmpeg → PCM` pipeline for live streams runs **inside whisper-live**, not the bot. The bot is a thin WebSocket client.

2. **Live streams must be detected before any download.** A live stream download never terminates. We cannot route to the normal video pipeline (`/api/yt-download` would hang for hours). Detection happens at the top of `process()`'s cache-miss branch via an HTTP probe to whisper-live; on a hit it raises `IsLiveError`, which the worker catches (mirroring `NotAVideoError`) and re-dispatches as `kind="live"`. This single interception covers ALL job sources (auto-paste, reply-trigger, slash commands) because they all funnel through `process()`.

3. **Live transcripts have no speaker labels.** `process_live` posts a plain embed with no `SpeakerRenameView` (that view requires diarization, which streaming mode doesn't do).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `whisper-live/transcriber.py` | faster-whisper wrapper; resident model; async chunk inference |
| Create | `whisper-live/server.py` | Starlette: `/health`, `/probe`, `/transcribe-chunk`, `WS /ws-url` |
| Create | `whisper-live/requirements.txt` | pip deps (faster-whisper, starlette, uvicorn, numpy, yt-dlp) |
| Create | `whisper-live/Dockerfile` | GPU image with ffmpeg + yt-dlp |
| Modify | `compose.yaml` | Add `whisper-live` service; wire env vars into bot + whisper |
| Modify | `app.py` | Add "Live" mic tab (uses `POST /transcribe-chunk`); add `LIVE_SERVICE_URL` |
| Modify | `bot/main.py` | `IsLiveError`, `_is_live_stream()` (HTTP probe), `process_live()` (WS client), `kind="live"`, emoji, worker re-route |
| Modify | `tests/test_regression.py` | Regression tests for new bot-side behaviour |

---

## Task 1: StreamingTranscriber

**Files:**
- Create: `whisper-live/transcriber.py`
- Test: `whisper-live/test_transcriber.py`

- [ ] **Step 1: Write the failing unit test**

Create `whisper-live/test_transcriber.py`:

```python
"""Unit tests for StreamingTranscriber. Stubs faster_whisper so no GPU needed."""
import asyncio
import types
import sys

# ── stub faster_whisper ────────────────────────────────────────────────────────
_seg = types.SimpleNamespace(text=" hello world", start=0.0, end=1.5)
_info = types.SimpleNamespace(language="en", duration=1.5)

fw = types.ModuleType("faster_whisper")

class _FakeModel:
    def __init__(self, *a, **k): pass
    def transcribe(self, audio, **k):
        return [_seg], _info

fw.WhisperModel = _FakeModel
sys.modules["faster_whisper"] = fw
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
from transcriber import StreamingTranscriber


def _sine_pcm(seconds: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    arr = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return arr.tobytes()


def test_transcribe_chunk_returns_segments():
    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._model = _FakeModel()
    tr._lock = asyncio.Lock()
    segs = asyncio.run(tr.transcribe_chunk(_sine_pcm(1.0), context=""))
    assert isinstance(segs, list)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello world"
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.5


def test_transcribe_chunk_strips_empty_segments():
    empty_seg = types.SimpleNamespace(text="   ", start=0.0, end=0.5)

    class _EmptyModel:
        def transcribe(self, audio, **k):
            return [empty_seg], _info

    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._lock = asyncio.Lock()
    tr._model = _EmptyModel()
    segs = asyncio.run(tr.transcribe_chunk(_sine_pcm(), context=""))
    assert segs == []


def test_transcribe_chunk_passes_context_as_initial_prompt():
    calls = {}

    class _TrackModel:
        def transcribe(self, audio, **k):
            calls.update(k)
            return [_seg], _info

    tr = StreamingTranscriber.__new__(StreamingTranscriber)
    tr._lock = asyncio.Lock()
    tr._model = _TrackModel()
    asyncio.run(tr.transcribe_chunk(_sine_pcm(), context="previous text"))
    assert calls.get("initial_prompt") == "previous text"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd whisper-live && python test_transcriber.py
```
Expected: `ModuleNotFoundError: No module named 'transcriber'`

- [ ] **Step 3: Implement `whisper-live/transcriber.py`**

```python
"""Resident faster-whisper model for streaming transcription.

Keeps the model loaded permanently (the live service is dedicated; VRAM is
not shared with the batch service). Serialises GPU calls via asyncio.Lock so
multiple concurrent sessions share one model safely.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np

SAMPLE_RATE = 16000  # expected input: 16 kHz mono int16 PCM


class StreamingTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._lock = asyncio.Lock()

    async def transcribe_chunk(self, pcm_bytes: bytes, context: str = "") -> list[dict]:
        """Transcribe a raw PCM chunk (16 kHz mono int16).

        Returns [{"text", "start", "end"}] with empty segments filtered.
        Serialises on self._lock so concurrent callers queue rather than
        racing on the GPU.
        """
        loop = asyncio.get_event_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._transcribe_sync, pcm_bytes, context
            )

    def _transcribe_sync(self, pcm_bytes: bytes, context: str) -> list[dict]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        kwargs: dict = dict(
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        if context:
            kwargs["initial_prompt"] = context
        segments, _info = self._model.transcribe(audio, **kwargs)
        return [
            {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
            for seg in segments
            if seg.text.strip()
        ]
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd whisper-live && python test_transcriber.py
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add whisper-live/transcriber.py whisper-live/test_transcriber.py
git commit -m "whisper-live: add StreamingTranscriber (faster-whisper, resident model)"
```

---

## Task 2: Starlette server (probe + chunk + URL WebSocket)

**Files:**
- Create: `whisper-live/server.py`

The server owns all media plumbing:
- `GET /health` — liveness + capacity
- `GET /probe?url=` — yt-dlp metadata probe → `{"is_live": bool, "title": str}` (used by the bot to decide routing)
- `POST /transcribe-chunk` — stateless PCM inference (used by the Gradio mic tab)
- `WS /ws-url` — client sends `{"url": "..."}`; server runs `yt-dlp | ffmpeg → PCM`, streams segment JSON, sends `{"type":"done","transcript":...}` when the stream ends (used by the bot)

- [ ] **Step 1: Write the failing unit test**

Append to `whisper-live/test_transcriber.py`:

```python
# ── server integration tests ───────────────────────────────────────────────────
import os
os.environ.setdefault("LIVE_MODEL", "stub")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("LIVE_MAX_STREAMS", "2")


def _make_app():
    import server as srv
    stub_tr = StreamingTranscriber.__new__(StreamingTranscriber)
    stub_tr._lock = asyncio.Lock()
    stub_tr._model = _FakeModel()
    srv._transcriber = stub_tr
    return srv.app


def test_health_endpoint_returns_ok():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "active_streams" in body


def test_transcribe_chunk_endpoint():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.post(
        "/transcribe-chunk",
        content=_sine_pcm(1.0),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["text"] == "hello world"


def test_transcribe_chunk_rejects_empty_body():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.post(
        "/transcribe-chunk",
        content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 400


def test_probe_rejects_missing_url():
    from starlette.testclient import TestClient
    client = TestClient(_make_app())
    resp = client.get("/probe")
    assert resp.status_code == 400


def test_ws_url_route_registered():
    """The /ws-url WebSocket route must be registered (full behaviour is
    covered by the Task 8 smoke test — subprocess can't be unit-tested here)."""
    app = _make_app()
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ws-url" in paths
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd whisper-live && python test_transcriber.py
```
Expected: `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Implement `whisper-live/server.py`**

```python
"""whisper-live streaming server.

Endpoints
---------
GET  /health             — liveness + capacity snapshot
GET  /probe?url=<url>     — yt-dlp metadata probe → {"is_live", "title"}
POST /transcribe-chunk    — stateless PCM inference (Gradio mic tab)
                            Body: raw 16 kHz mono int16 PCM. Query: context=<str>
                            Returns: {"segments": [{"text","start","end"}]}
WS   /ws-url              — client sends {"url": "..."} (text); server runs
                            yt-dlp|ffmpeg → PCM, streams {"type":"segment",...}
                            JSON lines, then {"type":"done","transcript":"..."}
                            when the live stream ends. Used by the Discord bot.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from transcriber import StreamingTranscriber

log = logging.getLogger("whisper-live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("LIVE_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
LIVE_MAX_STREAMS = int(os.environ.get("LIVE_MAX_STREAMS", "4"))
PORT = int(os.environ.get("LIVE_PORT", "7861"))
# Inference cadence: accumulate this many seconds of audio before a pass.
CHUNK_SECONDS = int(os.environ.get("LIVE_CHUNK_SECONDS", "10"))
CHUNK_THRESHOLD = CHUNK_SECONDS * 16000 * 2  # bytes of 16 kHz int16

_transcriber: StreamingTranscriber | None = None
_active_streams: int = 0


async def startup() -> None:
    global _transcriber
    log.info(f"Loading model {MODEL_NAME!r} on {DEVICE} ({COMPUTE_TYPE})…")
    _transcriber = StreamingTranscriber(MODEL_NAME, DEVICE, COMPUTE_TYPE)
    log.info(f"Ready — max concurrent streams: {LIVE_MAX_STREAMS}")


# ── HTTP routes ───────────────────────────────────────────────────────────────
async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model": MODEL_NAME,
        "max_streams": LIVE_MAX_STREAMS,
        "active_streams": _active_streams,
    })


async def probe(request: Request) -> JSONResponse:
    """yt-dlp metadata probe. Returns is_live + title without downloading."""
    url = request.query_params.get("url", "")
    if not url:
        return JSONResponse({"error": "missing url"}, status_code=400)
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "--no-download", "--dump-json", "--no-playlist",
            "--socket-timeout", "15", url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if not stdout:
            return JSONResponse({"is_live": False, "title": ""})
        meta = json.loads(stdout.decode())
        return JSONResponse({
            "is_live": meta.get("live_status") == "is_live",
            "title": meta.get("title") or "",
        })
    except Exception as e:
        log.warning(f"probe failed for {url!r}: {e}")
        return JSONResponse({"is_live": False, "title": ""})


async def transcribe_chunk(request: Request) -> JSONResponse:
    """Stateless HTTP endpoint for the Gradio mic tab."""
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)
    context = request.query_params.get("context", "")
    try:
        segments = await _transcriber.transcribe_chunk(body, context=context)
    except Exception as e:
        log.error(f"transcribe_chunk error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"segments": segments})


# ── WebSocket: URL → live transcript ────────────────────────────────────────────
async def ws_url_endpoint(websocket: WebSocket) -> None:
    global _active_streams
    await websocket.accept()

    if _active_streams >= LIVE_MAX_STREAMS:
        await websocket.send_text(json.dumps({"type": "error", "message": "server at capacity"}))
        await websocket.close(1013)
        return

    try:
        first = await websocket.receive_text()
        url = json.loads(first).get("url", "")
    except Exception:
        await websocket.send_text(json.dumps({"type": "error", "message": "expected {'url': ...} first message"}))
        await websocket.close(1003)
        return
    if not url:
        await websocket.send_text(json.dumps({"type": "error", "message": "empty url"}))
        await websocket.close(1003)
        return

    _active_streams += 1
    log.info(f"[ws-url] stream opened ({_active_streams}/{LIVE_MAX_STREAMS}): {url}")

    shell_cmd = (
        f"yt-dlp -f bestaudio --no-part -q -o - {shlex.quote(url)} "
        f"| ffmpeg -i pipe:0 -f s16le -ar 16000 -ac 1 -loglevel quiet pipe:1"
    )
    proc = await asyncio.create_subprocess_shell(
        shell_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )

    buffer = bytearray()
    parts: list[str] = []
    try:
        while True:
            chunk = await proc.stdout.read(CHUNK_THRESHOLD)
            if not chunk:
                break  # yt-dlp exited → live stream ended
            buffer.extend(chunk)
            if len(buffer) < CHUNK_THRESHOLD:
                continue
            context = " ".join(parts[-5:])
            segments = await _transcriber.transcribe_chunk(bytes(buffer), context)
            buffer.clear()
            for seg in segments:
                parts.append(seg["text"])
                await websocket.send_text(json.dumps({"type": "segment", **seg}))

        # Flush tail (>0.5 s remaining).
        if len(buffer) > 16000:
            segments = await _transcriber.transcribe_chunk(bytes(buffer), " ".join(parts[-5:]))
            for seg in segments:
                parts.append(seg["text"])
                await websocket.send_text(json.dumps({"type": "segment", **seg}))

        await websocket.send_text(json.dumps({"type": "done", "transcript": " ".join(parts)}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"[ws-url] error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        _active_streams -= 1
        log.info(f"[ws-url] stream closed ({_active_streams}/{LIVE_MAX_STREAMS})")


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/probe", probe),
        Route("/transcribe-chunk", transcribe_chunk, methods=["POST"]),
        WebSocketRoute("/ws-url", ws_url_endpoint),
    ],
    on_startup=[startup],
)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")
```

- [ ] **Step 4: Run tests — all should pass**

```bash
cd whisper-live && python test_transcriber.py
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add whisper-live/server.py
git commit -m "whisper-live: add Starlette server (probe + chunk + URL WebSocket)"
```

---

## Task 3: Dockerfile + requirements

**Files:**
- Create: `whisper-live/requirements.txt`
- Create: `whisper-live/Dockerfile`

- [ ] **Step 1: Create `whisper-live/requirements.txt`**

```
faster-whisper>=1.1.0
starlette>=0.41.0
uvicorn[standard]>=0.32.0
numpy>=1.26.0
yt-dlp>=2024.11.4
```

`yt-dlp` ships as a pip package — installing it here keeps the live service self-contained (it owns both the probe and the stream pipeline). `ffmpeg` is an apt package, installed in the Dockerfile below.

- [ ] **Step 2: Create `whisper-live/Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY transcriber.py server.py ./

EXPOSE 7861

CMD ["python3", "server.py"]
```

- [ ] **Step 3: Verify the image builds (no GPU needed for build)**

```bash
docker build -t whisper-live:local whisper-live/
```
Expected: build completes; confirm both tools are present:
```bash
docker run --rm whisper-live:local sh -c "yt-dlp --version && ffmpeg -version | head -1"
```
Expected: prints a yt-dlp version and an ffmpeg version line.

- [ ] **Step 4: Commit**

```bash
git add whisper-live/requirements.txt whisper-live/Dockerfile
git commit -m "whisper-live: add Dockerfile (ffmpeg + yt-dlp) and requirements"
```

---

## Task 4: compose.yaml — add whisper-live service

**Files:**
- Modify: `compose.yaml`

- [ ] **Step 1: Add the service block after the `whisper:` service, before `crawl4ai:`**

```yaml
  # ─── Live transcription sidecar ──────────────────────────────────────────
  # faster-whisper streaming service. Owns its own yt-dlp + ffmpeg (the bot
  # is python:3.13-slim and has neither — it delegates here over HTTP/WS just
  # as it delegates batch downloads to the `whisper` service). Model stays
  # resident so the first live job has no cold-load. Shares the GPU with
  # `whisper`; on the 5090 (32 GB) raise LIVE_MAX_STREAMS freely.
  whisper-live:
    image: erfianugrah/whisper-live:latest
    build:
      context: ./whisper-live
    pull_policy: missing
    environment:
      - LIVE_MODEL=${LIVE_MODEL:-large-v3-turbo}
      - DEVICE=cuda
      - COMPUTE_TYPE=${LIVE_COMPUTE_TYPE:-float16}
      - LIVE_MAX_STREAMS=${LIVE_MAX_STREAMS:-4}
      - LIVE_PORT=7861
      - LIVE_CHUNK_SECONDS=${LIVE_CHUNK_SECONDS:-10}
    ports:
      - "7861:7861"
    networks:
      - default
    volumes:
      - live-model-cache:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "python3", "-c",
             "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:7861/health',timeout=3).status==200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 120s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

- [ ] **Step 2: Add `LIVE_SERVICE_URL` to the `whisper:` service environment block**

In the `whisper:` service `environment:` list, add:
```yaml
      # whisper-live sidecar — used only by the Gradio Live tab (mic streaming).
      - LIVE_SERVICE_URL=${LIVE_SERVICE_URL:-http://whisper-live:7861}
```

- [ ] **Step 3: Add `WHISPER_LIVE_URL` to the `bot:` service environment block**

In the `bot:` service `environment:` list (near `WHISPER_API_URL`), add:
```yaml
      - WHISPER_LIVE_URL=http://whisper-live:7861
```

- [ ] **Step 4: Add whisper-live to the `bot:` `depends_on:` block**

```yaml
      whisper-live:
        condition: service_healthy
```

- [ ] **Step 5: Add the named volume**

In the `volumes:` section at the bottom of compose.yaml, add:
```yaml
  live-model-cache:  # faster-whisper HuggingFace model downloads
```

- [ ] **Step 6: Verify compose parses cleanly**

```bash
docker compose config --quiet
```
Expected: exits 0, no errors.

- [ ] **Step 7: Commit**

```bash
git add compose.yaml
git commit -m "compose: add whisper-live sidecar; wire env into bot + whisper"
```

---

## Task 5: Gradio "Live" mic tab in `app.py`

**Files:**
- Modify: `app.py`

Uses `POST /transcribe-chunk` (stateless), not the WebSocket. Gradio's stream callback is stateless-friendly, so HTTP is simpler than holding a WS open.

- [ ] **Step 1: Add `LIVE_SERVICE_URL` env var near the other service URL vars (around `LLM_VISION_API_URL`)**

```python
LIVE_SERVICE_URL = os.environ.get("LIVE_SERVICE_URL", "http://localhost:7861")
```

- [ ] **Step 2: Add the Live tab inside the existing `gr.Tabs()` block, after the `"YouTube"` tab**

```python
        with gr.Tab("Live"):
            gr.Markdown(
                "Transcribe from microphone in real time. Click **Record**, "
                "speak, and the transcript accumulates below. **Clear** resets it."
            )
            live_audio_input = gr.Audio(
                streaming=True,
                sources=["microphone"],
                label="Microphone",
            )
            live_transcript_box = gr.Textbox(
                label="Live Transcript",
                lines=12,
                interactive=False,
                placeholder="Transcript will appear here as you speak…",
            )
            live_state = gr.State({"buffer": b"", "transcript": ""})
            live_clear_btn = gr.Button("Clear", variant="secondary")
```

- [ ] **Step 3: Add the stream handler inside the `with gr.Blocks` block (before event wiring)**

```python
    def _live_chunk(audio_chunk, state):
        """Per-mic-chunk callback. Accumulates PCM in state; flushes to
        whisper-live every ~10 s (matches server CHUNK_SECONDS). Uses stdlib
        urllib (no `requests` dependency — not in the whisper image).
        Gradio runs handlers in a thread pool, so a blocking call is fine."""
        import json as _json
        import urllib.parse
        import urllib.request

        if audio_chunk is None:
            return state["transcript"], state
        sr, arr = audio_chunk
        if arr.dtype != np.int16:
            arr = (arr.clip(-1.0, 1.0) * 32767).astype(np.int16)
        if sr != 16000:
            from scipy.signal import resample_poly
            arr = resample_poly(arr, 16000, sr).astype(np.int16)
        state["buffer"] += arr.tobytes()

        THRESHOLD = 16000 * 2 * 10  # 10 s of 16 kHz int16
        if len(state["buffer"]) < THRESHOLD:
            return state["transcript"], state

        try:
            qs = urllib.parse.urlencode({"context": state["transcript"][-300:]})
            req = urllib.request.Request(
                f"{LIVE_SERVICE_URL}/transcribe-chunk?{qs}",
                data=bytes(state["buffer"]),
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = _json.loads(resp.read().decode())
            for seg in payload.get("segments", []):
                state["transcript"] += seg["text"] + " "
        except Exception as exc:
            log.warning(f"[live] chunk POST failed: {exc}")
        state["buffer"] = b""
        return state["transcript"], state
```

> **Dep note:** `scipy` is confirmed present in the whisper image (used at `app.py:453`). No `requests` dependency — the handler uses stdlib `urllib`.

- [ ] **Step 4: Wire events (after the clear button is defined)**

```python
    live_audio_input.stream(
        fn=_live_chunk,
        inputs=[live_audio_input, live_state],
        outputs=[live_transcript_box, live_state],
        show_progress=False,
    )
    live_clear_btn.click(
        fn=lambda: ("", {"buffer": b"", "transcript": ""}),
        outputs=[live_transcript_box, live_state],
    )
```

- [ ] **Step 5: Verify the UI builds at import time**

```bash
python3 -c "import app" 2>&1 | grep -iE "error|exception|traceback" | head -5
```
Expected: no output (clean import; the Gradio block builds at import).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "app: add Live mic tab in Gradio UI (POST /transcribe-chunk)"
```

---

## Task 6: Bot plumbing — IsLiveError, probe helper, env var, emoji

**Files:**
- Modify: `bot/main.py`

This task adds everything except `process_live` (Task 7). No behaviour changes for existing flows until `process_live` exists, so `make test` stays green throughout.

- [ ] **Step 1: Add `WHISPER_LIVE_URL` env var directly after `WHISPER_API`**

Find:
```python
WHISPER_API = os.environ.get("WHISPER_API_URL", "http://localhost:7860")
```
Add after:
```python
WHISPER_LIVE_URL = os.environ.get("WHISPER_LIVE_URL", "http://localhost:7861")
# WebSocket form derived from the HTTP URL (http→ws, https→wss).
WHISPER_LIVE_WS = WHISPER_LIVE_URL.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
```

- [ ] **Step 2: Add `PROCESSING_EMOJI_LIVE` and include it in the `PROCESSING_EMOJI` tuple**

Find:
```python
PROCESSING_EMOJI_VIDEO = "\U0001f3a7"  # 🎧
PROCESSING_EMOJI_WEB = "\U0001f4f0"    # 📰
PROCESSING_EMOJI_IMAGE = "\U0001f5bc\ufe0f"  # 🖼️ image OCR + VLM
```
Add a fourth constant:
```python
PROCESSING_EMOJI_LIVE = "\U0001f399\ufe0f"   # 🎙️ live transcription
```
Then find the `PROCESSING_EMOJI = (` tuple and add an entry inside it:
```python
    PROCESSING_EMOJI_LIVE,     # 🎙️ live transcription
```

- [ ] **Step 3: Accept `"live"` in the `Job.__post_init__` validator**

Find:
```python
        if self.kind not in ("video", "web", "litmus", "image"):
```
Replace with:
```python
        if self.kind not in ("video", "web", "litmus", "image", "live"):
```
And update the adjacent error-message string the same way:
```python
                f"Job.kind must be 'video' | 'web' | 'litmus' | 'image' | 'live', "
```

- [ ] **Step 4: Update the `_RetrySpec.kind` comment (cosmetic; the field already stores any string)**

Find:
```python
    kind: str                      # "video" | "web" | "litmus" | "image"
```
Replace with:
```python
    kind: str                      # "video" | "web" | "litmus" | "image" | "live"
```

- [ ] **Step 5: Add the `IsLiveError` exception class next to `NotAVideoError`**

Find the `NotAVideoError` class definition (around line 2706). Add immediately after it:
```python
class IsLiveError(Exception):
    """process() determined the URL is a currently-airing live stream.

    Caught by worker() (mirroring NotAVideoError) to re-route the job to
    kind='live' / process_live, which streams via whisper-live instead of
    downloading a never-terminating file."""
```

- [ ] **Step 6: Add the `_is_live_stream` HTTP-probe helper**

Add after the `_rate_limit_check` function (around line 660):
```python
async def _is_live_stream(url: str) -> bool:
    """Ask whisper-live whether `url` is a currently-airing live stream.

    Fails safe (returns False) on any error — if whisper-live is down or
    the probe errors, the job falls through to the normal video pipeline
    rather than getting stuck. Only currently-live streams return True;
    VOD'd streams ('was_live') route to the normal video pipeline."""
    if http is None:
        return False
    try:
        async with http.get(
            f"{WHISPER_LIVE_URL}/probe",
            params={"url": url},
            timeout=aiohttp.ClientTimeout(total=35),
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return bool(data.get("is_live"))
    except Exception:
        return False
```

- [ ] **Step 7: Insert the live probe at the top of `process()`'s cache-miss branch**

Find the cache-miss `else:` branch in `process()`:
```python
    else:
        # 2. Download. Keep the video stream alongside audio when VLM is
        # enabled — /api/describe needs a video file to extract frames
```
Insert the probe as the first statement in that `else` block, before the download comment:
```python
    else:
        # 1b. Live-stream gate. A live stream download never terminates, so
        # detect-and-reroute BEFORE the download. Cache hits skip this (a
        # cached transcript means the stream already finished as a VOD).
        # _is_live_stream fails safe to False, so a down whisper-live just
        # means we attempt the normal video path.
        if await _is_live_stream(job.url):
            raise IsLiveError(job.url)
        # 2. Download. Keep the video stream alongside audio when VLM is
        # enabled — /api/describe needs a video file to extract frames
```

- [ ] **Step 8: Catch `IsLiveError` in `worker()` and re-route to `kind="live"`**

Find the `except NotAVideoError as e:` block in `worker()`. Add a new `except` clause directly before it:
```python
            except IsLiveError as e:
                log.info("[%s] live stream — re-routing to live pipeline: %s",
                         job.video_id, e)
                job.kind = "live"
                # Routing change, not a retry — re-enter loop with new handler.
                continue
            except NotAVideoError as e:
```

- [ ] **Step 9: Add the `kind == "live"` dispatch in `worker()`'s handler selection**

Find:
```python
                elif job.kind == "image":
                    handler = process_image
                else:
                    handler = process
```
Replace with:
```python
                elif job.kind == "image":
                    handler = process_image
                elif job.kind == "live":
                    handler = process_live
                else:
                    handler = process
```

- [ ] **Step 10: Run the full regression suite — must stay green**

```bash
make test
```
Expected: all 361 existing tests pass. (`process_live` is referenced in the dispatch but never reached by any test path yet — Python resolves the name at call time, and `make test` imports the module, so define `process_live` in Task 7 before any live job runs. The import itself succeeds because the reference is inside a function body, not at module top level.)

> **Important:** Step 9 references `process_live`, which doesn't exist until Task 7. Module import still succeeds (the name is only looked up when a live job dispatches), so `make test` passes. But do **not** deploy between Task 6 and Task 7.

- [ ] **Step 11: Commit**

```bash
git add bot/main.py
git commit -m "bot: plumb live job kind — IsLiveError reroute, probe helper, emoji, env"
```

---

## Task 7: `process_live()` handler

**Files:**
- Modify: `bot/main.py`
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Write the failing regression tests**

Add to `tests/test_regression.py` (near `test_bot_job_has_translate_field`):
```python
def test_bot_is_live_error_defined():
    """IsLiveError must be a defined exception class."""
    assert "class IsLiveError(" in BOT_SRC


def test_bot_live_emoji_defined():
    """PROCESSING_EMOJI_LIVE defined and present in the PROCESSING_EMOJI tuple."""
    assert "PROCESSING_EMOJI_LIVE" in BOT_SRC
    tuple_src = BOT_SRC[BOT_SRC.index("PROCESSING_EMOJI = ("):][:500]
    assert "PROCESSING_EMOJI_LIVE" in tuple_src


def test_bot_validator_accepts_live_kind():
    """Job.__post_init__ must accept kind='live'."""
    idx = BOT_SRC.index("self.kind not in")
    assert '"live"' in BOT_SRC[idx:idx + 120]


def test_bot_is_live_stream_uses_probe_endpoint():
    """_is_live_stream must call whisper-live's /probe (not shell yt-dlp —
    the bot container has no yt-dlp/ffmpeg)."""
    assert "async def _is_live_stream(" in BOT_SRC
    fn_src = BOT_SRC[BOT_SRC.index("async def _is_live_stream("):][:900]
    assert "/probe" in fn_src
    assert "WHISPER_LIVE_URL" in fn_src
    # Must NOT shell out to yt-dlp from the bot.
    assert "create_subprocess" not in fn_src


def test_bot_process_uses_is_live_gate():
    """process() must raise IsLiveError when _is_live_stream is true."""
    proc_src = BOT_SRC[BOT_SRC.index("async def process(job: Job):"):]
    proc_src = proc_src[:proc_src.index("async def process_url")]
    assert "_is_live_stream" in proc_src
    assert "raise IsLiveError" in proc_src


def test_bot_worker_reroutes_is_live():
    """worker() must catch IsLiveError and set kind='live'."""
    worker_src = BOT_SRC[BOT_SRC.index("async def worker("):][:5000]
    assert "except IsLiveError" in worker_src
    assert 'job.kind = "live"' in worker_src


def test_bot_worker_dispatches_live_handler():
    """worker() must dispatch kind='live' to process_live."""
    worker_src = BOT_SRC[BOT_SRC.index("async def worker("):][:5000]
    assert 'job.kind == "live"' in worker_src
    assert "process_live" in worker_src


def test_bot_process_live_defined_and_uses_ws():
    """process_live must exist and connect to whisper-live via WebSocket."""
    assert "async def process_live(" in BOT_SRC
    fn_src = BOT_SRC[BOT_SRC.index("async def process_live("):]
    next_def = fn_src.index("\nasync def ", 1)
    fn_src = fn_src[:next_def]
    assert "ws_connect" in fn_src
    assert "WHISPER_LIVE_WS" in fn_src
    assert "summarize(" in fn_src
    # No SpeakerRenameView on live (no diarization in streaming mode).
    assert "SpeakerRenameView" not in fn_src


def test_bot_whisper_live_env_vars():
    """Both HTTP and WS forms of the whisper-live URL must be derived."""
    assert "WHISPER_LIVE_URL" in BOT_SRC
    assert "WHISPER_LIVE_WS" in BOT_SRC
```

- [ ] **Step 2: Run to confirm failures**

```bash
make test 2>&1 | grep -iE "fail|error" | head -20
```
Expected: the new `test_bot_process_live_defined_and_uses_ws` fails (`process_live` not defined); others may already pass from Task 6.

- [ ] **Step 3: Implement `process_live()` after `process_image()` (around line 4465)**

```python
async def process_live(job: Job):
    """Stream-transcribe a currently-airing live URL via whisper-live.

    The bot is a thin WebSocket client: it sends the URL to whisper-live's
    /ws-url, which runs yt-dlp|ffmpeg internally and streams back transcript
    segments. When the live stream ends (yt-dlp exits), whisper-live sends a
    'done' frame; we then summarise and post (same prompts as video jobs).
    Live streams have no speaker labels, so no SpeakerRenameView is attached.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    await _job_react(job, PROCESSING_EMOJI_LIVE)  # 🎙️
    _inflight_phase(job, PHASE_TRANSCRIBING)

    # 1. Title via probe (best-effort; the WS path doesn't return it).
    title = "Live Stream"
    try:
        async with http.get(
            f"{WHISPER_LIVE_URL}/probe", params={"url": job.url},
            timeout=aiohttp.ClientTimeout(total=35),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                title = data.get("title") or title
    except Exception:
        pass

    # 2. Stream: open WS, send URL, collect segments until 'done'.
    transcript_parts: list[str] = []
    stream_start = time.monotonic()
    async with http.ws_connect(f"{WHISPER_LIVE_WS}/ws-url", timeout=60) as ws:
        await ws.send_json({"url": job.url})
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "segment":
                    transcript_parts.append(data["text"])
                elif data["type"] == "done":
                    break
                elif data["type"] == "error":
                    raise RuntimeError(f"whisper-live: {data['message']}")
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break

    if not transcript_parts:
        raise PermanentError("No speech detected in live stream — nothing to summarise.")

    transcript = " ".join(transcript_parts)
    duration = int(time.monotonic() - stream_start)
    duration_str = format_duration(duration)

    # 3. Summarise (mirror the video handler's brief + key_points calls).
    _inflight_phase(job, PHASE_SUMMARIZING)
    await _job_react(job, "\U0001f9e0")  # 🧠
    brief, key_points = await asyncio.gather(
        summarize(
            transcript, PROMPT_BRIEF, LLM_MAX_TOKENS_BRIEF,
            reduce_template=REDUCE_BRIEF,
            title=title, duration=duration_str, reference_block="",
        ),
        summarize(
            transcript, PROMPT_KEY_POINTS, LLM_MAX_TOKENS_KEY_POINTS,
            reduce_template=REDUCE_KEY_POINTS,
            title=title, duration=duration_str, reference_block="",
            char_cap=SUMMARY_CHAR_CAP,
        ),
    )
    brief = sanitize_llm_output(brief)
    key_points = sanitize_llm_output(key_points)

    # 4. Post embeds (no SpeakerRenameView — streaming mode has no speakers).
    detail_channel = resolve_summary_channel(job.channel)
    await send_long_embed(detail_channel, "Key Points", key_points, 0x9B59B6)

    embed = discord.Embed(
        title=f"TL;DW: {truncate(title, 240)}",
        url=job.url,
        description=truncate(brief, 4000),
        color=0x9B59B6,  # purple — distinct from video (red) / web (blue)
    )
    embed.set_footer(text=f"Live · {duration_str}")
    if job.message is not None:
        await job.channel.send(embed=embed, reference=job.message)
    else:
        await job.channel.send(embed=embed)

    for emoji in PROCESSING_EMOJI:
        await _job_remove_react(job, emoji)
    await _job_react(job, "\u2705")  # ✅
    log.info("[%s] Done — live summary posted (%d chars, %s)",
             job.video_id, len(transcript), duration_str)
```

- [ ] **Step 4: Run the full regression suite — all should pass**

```bash
make test
```
Expected: all tests pass (361 original + 9 new = 370+).

- [ ] **Step 5: Commit**

```bash
git add bot/main.py tests/test_regression.py
git commit -m "bot: add process_live() — WS client to whisper-live, summarise on stream end"
```

---

## Task 8: End-to-end smoke test

- [ ] **Step 1: Build and start whisper-live locally (5090)**

```bash
cd whisper-live
pip install -r requirements.txt
LIVE_MODEL=large-v3-turbo DEVICE=cuda python3 server.py
```
Expected: `Ready — max concurrent streams: 4`; model loads without OOM.

- [ ] **Step 2: Health + probe**

```bash
curl -s http://localhost:7861/health | python3 -m json.tool
# Probe a known live stream (replace with a currently-live URL):
curl -s "http://localhost:7861/probe?url=https://www.youtube.com/watch?v=<LIVE_ID>" | python3 -m json.tool
```
Expected: health returns `status: ok`; probe returns `{"is_live": true, "title": "..."}` for a live URL, `{"is_live": false, ...}` for a VOD.

- [ ] **Step 3: `/transcribe-chunk` with a real audio file**

```bash
ffmpeg -i /path/to/any.mp3 -f s16le -ar 16000 -ac 1 /tmp/test.pcm
curl -s -X POST http://localhost:7861/transcribe-chunk \
     -H "Content-Type: application/octet-stream" \
     --data-binary @/tmp/test.pcm | python3 -m json.tool
```
Expected: `{"segments": [{"text": "...", ...}]}`.

- [ ] **Step 4: `/ws-url` against a live stream (Python one-liner)**

```bash
python3 - <<'PY'
import asyncio, json, aiohttp
URL = "https://www.youtube.com/watch?v=<LIVE_ID>"
async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect("ws://localhost:7861/ws-url") as ws:
            await ws.send_json({"url": URL})
            async for m in ws:
                d = json.loads(m.data)
                print(d.get("type"), d.get("text", "")[:80])
                if d["type"] in ("done", "error"): break
asyncio.run(main())
PY
```
Expected: a stream of `segment` lines, then `done`. (Ctrl-C to stop early — a real live stream runs until the broadcaster ends it.)

- [ ] **Step 5: Full stack + Discord**

```bash
make build && make redeploy
```
Paste a currently-live YouTube/Twitch URL in a monitored channel. Expected: 🎙️ reaction appears, then 🧠 when summarising (after the stream ends), then a purple TL;DW embed + Key Points. A VOD URL still routes to the normal 🎧 video flow.

- [ ] **Step 6: Gradio Live tab**

Open `http://localhost:7860` → **Live** tab → grant mic → speak. Expected: transcript text appears within ~10 s.

- [ ] **Step 7: Final regression run + commit**

```bash
make test
git add -A && git commit -m "whisper-live: end-to-end smoke test green, feature complete"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Multi-stream live transcription | Task 1 (`_lock`) + Task 2 (`LIVE_MAX_STREAMS`) |
| faster-whisper streaming (not WhisperX batch) | Task 1 |
| Separate `whisper-live` service | Tasks 2–4 |
| 5090 — no artificial VRAM limits | `LIVE_MAX_STREAMS` freely tunable |
| Bot transcribes live URL → summary at end | Tasks 6–7 |
| UI live transcription (mic) | Task 5 |
| Regression tests | Task 7 (9 tests) |
| Compose integration | Task 4 |

### Corrections applied during verification (why this differs from a naive plan)

1. **Bot has no yt-dlp/ffmpeg** (`bot/Dockerfile` = `python:3.13-slim`). The `yt-dlp | ffmpeg` pipeline lives entirely in whisper-live; the bot is a WS client. (Affected Tasks 2, 3, 6, 7.)
2. **`SpeakerRenameView` signature** is `(job_video_id, channel_id, translate)` — and live mode has no speakers, so `process_live` attaches no view. (Affected Task 7.)
3. **Live detection must precede download** (a live download never ends) — implemented as `IsLiveError` raised in `process()`'s cache-miss branch + worker re-route, mirroring the existing `NotAVideoError` idiom. This one interception covers auto-paste, reply-trigger, and slash-command jobs (all funnel through `process()`), avoiding edits to 7+ job-creation sites. (Affected Task 6.)
4. **`summarize()` call shape** verified against the video handler: extra kwargs (`char_cap`, `reference_block`) are ignored by `str.format()` when a template doesn't use them, and missing-required would `KeyError` — `PROMPT_BRIEF`/`PROMPT_KEY_POINTS`/`REDUCE_*` placeholders all satisfied by the calls in Task 7.
5. **Reaction lifecycle** — each handler owns its own reactions; `process_live` reacts 🎙️ → 🧠 → removes `PROCESSING_EMOJI` → ✅, matching `process_image`.

### Scope exclusions (deliberate, not gaps)

- **Discord voice-channel recording** — needs discord.py voice + PyNaCl; separate plan.
- **VLM enrichment / chapters for live** — frame extraction on an unbounded stream and time-ordered chapters don't apply to streaming; omitted.
- **`/ws` raw-PCM WebSocket** — dropped; Gradio uses `/transcribe-chunk`, the bot uses `/ws-url`. No third consumer.

### Dependency checks to run before starting

- `scipy` importable in the whisper image (Task 5 resampling). `python3 -c "import scipy"` — add to `requirements.txt` if it fails.
- whisper-live CUDA base (`12.4.1`) vs the 5090's driver: if CTranslate2 errors on load, bump the base to a `12.6`/`12.8` cudnn runtime tag and `faster-whisper`/`ctranslate2` accordingly.
