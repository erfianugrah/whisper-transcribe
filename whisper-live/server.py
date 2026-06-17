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
WS   /ws-stream           — client streams raw 16 kHz mono int16 PCM (binary
                            frames); server runs LocalAgreement streaming and
                            replies {"type":"commit","text"} (stable, final) and
                            {"type":"partial","text"} (provisional). Send text
                            "done" (or disconnect) to flush + close. Used by the
                            browser mic tab via the whisper-service WS proxy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def _lifespan(app):
    # Starlette ≥1.0 removed the on_startup= kwarg; use the lifespan
    # context-manager pattern (mirrors app.py:_lifespan).
    global _transcriber
    log.info(f"Loading model {MODEL_NAME!r} on {DEVICE} ({COMPUTE_TYPE})…")
    _transcriber = StreamingTranscriber(MODEL_NAME, DEVICE, COMPUTE_TYPE)
    log.info(f"Ready — max concurrent streams: {LIVE_MAX_STREAMS}")
    yield


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


# ── WebSocket: browser mic PCM → streaming (LocalAgreement) transcript ──────────
# Receiving and inference are decoupled: the receive loop only enqueues audio
# (never blocks, so keepalive pings stay answered), and a separate task re-runs
# inference on the current buffer at a fixed wall-clock cadence.
PROCESS_INTERVAL = float(os.environ.get("LIVE_PROCESS_INTERVAL", "0.6"))
# Streaming/VAD knobs (env-tunable without a rebuild).
_SESSION_KW = dict(
    min_chunk_s=float(os.environ.get("LIVE_MIN_CHUNK_S", "1.0")),
    trim_s=float(os.environ.get("LIVE_TRIM_S", "8.0")),
    tail_silence_s=float(os.environ.get("LIVE_TAIL_SILENCE_S", "0.6")),
    silence_rms=float(os.environ.get("LIVE_SILENCE_RMS", "0.006")),
    min_silence_ms=int(os.environ.get("LIVE_MIN_SILENCE_MS", "300")),
)


async def ws_stream_endpoint(websocket: WebSocket) -> None:
    global _active_streams
    await websocket.accept()
    if _active_streams >= LIVE_MAX_STREAMS:
        await websocket.send_text(json.dumps({"type": "error", "message": "server at capacity"}))
        await websocket.close(1013)
        return
    _active_streams += 1
    session = _transcriber.new_session(**_SESSION_KW)
    stop = asyncio.Event()
    log.info(f"[ws-stream] opened ({_active_streams}/{LIVE_MAX_STREAMS})")

    async def processor() -> None:
        while not stop.is_set():
            await asyncio.sleep(PROCESS_INTERVAL)
            try:
                committed, partial, eou = await _transcriber.session_process(session)
            except Exception as e:
                log.error(f"[ws-stream] inference error: {e}")
                continue
            if committed or eou:
                await websocket.send_text(
                    json.dumps({"type": "commit", "text": committed, "eou": eou})
                )
            if partial:
                await websocket.send_text(json.dumps({"type": "partial", "text": partial}))

    proc_task = asyncio.create_task(processor())
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                session.insert_audio(data)
                continue
            text = msg.get("text")
            if text and (text == "done" or '"done"' in text):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"[ws-stream] error: {e}")
    finally:
        stop.set()
        proc_task.cancel()
        try:
            final = await _transcriber.session_finish(session)
            if final:
                await websocket.send_text(
                    json.dumps({"type": "commit", "text": final, "eou": True})
                )
            await websocket.send_text(json.dumps({"type": "done"}))
        except Exception:
            pass
        _active_streams -= 1
        log.info(f"[ws-stream] closed ({_active_streams}/{LIVE_MAX_STREAMS})")


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/probe", probe),
        Route("/transcribe-chunk", transcribe_chunk, methods=["POST"]),
        WebSocketRoute("/ws-url", ws_url_endpoint),
        WebSocketRoute("/ws-stream", ws_stream_endpoint),
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")
