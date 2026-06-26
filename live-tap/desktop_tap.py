#!/usr/bin/env python3
"""desktop_tap.py — standalone live audio → transcript tap.

Captures system / OBS / mic audio, streams raw 16 kHz mono PCM to the
whisper-live `/ws-stream` endpoint, and prints the committed transcript to
stdout. No Discord bot, no browser, no SPA — just a pipe you can hang an LLM
off the end of:

    python desktop_tap.py | your-llm-research-tool

Audio source
------------
whisper-live runs in Docker and cannot see the host's audio devices, so the
*capture* happens here. Three ways:

  • `--loopback` — OS-native system-audio capture, NO virtual cable: WASAPI
    loopback on Windows (pyaudiowpatch), PulseAudio/PipeWire monitor on Linux
    (soundcard). Captures whatever is playing. `pip install pyaudiowpatch`
    (Windows) or `pip install soundcard` (Linux). List devices:
    `--list-loopback`.
  • ffmpeg device (default) — `--device 'audio=...'` via dshow/pulse. Needs a
    capturable device (a real mic, or a virtual cable for system audio).
  • `--self-test` — synthetic tone, no hardware (connectivity check).

Docker Desktop forwards container port 7861 to localhost on both Windows and
WSL, so the default `--url ws://localhost:7861/ws-stream` works from either
side. For Windows system audio, run on Windows (loopback needs the host APIs).
Optional session handshake: `--language en` / `--translate`.

Getting OBS / desktop audio into a capturable device
----------------------------------------------------
Windows has no built-in loopback device ffmpeg can grab directly. Pick one:

  • VB-Audio Virtual Cable (simplest). Set Windows default playback (or OBS
    "Monitoring Device") to "CABLE Input". Then capture "CABLE Output":
      --device 'audio=CABLE Output (VB-Audio Virtual Cable)'
  • VoiceMeeter — same idea, route OBS/desktop to a virtual bus, capture it.
  • A plain microphone needs no virtual cable:
      --device 'audio=Microphone (Your Mic Name)'

List the exact device names ffmpeg sees:
    python desktop_tap.py --list-devices

Quick connectivity check (no audio hardware — sends a 5 s sine tone):
    python desktop_tap.py --self-test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys

import websockets

SAMPLE_RATE = 16000           # whisper-live expects 16 kHz mono int16
BYTES_PER_SEC = SAMPLE_RATE * 2
READ_CHUNK = BYTES_PER_SEC // 10  # ~100 ms per WS frame


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


# ── ffmpeg device enumeration ──────────────────────────────────────────────────
async def list_devices(ffmpeg: str) -> int:
    """Dump dshow audio/video device names (ffmpeg writes them to stderr)."""
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-hide_banner", "-list_devices", "true",
        "-f", "dshow", "-i", "dummy",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    sys.stderr.write(err.decode(errors="replace"))
    sys.stderr.flush()
    return 0


# ── audio producers (yield raw 16 kHz mono int16 PCM blocks) ───────────────────
async def ffmpeg_source(ffmpeg: str, device: str, input_format: str):
    """Spawn ffmpeg capturing `device` → 16 kHz mono s16le on stdout."""
    args = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", input_format, "-i", device,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
    ]
    log(f"[tap] spawning: {' '.join(args)}")
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async def warn_stderr() -> None:
        assert proc.stderr is not None
        async for line in proc.stderr:
            log("[ffmpeg]", line.decode(errors="replace").rstrip())

    asyncio.create_task(warn_stderr())
    try:
        assert proc.stdout is not None
        while True:
            block = await proc.stdout.read(READ_CHUNK)
            if not block:
                break
            yield block
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def sine_source(seconds: float = 5.0):
    """Synthetic 220 Hz tone — validates the WS path without audio hardware."""
    total = int(seconds * SAMPLE_RATE)
    phase = 0.0
    step = 2 * math.pi * 220 / SAMPLE_RATE
    emitted = 0
    while emitted < total:
        n = min(READ_CHUNK // 2, total - emitted)
        buf = bytearray()
        for _ in range(n):
            buf += struct.pack("<h", int(math.sin(phase) * 8000))
            phase += step
        emitted += n
        yield bytes(buf)
        await asyncio.sleep(n / SAMPLE_RATE)  # pace at real-time


# ── native loopback (no virtual cable) ──────────────────────────────────
# Captures whatever the system is *playing* via OS-native loopback — WASAPI on
# Windows (pyaudiowpatch), PulseAudio/PipeWire monitor on Linux (soundcard).
# No VB-Audio / virtual cable, no driver install. Backend auto-selected per OS.
def _to_pcm16(mono_float) -> bytes:
    """float32 mono in [-1, 1] → little-endian int16 PCM bytes."""
    import numpy as np
    return (np.clip(mono_float, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _resample_linear(mono_float, in_rate: int, out_rate: int = SAMPLE_RATE):
    """Nearest-sample linear resample of a mono float array to out_rate. Good
    enough for speech (whisper is robust to mild aliasing); avoids a scipy dep."""
    import numpy as np
    if in_rate == out_rate or mono_float.size == 0:
        return mono_float
    n_out = int(round(mono_float.size * out_rate / in_rate))
    idx = np.clip(
        (np.arange(n_out) * in_rate / out_rate).astype(np.int64), 0, mono_float.size - 1
    )
    return mono_float[idx]


async def _soundcard_source(device_name: str | None):
    """Linux (and cross-platform fallback) loopback via the `soundcard` pkg,
    which resamples to SAMPLE_RATE for us."""
    import numpy as np
    import soundcard as sc

    if device_name:
        mic = sc.get_microphone(device_name, include_loopback=True)
    else:
        try:
            mic = sc.get_microphone(sc.default_speaker().name, include_loopback=True)
        except Exception:
            mic = sc.default_microphone()  # Linux monitor source fallback
    log(f"[tap] loopback (soundcard): {mic.name}")
    loop = asyncio.get_event_loop()
    block = SAMPLE_RATE // 10  # ~100 ms
    with mic.recorder(samplerate=SAMPLE_RATE, channels=None) as rec:
        while True:
            data = await loop.run_in_executor(None, rec.record, block)
            mono = data.mean(axis=1) if getattr(data, "ndim", 1) > 1 else data
            yield _to_pcm16(mono)


def _pwp_default_loopback(p):
    """Resolve the WASAPI loopback device for the default render endpoint."""
    import pyaudiowpatch as pyaudio
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    dev = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if not dev.get("isLoopbackDevice"):
        for lb in p.get_loopback_device_info_generator():
            if dev["name"] in lb["name"]:
                return lb
    return dev


async def _pyaudiowpatch_source(device_name: str | None):
    """Windows WASAPI loopback via pyaudiowpatch (int16 native rate → 16 kHz mono)."""
    import numpy as np
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        if device_name:
            dev = next(
                d for d in p.get_loopback_device_info_generator()
                if device_name.lower() in d["name"].lower()
            )
        else:
            dev = _pwp_default_loopback(p)
        in_rate = int(dev["defaultSampleRate"])
        in_ch = int(dev["maxInputChannels"]) or 2
        block = int(in_rate * 0.1)
        log(f"[tap] loopback (pyaudiowpatch): {dev['name']} {in_rate}Hz {in_ch}ch")
        stream = p.open(
            format=pyaudio.paInt16, channels=in_ch, rate=in_rate,
            frames_per_buffer=block, input=True, input_device_index=dev["index"],
        )
        loop = asyncio.get_event_loop()
        try:
            while True:
                raw = await loop.run_in_executor(None, stream.read, block, False)
                a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if in_ch > 1:
                    a = a.reshape(-1, in_ch).mean(axis=1)
                yield _to_pcm16(_resample_linear(a, in_rate))
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()


async def loopback_source(device_name: str | None = None):
    """OS-native system-audio loopback. Windows → pyaudiowpatch (fall back to
    soundcard); everything else → soundcard."""
    if sys.platform == "win32":
        try:
            import pyaudiowpatch  # noqa: F401
        except ImportError:
            log("[tap] pyaudiowpatch not installed; falling back to soundcard")
        else:
            async for b in _pyaudiowpatch_source(device_name):
                yield b
            return
    async for b in _soundcard_source(device_name):
        yield b


async def list_loopback() -> int:
    """Enumerate available loopback devices for the current OS."""
    if sys.platform == "win32":
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            for d in p.get_loopback_device_info_generator():
                log(f"  [{d['index']}] {d['name']} ({int(d['defaultSampleRate'])} Hz)")
        finally:
            p.terminate()
    else:
        import soundcard as sc
        for m in sc.all_microphones(include_loopback=True):
            log(f"  {m.name}")
    return 0


# ── transcript sink ────────────────────────────────────────────────────────────
class TranscriptPrinter:
    """Renders commit/partial frames to stdout (commits) + stderr (partials).

    A newline is emitted on each end-of-utterance (`eou`) so a downstream pipe
    gets one line per spoken utterance. `out` optionally tees committed text to
    a file for later use.
    """

    def __init__(self, show_partials: bool, out=None) -> None:
        self.show_partials = show_partials
        self.out = out
        self._line = ""

    def _emit(self, s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()
        if self.out is not None:
            self.out.write(s)
            self.out.flush()

    def handle(self, msg: dict) -> bool:
        t = msg.get("type")
        if t == "commit":
            text = (msg.get("text") or "").strip()
            if text:
                sep = " " if self._line and not self._line.endswith(("\n", " ")) else ""
                self._emit(sep + text)
                self._line += sep + text
            if msg.get("eou") and self._line.strip():
                self._emit("\n")
                self._line = ""
        elif t == "partial" and self.show_partials:
            log("…", (msg.get("text") or "").strip())
        elif t == "error":
            log("[error]", msg.get("message"))
        elif t == "done":
            if self._line.strip():
                self._emit("\n")
            return True
        return False


async def _producer(source, queue: asyncio.Queue) -> None:
    """Drain the audio source into a bounded queue, dropping oldest on overflow
    so a stalled / reconnecting WS never balloons memory."""
    async for block in source:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(block)
    await queue.put(None)  # sentinel: source ended


async def _session(url: str, queue: asyncio.Queue, printer: TranscriptPrinter,
                   config: dict | None = None):
    """One WS connection. Returns ('ended'|'dropped', connected_seconds).

    'ended'   — audio source finished; transcript flushed cleanly.
    'dropped' — whisper-live closed the socket (restart / capacity / network).
    `config` (e.g. {"language": "en", "translate": true}) is sent as the first
    text frame before any audio — the whisper-live per-session handshake.
    """
    t0 = asyncio.get_event_loop().time()
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        if config:
            await ws.send(json.dumps(config))
        log("[tap] connected — streaming audio (Ctrl-C to stop)")
        done = asyncio.Event()

        async def receiver() -> None:
            try:
                async for raw in ws:
                    if printer.handle(json.loads(raw)):
                        done.set()
                        break
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())
        try:
            while True:
                block = await queue.get()
                if block is None:  # source ended → flush + close
                    await ws.send("done")
                    try:
                        await asyncio.wait_for(recv_task, timeout=15)
                    except asyncio.TimeoutError:
                        recv_task.cancel()
                    return "ended", asyncio.get_event_loop().time() - t0
                await ws.send(block)  # bytes → binary frame
        except websockets.ConnectionClosed:
            recv_task.cancel()
            return "dropped", asyncio.get_event_loop().time() - t0


async def run(url: str, source, show_partials: bool, max_reconnects: int,
              out=None, config: dict | None = None) -> int:
    """Stream `source` to whisper-live, auto-reconnecting on drops.

    Mirrors the SPA / voice-bot resilience: a whisper-live restart or network
    blip re-dials with exponential backoff. The audio source keeps running
    across reconnects (recent audio buffers, oldest dropped). A session that
    stays up >30 s resets the reconnect budget — so transient blips don't
    accumulate toward the cap over a long capture, but a server that's down or
    permanently at capacity still gives up after `max_reconnects`.
    """
    printer = TranscriptPrinter(show_partials, out=out)
    queue: asyncio.Queue = asyncio.Queue(maxsize=300)  # ~30 s of 100 ms blocks
    prod = asyncio.create_task(_producer(source, queue))
    attempts = 0
    log(f"[tap] connecting to {url}")
    try:
        while True:
            try:
                status, elapsed = await _session(url, queue, printer, config)
            except (OSError, websockets.WebSocketException) as e:
                status, elapsed = "dropped", 0.0
                log(f"[tap] connection failed: {e}")
            if status == "ended":
                return 0
            if elapsed > 30:
                attempts = 0  # the session was healthy; reset the budget
            attempts += 1
            if attempts > max_reconnects:
                log(f"[tap] giving up after {max_reconnects} reconnect attempts")
                return 1
            backoff = min(30, 2 ** attempts)
            log(f"[tap] reconnecting in {backoff}s ({attempts}/{max_reconnects})")
            await asyncio.sleep(backoff)
    finally:
        prod.cancel()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stream desktop/OBS/mic audio to whisper-live and print the transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", default="ws://localhost:7861/ws-stream",
                   help="whisper-live /ws-stream WebSocket (default: %(default)s)")
    p.add_argument("--device", default="audio=CABLE Output (VB-Audio Virtual Cable)",
                   help="ffmpeg input device (default: %(default)s)")
    p.add_argument("--input-format", default="dshow",
                   help="ffmpeg -f input format: dshow (Windows), pulse (Linux), "
                        "avfoundation (macOS) (default: %(default)s)")
    p.add_argument("--ffmpeg", default="ffmpeg",
                   help="ffmpeg binary (use 'ffmpeg.exe' from WSL) (default: %(default)s)")
    p.add_argument("--partials", action="store_true",
                   help="print provisional (uncommitted) words to stderr")
    p.add_argument("--out", metavar="FILE",
                   help="also append committed transcript to this file")
    p.add_argument("--max-reconnects", type=int, default=5,
                   help="consecutive reconnect attempts before giving up "
                        "(reset after a >30 s healthy session) (default: %(default)s)")
    p.add_argument("--loopback", action="store_true",
                   help="capture system audio via OS-native loopback (no virtual "
                        "cable): WASAPI on Windows, PulseAudio monitor on Linux")
    p.add_argument("--loopback-device", metavar="NAME",
                   help="loopback device name substring (default: system default "
                        "output). List with --list-loopback")
    p.add_argument("--language", metavar="LANG",
                   help="pin spoken language (e.g. en) via the session handshake; "
                        "default = server auto-detect")
    p.add_argument("--translate", action="store_true",
                   help="translate to English (session handshake)")
    p.add_argument("--list-devices", action="store_true",
                   help="list ffmpeg dshow devices and exit")
    p.add_argument("--list-loopback", action="store_true",
                   help="list OS-native loopback devices and exit")
    p.add_argument("--self-test", action="store_true",
                   help="send a 5 s synthetic tone instead of capturing audio")
    args = p.parse_args()

    config: dict = {}
    if args.language:
        config["language"] = args.language
    if args.translate:
        config["translate"] = True
    config = config or None

    out = open(args.out, "a", encoding="utf-8") if args.out else None
    try:
        if args.list_devices:
            return asyncio.run(list_devices(args.ffmpeg))
        if args.list_loopback:
            return asyncio.run(list_loopback())
        if args.self_test:
            return asyncio.run(run(args.url, sine_source(), args.partials,
                                   args.max_reconnects, out, config))
        if args.loopback:
            src = loopback_source(args.loopback_device)
        else:
            src = ffmpeg_source(args.ffmpeg, args.device, args.input_format)
        return asyncio.run(run(args.url, src, args.partials, args.max_reconnects,
                               out, config))
    except KeyboardInterrupt:
        log("\n[tap] stopped")
        return 0
    except (OSError, websockets.WebSocketException) as e:
        log(f"[tap] connection failed: {e}")
        return 1
    finally:
        if out is not None:
            out.close()


if __name__ == "__main__":
    raise SystemExit(main())
