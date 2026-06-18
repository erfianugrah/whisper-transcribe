"""Discord voice-call live transcription.

See docs/plans/2026-06-18-discord-voice-transcription.md for the full design.

This module is import-safe even when `discord-ext-voice-recv` is absent (e.g.
the test stub layer): the extension import is guarded and exposed via
`VOICE_RECV_AVAILABLE`. The pure audio helpers (`resample_48k_stereo_to_16k_mono`,
`mix_streams`, `SilenceInjector`) depend only on numpy and are unit-tested.

Phase 1 (current scope): live transcription. On `/transcribe-join` we connect
with `VoiceRecvClient`, sum every speaker into one 16 kHz mono stream (slot
mixer + wall-clock silence reconstruction), pipe it to whisper-live's
`/ws-stream`, and post committed utterances to the configured transcript
channel as they land. `/transcribe-leave` flushes, closes the stream, and
disconnects. Everything is gated behind `VOICE_TRANSCRIBE_ENABLED`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np

log = logging.getLogger("tldw.voice")

# ── Guarded extension import ────────────────────────────────────────────────
# discord-ext-voice-recv is alpha and not installed in the test stub layer.
# Import defensively so `import voice` never breaks `import main`.
try:
    from discord.ext import voice_recv  # type: ignore

    VOICE_RECV_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only when lib missing
    voice_recv = None  # type: ignore
    VOICE_RECV_AVAILABLE = False
    _IMPORT_ERROR = e

# ── Config (env-tunable, no rebuild) ────────────────────────────────────────
VOICE_TRANSCRIBE_ENABLED = os.environ.get("VOICE_TRANSCRIBE_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)
# Discord delivers 48 kHz 16-bit stereo PCM; whisper-live wants 16 kHz mono.
DISCORD_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
DECIMATION = DISCORD_SAMPLE_RATE // TARGET_SAMPLE_RATE  # 3
# Cap silence reconstruction so a 10-minute idle doesn't enqueue 10 min of zeros.
VOICE_MAX_SILENCE_S = float(os.environ.get("VOICE_MAX_SILENCE_S", "2.0"))
# Minimum gap before we bother padding (≈ 2 Discord frames).
VOICE_MIN_GAP_S = float(os.environ.get("VOICE_MIN_GAP_S", "0.04"))

# whisper-live streaming endpoint (binary 16 kHz mono PCM in, JSON commits out).
WHISPER_LIVE_URL = os.environ.get("WHISPER_LIVE_URL", "http://localhost:7861")
WHISPER_LIVE_WS = WHISPER_LIVE_URL.replace("http://", "ws://", 1).replace(
    "https://", "wss://", 1
)
# Channel that live transcript lines are posted to. 0 / unset → post in the
# channel where /transcribe-join was invoked.
VOICE_TRANSCRIPT_CHANNEL_ID = int(
    os.environ.get("VOICE_TRANSCRIPT_CHANNEL_ID", "0") or "0"
)


_CONSENT_NOTICE = (
    "🔴 **This voice channel is now being transcribed.** "
    "Audio is converted to text live and not stored. "
    "Leave the channel if you do not consent."
)


# ── Pure audio helpers (numpy; unit-tested) ─────────────────────────────────
def resample_48k_stereo_to_16k_mono(pcm: bytes) -> bytes:
    """48 kHz int16 stereo (interleaved L,R) → 16 kHz int16 mono.

    Downmix by averaging the two channels, then decimate 3:1 by block-average
    (a crude but adequate anti-alias low-pass for 16 kHz speech ASR). Returns
    little-endian int16 bytes. Empty / sub-frame input → b"".
    """
    if not pcm:
        return b""
    # Trim to a whole number of stereo int16 frames (4 bytes each).
    usable = len(pcm) - (len(pcm) % 4)
    if usable <= 0:
        return b""
    samples = np.frombuffer(pcm[:usable], dtype="<i2")
    stereo = samples.reshape(-1, 2).astype(np.int32)
    mono = stereo.mean(axis=1)  # float64, averaged L+R
    # Block-average decimate by 3 (drop the ragged tail < DECIMATION).
    n = (mono.shape[0] // DECIMATION) * DECIMATION
    if n == 0:
        return b""
    decimated = mono[:n].reshape(-1, DECIMATION).mean(axis=1)
    out = np.clip(np.round(decimated), -32768, 32767).astype("<i2")
    return out.tobytes()


def mix_streams(buffers: list[bytes]) -> bytes:
    """Sum N equal-length 16 kHz mono int16 buffers into one, clipping to int16.

    Used by the Phase-1 single mixed stream. Shorter buffers are zero-padded to
    the longest length so a late/short packet doesn't truncate the mix.
    """
    chunks = [np.frombuffer(b, dtype="<i2").astype(np.int32) for b in buffers if b]
    if not chunks:
        return b""
    length = max(c.shape[0] for c in chunks)
    acc = np.zeros(length, dtype=np.int32)
    for c in chunks:
        acc[: c.shape[0]] += c
    return np.clip(acc, -32768, 32767).astype("<i2").tobytes()


class SilenceInjector:
    """Reconstructs the silence Discord omits during quiet periods.

    Discord stops sending RTP packets when a user is silent, but whisper-live
    detects end-of-utterance from trailing silence — so consecutive utterances
    would glue together. Before forwarding a frame, ask `silence_before(now,
    frame_seconds)` for zero-PCM bytes representing the gap since the previously
    expected audio position. Padding is capped at `VOICE_MAX_SILENCE_S`.

    `now` is a monotonic-clock float (seconds). The class is clock-agnostic for
    testability — the caller supplies the timestamp.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_silence_s: float = VOICE_MAX_SILENCE_S,
        min_gap_s: float = VOICE_MIN_GAP_S,
    ):
        self.sample_rate = sample_rate
        self.max_silence_samples = int(max_silence_s * sample_rate)
        self.min_gap_s = min_gap_s
        self._next_expected: float | None = None

    def silence_before(self, now: float, frame_seconds: float) -> bytes:
        """Zero-PCM bytes to enqueue before the frame arriving at `now`."""
        if self._next_expected is None:
            self._next_expected = now + frame_seconds
            return b""
        gap = now - self._next_expected
        self._next_expected = now + frame_seconds
        if gap <= self.min_gap_s:
            return b""
        n = min(int(gap * self.sample_rate), self.max_silence_samples)
        return b"\x00\x00" * n

    def reset(self) -> None:
        self._next_expected = None


# ── Opus robustness + libopus load + Phase 1 streaming ────────────────────────
def install_opus_robustness_patch() -> bool:
    """Patch discord-ext-voice-recv 0.5.2a179 for discord.py 2.7.1 compatibility.

    Two monkeypatches on `PacketDecoder` (idempotent; no-op if the extension is
    missing):

    1. **DAVE payload decryption.** discord.py 2.7.1 made Discord's end-to-end
       voice encryption (DAVE) mandatory. The library's transport-layer AEAD
       decrypt succeeds, but the opus *payload* is still DAVE-encrypted, so it
       decodes to noise/gibberish (voice_recv issue #53 / PR #54 — not yet in
       any PyPI release). We replace `_process_packet` to DAVE-decrypt via the
       connection's `dave_session` before opus decode, mirroring the community
       fix. Without this, ALL received audio is unintelligible.

    2. **Corrupted-packet guard.** `PacketRouter._do_run` has no per-packet
       guard: an `OpusError: corrupted stream` (intermittent on real audio)
       propagates to `run()`, which calls `stop_listening()` — killing the whole
       call. We wrap `pop_data` to drop the bad frame and return None instead.
    """
    if not VOICE_RECV_AVAILABLE:
        return False
    try:
        from discord.ext.voice_recv.opus import PacketDecoder, VoiceData
        from discord.opus import OpusError
    except Exception as e:  # pragma: no cover
        log.warning("voice: could not install opus robustness patch: %s", e)
        return False
    try:
        from davey import MediaType
        _has_dave = True
    except Exception:
        MediaType = None  # type: ignore
        _has_dave = False
    if getattr(PacketDecoder, "_tldw_patched", False):
        return True

    # (2) corrupted-packet guard
    _orig_pop = PacketDecoder.pop_data

    def _safe_pop_data(self, *, timeout: float = 0):
        try:
            return _orig_pop(self, timeout=timeout)
        except OpusError as e:
            log.debug(
                "voice: dropped corrupted opus packet (ssrc=%s): %s",
                getattr(self, "ssrc", "?"), e,
            )
            return None

    PacketDecoder.pop_data = _safe_pop_data

    # (1) DAVE payload decryption — resolve member, decrypt E2EE, then decode
    def _process_packet(self, packet):
        pcm = None
        member = self._get_cached_member()
        if member is None:
            self._cached_id = self.sink.voice_client._get_id_from_ssrc(self.ssrc)
            member = self._get_cached_member()
        try:
            conn = getattr(self.sink.voice_client, "_connection", None)
            dave = getattr(conn, "dave_session", None)
            if (
                _has_dave and member is not None
                and not packet.is_silence()
                and packet.decrypted_data is not None
                and dave is not None and getattr(dave, "ready", False)
            ):
                packet.decrypted_data = dave.decrypt(
                    member.id, MediaType.audio, packet.decrypted_data
                )
        except Exception as e:  # never kill the recv thread over one packet
            log.debug("voice: DAVE decrypt skipped (ssrc=%s): %s", self.ssrc, e)
        if not self.sink.wants_opus():
            packet, pcm = self._decode_packet(packet)
        data = VoiceData(packet, member, pcm=pcm)
        self._last_seq = packet.sequence
        self._last_ts = packet.timestamp
        return data

    PacketDecoder._process_packet = _process_packet
    PacketDecoder._tldw_patched = True
    log.info(
        "voice: installed voice_recv patch (DAVE decrypt=%s + corrupted-packet guard)",
        "on" if _has_dave else "UNAVAILABLE",
    )
    return True


class VoiceTranscriptionSession:
    """Streams Discord voice to whisper-live exactly like the browser-mic Live
    tab does — one continuous, in-order 16 kHz mono PCM stream.

    Audio arrives on the receive (router) thread via `feed()`, which only
    resamples and hands the frame to the event loop (never blocks/awaits — the
    router thread must stay non-blocking). The event loop appends frames to a
    queue **in arrival order** (the library's per-speaker jitter buffer already
    delivers them sequenced), inserting reconstructed silence for real pauses
    (`SilenceInjector`) so whisper-live can detect end-of-utterance. A sender
    task drains the queue to the WebSocket; a reader task posts committed
    utterances to Discord.

    NOTE: this is a single mixed stream. Concurrent speakers interleave rather
    than sum (true overlap mixing + per-speaker attribution is Phase 2). For the
    common turn-taking case this produces clean, intelligible audio — matching
    the proven mic path instead of reinventing timing/mixing.
    """

    SEND_INTERVAL = 0.25  # how often the sender drains the queue to the WS

    def __init__(self, loop, ws, post_cb, max_silence_s: float = VOICE_MAX_SILENCE_S):
        self._loop = loop
        self._ws = ws
        self._post_cb = post_cb  # async fn(text: str)
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._injector = SilenceInjector(max_silence_s=max_silence_s)
        self._stop = asyncio.Event()
        self._sender_task = None
        self._reader_task = None
        self._sent_bytes = 0

    # — recv thread —
    def feed(self, mono16: bytes) -> None:
        """Enqueue one resampled 16 kHz mono frame, in arrival order. Called
        off-loop on the recv thread; no awaits, no blocking."""
        if not mono16:
            return
        try:
            self._loop.call_soon_threadsafe(self._ingest, time.monotonic(), mono16)
        except RuntimeError:
            pass  # loop closed mid-call

    # — event loop —
    def _ingest(self, now: float, mono16: bytes) -> None:
        frame_s = (len(mono16) // 2) / TARGET_SAMPLE_RATE
        sil = self._injector.silence_before(now, frame_s)
        if sil:
            self._queue.put_nowait(sil)
        self._queue.put_nowait(mono16)

    def start(self) -> None:
        self._sender_task = self._loop.create_task(self._sender_loop())
        self._reader_task = self._loop.create_task(self._reader_loop())

    async def _drain_once(self) -> bytes:
        """Concatenate everything currently queued into one PCM blob."""
        parts = []
        while not self._queue.empty():
            parts.append(self._queue.get_nowait())
        return b"".join(parts)

    async def _sender_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.SEND_INTERVAL)
                pcm = await self._drain_once()
                if pcm:
                    self._sent_bytes += len(pcm)
                    await self._ws.send_bytes(pcm)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("voice: sender loop error: %s", e)

    async def _reader_loop(self) -> None:
        import aiohttp

        pending: list[str] = []
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    if t == "commit":
                        text = (data.get("text") or "").strip()
                        if text:
                            pending.append(text)
                        if data.get("eou") and pending:
                            line = " ".join(pending).strip()
                            pending.clear()
                            if line:
                                await self._post_cb(line)
                    elif t == "done":
                        break
                    elif t == "error":
                        log.error("voice: whisper-live error: %s", data.get("message"))
                        await self._post_cb(
                            "⚠️ transcription backend rejected the stream: "
                            f"{data.get('message')}"
                        )
                        break
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("voice: reader loop error: %s", e)
        finally:
            if pending:
                line = " ".join(pending).strip()
                if line:
                    try:
                        await self._post_cb(line)
                    except Exception:
                        pass

    async def close(self) -> None:
        """Flush remaining audio, signal whisper-live done, drain final commits."""
        self._stop.set()
        if self._sender_task:
            self._sender_task.cancel()
        try:
            pcm = await self._drain_once()
            if pcm:
                await self._ws.send_bytes(pcm)
            await self._ws.send_str("done")
        except Exception:
            pass
        if self._reader_task:
            try:
                await asyncio.wait_for(self._reader_task, timeout=10)
            except Exception:
                self._reader_task.cancel()


def opus_loaded() -> bool:
    """Load libopus (needed to decode received voice). Returns success."""
    try:
        import discord  # local import; main owns the global discord import

        if discord.opus.is_loaded():
            return True
        discord.opus._load_default()
        return discord.opus.is_loaded()
    except Exception as e:
        log.error("voice: libopus failed to load (%s) — voice disabled", e)
        return False


def register_voice_commands(bot) -> bool:
    """Register /transcribe-join + /transcribe-leave. Returns whether enabled.

    No-op (returns False) unless VOICE_TRANSCRIBE_ENABLED and the extension +
    libopus are both available — so a misconfigured deploy degrades to "feature
    off", never a crash.
    """
    if not VOICE_TRANSCRIBE_ENABLED:
        log.info("voice: VOICE_TRANSCRIBE_ENABLED not set — voice transcription off")
        return False
    if not VOICE_RECV_AVAILABLE:
        log.error("voice: discord-ext-voice-recv not importable — voice off (%s)", _IMPORT_ERROR)
        return False
    if not opus_loaded():
        return False
    install_opus_robustness_patch()

    import aiohttp
    import discord

    # Per-guild live state: guild_id -> {"session", "http", "ws"} so /leave can
    # tear everything down cleanly even across reconnects.
    _live: dict[int, dict] = {}

    def _resolve_post_channel(interaction):
        """Where transcript lines go: the configured channel, else the invoking one."""
        if VOICE_TRANSCRIPT_CHANNEL_ID:
            ch = bot.get_channel(VOICE_TRANSCRIPT_CHANNEL_ID)
            if ch is not None:
                return ch
            log.warning(
                "voice: VOICE_TRANSCRIPT_CHANNEL_ID=%s not found — posting in invoking channel",
                VOICE_TRANSCRIPT_CHANNEL_ID,
            )
        return interaction.channel

    @bot.tree.command(
        name="transcribe-join",
        description="Join your voice channel and live-transcribe it to the transcript channel",
    )
    async def transcribe_join(interaction: "discord.Interaction"):
        user = interaction.user
        channel = getattr(getattr(user, "voice", None), "channel", None)
        if channel is None:
            await interaction.response.send_message(
                "You must be in a voice channel first.", ephemeral=True
            )
            return
        if interaction.guild.voice_client is not None:
            await interaction.response.send_message(
                "Already connected to a voice channel in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)
        post_channel = _resolve_post_channel(interaction)
        loop = asyncio.get_running_loop()

        async def _post(line: str) -> None:
            try:
                await post_channel.send(line[:2000])
            except Exception as e:
                log.error("voice: failed to post transcript line: %s", e)

        # Open the whisper-live streaming socket BEFORE joining voice, so a
        # backend outage fails the command cleanly (no orphaned VC connection).
        client = aiohttp.ClientSession()
        try:
            ws = await client.ws_connect(f"{WHISPER_LIVE_WS}/ws-stream", timeout=30)
        except Exception as e:
            await client.close()
            log.error("voice: could not reach whisper-live /ws-stream: %s", e)
            await interaction.followup.send(
                "❌ Couldn't reach the transcription backend (whisper-live). Try again later.",
                ephemeral=True,
            )
            return

        session = VoiceTranscriptionSession(loop, ws, _post)
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)

        def _on_packet(speaker, data) -> None:
            # Runs on the recv (router) thread: resample only, then hand off.
            try:
                pcm = getattr(data, "pcm", None)
                if not pcm:
                    return
                mono16 = resample_48k_stereo_to_16k_mono(pcm)
                if mono16:
                    session.feed(mono16)
            except Exception as e:  # never raise on the recv thread
                log.debug("voice: packet handler error: %s", e)

        session.start()
        vc.listen(voice_recv.BasicSink(_on_packet))
        _live[interaction.guild.id] = {"session": session, "http": client, "ws": ws}

        await post_channel.send(_CONSENT_NOTICE)
        where = (
            f" Transcript → <#{post_channel.id}>."
            if post_channel.id != channel.id else ""
        )
        await interaction.followup.send(
            f"Joined **{channel.name}** — live transcription active.{where}",
            ephemeral=True,
        )
        log.info(
            "voice: joined %s (guild %s) → streaming to whisper-live, posting in %s",
            channel.name, interaction.guild.id, post_channel.id,
        )

    @bot.tree.command(
        name="transcribe-leave",
        description="Stop transcribing and leave the voice channel",
    )
    async def transcribe_leave(interaction: "discord.Interaction"):
        vc = interaction.guild.voice_client
        if vc is None:
            await interaction.response.send_message(
                "Not connected to a voice channel here.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        try:
            vc.stop_listening()
        except Exception:
            pass
        state = _live.pop(interaction.guild.id, None)
        if state:
            try:
                await state["session"].close()
            except Exception as e:
                log.debug("voice: session close error: %s", e)
            try:
                await state["ws"].close()
            except Exception:
                pass
            try:
                await state["http"].close()
            except Exception:
                pass
        await vc.disconnect()
        await interaction.followup.send("Left the voice channel.", ephemeral=True)
        log.info("voice: left voice channel (guild %s)", interaction.guild.id)

    log.info("voice: /transcribe-join + /transcribe-leave registered")
    return True
