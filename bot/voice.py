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
from datetime import datetime, timedelta, timezone

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
# Max simultaneous per-speaker transcription streams (each uses one whisper-live
# slot; whisper-live's LIVE_MAX_STREAMS caps the pool, default 4).
VOICE_MAX_SPEAKERS = int(os.environ.get("VOICE_MAX_SPEAKERS", "4"))
# Close a speaker's stream after this many seconds of silence, freeing the slot
# for another speaker and flushing their final utterance.
VOICE_SPEAKER_IDLE_S = float(os.environ.get("VOICE_SPEAKER_IDLE_S", "45"))
# How often each speaker's sender drains buffered PCM to whisper-live (latency).
VOICE_SEND_INTERVAL = float(os.environ.get("VOICE_SEND_INTERVAL", "0.15"))
# Per-call transcript threads older than this are deleted by the background
# cleanup loop. 0 (or VOICE_TRANSCRIPT_CHANNEL_ID unset) disables auto-purge.
VOICE_TRANSCRIPT_RETENTION_DAYS = float(
    os.environ.get("VOICE_TRANSCRIPT_RETENTION_DAYS", "7")
)
# How often the auto-purge loop runs.
VOICE_CLEANUP_INTERVAL_H = float(os.environ.get("VOICE_CLEANUP_INTERVAL_H", "6"))

# Per-call threads are named "🎙️ <channel> — <date> <time>"; this prefix plus
# bot ownership is how cleanup distinguishes our threads from human-made ones.
_THREAD_NAME_PREFIX = "\U0001f399"


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


class VoiceUserSession:
    """One speaker's continuous 16 kHz mono stream to whisper-live.

    Mirrors the browser-mic Live tab: resampled frames are appended in arrival
    order (the per-speaker jitter buffer already sequences them), real pauses
    are reconstructed with `SilenceInjector` (so whisper-live detects
    end-of-utterance), a sender task streams the queue to the socket, and a
    reader task posts each committed utterance — attributed to this speaker and
    timestamped — via `post_cb(display_name, text)`.
    """

    def __init__(self, loop, ws, user_id, display_name, post_cb,
                 max_silence_s: float = VOICE_MAX_SILENCE_S,
                 send_interval: float = VOICE_SEND_INTERVAL):
        self._loop = loop
        self._ws = ws
        self.user_id = user_id
        self.display_name = display_name
        self._post_cb = post_cb  # async fn(display_name: str, text: str)
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._injector = SilenceInjector(max_silence_s=max_silence_s)
        self._send_interval = send_interval
        self._stop = asyncio.Event()
        self._sender_task = None
        self._reader_task = None
        self._last_audio = time.monotonic()

    # — event loop (called from the manager, which is fed off the recv thread) —
    def feed(self, mono16: bytes) -> None:
        if not mono16:
            return
        self._last_audio = time.monotonic()
        frame_s = (len(mono16) // 2) / TARGET_SAMPLE_RATE
        sil = self._injector.silence_before(time.monotonic(), frame_s)
        if sil:
            self._queue.put_nowait(sil)
        self._queue.put_nowait(mono16)

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_audio

    def start(self) -> None:
        self._sender_task = self._loop.create_task(self._sender_loop())
        self._reader_task = self._loop.create_task(self._reader_loop())

    async def _drain_once(self) -> bytes:
        parts = []
        while not self._queue.empty():
            parts.append(self._queue.get_nowait())
        return b"".join(parts)

    async def _sender_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._send_interval)
                pcm = await self._drain_once()
                if pcm:
                    await self._ws.send_bytes(pcm)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("voice: sender loop error (%s): %s", self.display_name, e)

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
                                await self._post_cb(self.display_name, line)
                    elif t == "done":
                        break
                    elif t == "error":
                        log.error("voice: whisper-live error: %s", data.get("message"))
                        break
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("voice: reader loop error (%s): %s", self.display_name, e)
        finally:
            if pending:
                line = " ".join(pending).strip()
                if line:
                    try:
                        await self._post_cb(self.display_name, line)
                    except Exception:
                        pass

    async def close(self) -> None:
        """Flush remaining audio, signal done, drain final commits, close WS."""
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
        try:
            await self._ws.close()
        except Exception:
            pass


class VoiceCallManager:
    """Fans Discord voice out to one whisper-live stream per active speaker and
    posts attributed, timestamped lines to a per-call thread.

    `feed(uid, name, mono16)` runs on the event loop (scheduled off the recv
    thread). The first frame from a new speaker opens a dedicated whisper-live
    socket (capacity-gated by VOICE_MAX_SPEAKERS); a reaper closes streams idle
    for VOICE_SPEAKER_IDLE_S to free slots. Idle speakers re-open transparently
    when they speak again.
    """

    def __init__(self, loop, http, ws_url, thread,
                 max_speakers: int = VOICE_MAX_SPEAKERS,
                 max_silence_s: float = VOICE_MAX_SILENCE_S):
        self._loop = loop
        self._http = http
        self._ws_url = ws_url
        self._thread = thread
        self._max_speakers = max_speakers
        self._max_silence_s = max_silence_s
        self._sessions: dict[int, VoiceUserSession] = {}
        self._creating: set[int] = set()
        self._pending: dict[int, list[bytes]] = {}
        self._capacity_notified = False
        self._stop = asyncio.Event()
        self._reaper_task = loop.create_task(self._reaper())

    # — recv thread —
    def submit(self, uid: int, name: str, mono16: bytes) -> None:
        if not mono16 or not uid:
            return
        try:
            self._loop.call_soon_threadsafe(self.feed, uid, name, mono16)
        except RuntimeError:
            pass  # loop closed mid-call

    # — event loop —
    def feed(self, uid: int, name: str, mono16: bytes) -> None:
        sess = self._sessions.get(uid)
        if sess is not None:
            sess.feed(mono16)
            return
        if uid in self._creating:
            self._pending.setdefault(uid, []).append(mono16)
            return
        if len(self._sessions) >= self._max_speakers:
            if not self._capacity_notified:
                self._capacity_notified = True
                self._loop.create_task(self._post(
                    f"⚠️ More than {self._max_speakers} people speaking at once — "
                    "some audio isn't being transcribed."))
            return
        self._creating.add(uid)
        self._pending.setdefault(uid, []).append(mono16)
        self._loop.create_task(self._open(uid, name))

    async def _open(self, uid: int, name: str) -> None:
        try:
            ws = await self._http.ws_connect(f"{self._ws_url}/ws-stream", timeout=30)
        except Exception as e:
            log.error("voice: failed to open whisper-live stream for %s: %s", name, e)
            self._creating.discard(uid)
            self._pending.pop(uid, None)
            return
        sess = VoiceUserSession(
            self._loop, ws, uid, name, self._post_line, self._max_silence_s)
        sess.start()
        self._sessions[uid] = sess
        self._creating.discard(uid)
        for frame in self._pending.pop(uid, []):
            sess.feed(frame)
        log.info("voice: opened stream for %s (%d active)", name, len(self._sessions))

    async def _post_line(self, name: str, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        await self._post(f"`[{ts}]` **{name}:** {text}")

    async def _post(self, body: str) -> None:
        try:
            await self._thread.send(body[:2000])
        except Exception as e:
            log.error("voice: failed to post to thread: %s", e)

    async def _reaper(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(5)
                for uid, sess in list(self._sessions.items()):
                    if sess.idle_seconds() > VOICE_SPEAKER_IDLE_S:
                        self._sessions.pop(uid, None)
                        self._loop.create_task(self._reap_close(sess))
        except asyncio.CancelledError:
            pass

    async def _reap_close(self, sess: VoiceUserSession) -> None:
        try:
            await sess.close()
            log.info("voice: closed idle stream for %s (%d active)",
                     sess.display_name, len(self._sessions))
        except Exception:
            pass

    async def close(self) -> None:
        self._stop.set()
        if self._reaper_task:
            self._reaper_task.cancel()
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for sess in sessions:
            try:
                await sess.close()
            except Exception:
                pass


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


# ── Transcript export + thread cleanup ──────────────────────────────────────
def _is_bot_call_thread(thread, bot_user_id: int) -> bool:
    """True only for per-call transcript threads WE created.

    Two guards so cleanup never touches a human-made thread: the thread must be
    owned by the bot AND carry the 🎙️ name prefix that `/transcribe-join` uses.
    """
    try:
        owner_ok = getattr(thread, "owner_id", None) == bot_user_id
        name_ok = (getattr(thread, "name", "") or "").startswith(_THREAD_NAME_PREFIX)
        return bool(owner_ok and name_ok)
    except Exception:
        return False


async def _build_transcript_file(thread):
    """Read a thread's full history (oldest-first) into a downloadable .txt File.

    Skips the consent notice and the control/marker lines so the export is just
    the attributed transcript. Returns (discord.File, line_count) or (None, 0).
    """
    import io

    import discord

    lines: list[str] = []
    try:
        async for msg in thread.history(limit=None, oldest_first=True):
            content = (msg.content or "").strip()
            if not content:
                continue
            if content.startswith("\U0001f534"):  # 🔴 consent notice
                continue
            if content in ("— transcription ended —",):
                continue
            if content.startswith("— ") and content.endswith(" —"):
                continue
            lines.append(content)
    except Exception as e:
        log.error("voice: transcript export read failed: %s", e)
        return None, 0
    if not lines:
        return None, 0
    header = (
        f"# Transcript — {getattr(thread, 'name', 'voice call')}\n"
        f"# exported {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n\n"
    )
    body = header + "\n".join(lines) + "\n"
    buf = io.BytesIO(body.encode("utf-8"))
    fname = f"transcript-{getattr(thread, 'id', 'call')}.txt"
    return discord.File(buf, filename=fname), len(lines)


async def _iter_call_threads(channel, bot_user_id: int):
    """Yield every bot-created per-call thread in `channel` (active + archived)."""
    seen: set[int] = set()
    for t in list(getattr(channel, "threads", []) or []):
        if _is_bot_call_thread(t, bot_user_id):
            seen.add(t.id)
            yield t
    try:
        async for t in channel.archived_threads(limit=None):
            if t.id in seen:
                continue
            if _is_bot_call_thread(t, bot_user_id):
                yield t
    except Exception as e:
        log.debug("voice: archived_threads scan failed: %s", e)


async def _purge_old_threads(channel, bot_user_id: int, older_than_days: float) -> int:
    """Delete bot-created call threads older than `older_than_days`.

    `older_than_days <= 0` deletes ALL bot-created call threads (the one-off
    bulk wipe). Returns the number of threads deleted.
    """
    cutoff = None
    if older_than_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    deleted = 0
    async for t in _iter_call_threads(channel, bot_user_id):
        created = getattr(t, "created_at", None)
        if cutoff is not None and created is not None and created > cutoff:
            continue
        try:
            await t.delete()
            deleted += 1
        except Exception as e:
            log.warning("voice: failed to delete thread %s: %s", getattr(t, "id", "?"), e)
    return deleted


async def voice_transcript_cleanup_loop(bot) -> None:
    """Background loop: delete per-call threads older than the retention window.

    No-op unless a dedicated transcript channel is configured and retention > 0
    (without a fixed channel we can't safely know which threads are ours).
    """
    if not VOICE_TRANSCRIPT_CHANNEL_ID or VOICE_TRANSCRIPT_RETENTION_DAYS <= 0:
        log.info(
            "voice: transcript auto-purge disabled "
            "(channel=%s retention_days=%s)",
            VOICE_TRANSCRIPT_CHANNEL_ID, VOICE_TRANSCRIPT_RETENTION_DAYS,
        )
        return
    await bot.wait_until_ready()
    interval_s = max(300.0, VOICE_CLEANUP_INTERVAL_H * 3600.0)
    while not bot.is_closed():
        try:
            channel = bot.get_channel(VOICE_TRANSCRIPT_CHANNEL_ID)
            if channel is not None:
                n = await _purge_old_threads(
                    channel, bot.user.id, VOICE_TRANSCRIPT_RETENTION_DAYS
                )
                if n:
                    log.info(
                        "voice: auto-purge removed %d transcript thread(s) older "
                        "than %s day(s)", n, VOICE_TRANSCRIPT_RETENTION_DAYS,
                    )
        except Exception as e:
            log.error("voice: cleanup loop error: %s", e)
        await asyncio.sleep(interval_s)


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

    # Per-guild live state: guild_id -> {"manager", "http", "thread"} so /leave
    # can tear everything down cleanly even across reconnects.
    _live: dict[int, dict] = {}

    # ── Per-thread control buttons (Export / Delete) ────────────────────────
    # Persistent view: static custom_ids + timeout=None so the buttons keep
    # working after a bot restart. The interaction's channel IS the thread the
    # button lives in, so no per-thread state needs to survive.
    class TranscriptControlView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=None)

        @discord.ui.button(
            label="Export", emoji="\U0001f4c4",
            style=discord.ButtonStyle.secondary,
            custom_id="voice:export",
        )
        async def export(self, interaction, button):
            await interaction.response.defer(ephemeral=True)
            thread = interaction.channel
            file, n = await _build_transcript_file(thread)
            if file is None:
                await interaction.followup.send(
                    "Nothing to export — this transcript is empty.", ephemeral=True
                )
                return
            await interaction.followup.send(
                f"📄 Transcript export — {n} line(s).", file=file, ephemeral=True
            )

        @discord.ui.button(
            label="Delete", emoji="\U0001f5d1\ufe0f",
            style=discord.ButtonStyle.danger,
            custom_id="voice:delete",
        )
        async def delete(self, interaction, button):
            perms = getattr(interaction.user, "guild_permissions", None)
            if not (perms and (perms.manage_threads or perms.manage_messages)):
                await interaction.response.send_message(
                    "You need **Manage Threads** to delete this transcript.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                "Deleting this transcript thread…", ephemeral=True
            )
            try:
                await interaction.channel.delete()
            except Exception as e:
                log.warning("voice: button delete failed: %s", e)

    # Register the persistent view so its buttons survive restarts.
    try:
        bot.add_view(TranscriptControlView())
    except Exception as e:
        log.debug("voice: add_view failed (already registered?): %s", e)

    # Background auto-purge of old transcript threads (no-op unless configured).
    bot.loop.create_task(voice_transcript_cleanup_loop(bot))

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

        # Create a per-call thread so each session's transcript is self-contained.
        # Falls back to posting in the channel if the bot lacks thread perms.
        thread = post_channel
        try:
            thread = await post_channel.create_thread(
                name=f"\U0001f399\ufe0f {channel.name} — {datetime.now():%Y-%m-%d %H:%M}",
                type=discord.ChannelType.public_thread,
            )
        except Exception as e:
            log.warning("voice: could not create thread (%s) — posting in channel", e)

        client = aiohttp.ClientSession()
        manager = VoiceCallManager(loop, client, WHISPER_LIVE_WS, thread)
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)

        def _on_packet(speaker, data) -> None:
            # Runs on the recv (router) thread: resample only, then hand off.
            try:
                pcm = getattr(data, "pcm", None)
                if not pcm:
                    return
                uid = getattr(speaker, "id", 0) or 0
                name = (getattr(speaker, "display_name", None)
                        or getattr(speaker, "name", None) or "Unknown")
                mono16 = resample_48k_stereo_to_16k_mono(pcm)
                if mono16 and uid:
                    manager.submit(uid, name, mono16)
            except Exception as e:  # never raise on the recv thread
                log.debug("voice: packet handler error: %s", e)

        vc.listen(voice_recv.BasicSink(_on_packet))
        _live[interaction.guild.id] = {"manager": manager, "http": client, "thread": thread}

        await thread.send(_CONSENT_NOTICE)
        await interaction.followup.send(
            f"Joined **{channel.name}** — live transcription active. "
            f"Transcript → {thread.mention if hasattr(thread, 'mention') else ''}",
            ephemeral=True,
        )
        log.info(
            "voice: joined %s (guild %s) → per-speaker streams, thread=%s",
            channel.name, interaction.guild.id, getattr(thread, "id", "?"),
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
        thread = None
        if state:
            thread = state.get("thread")
            try:
                await state["manager"].close()
            except Exception as e:
                log.debug("voice: manager close error: %s", e)
            try:
                await state["http"].close()
            except Exception:
                pass
        await vc.disconnect()
        # Post Export/Delete controls, then archive so it drops out of the
        # active-threads list (auto-purge deletes it after the retention window).
        if thread is not None and hasattr(thread, "send") and thread is not interaction.channel:
            try:
                await thread.send(
                    "— transcription ended — use the buttons to export or delete this thread.",
                    view=TranscriptControlView(),
                )
                await thread.edit(archived=True)
            except Exception as e:
                log.debug("voice: leave finalize failed: %s", e)
        await interaction.followup.send("Left the voice channel.", ephemeral=True)
        log.info("voice: left voice channel (guild %s)", interaction.guild.id)

    @bot.tree.command(
        name="transcribe-cleanup",
        description="Delete old voice-transcript threads (admin). older_than_days=0 wipes all.",
    )
    async def transcribe_cleanup(interaction: "discord.Interaction", older_than_days: float = -1.0):
        perms = getattr(interaction.user, "guild_permissions", None)
        if not (perms and (perms.manage_threads or perms.manage_messages)):
            await interaction.response.send_message(
                "You need **Manage Threads** to run cleanup.", ephemeral=True
            )
            return
        # Default (unset) → use the configured retention window.
        days = VOICE_TRANSCRIPT_RETENTION_DAYS if older_than_days < 0 else older_than_days
        channel = _resolve_post_channel(interaction)
        await interaction.response.defer(ephemeral=True)
        try:
            n = await _purge_old_threads(channel, bot.user.id, days)
        except Exception as e:
            log.error("voice: /transcribe-cleanup failed: %s", e)
            await interaction.followup.send(f"Cleanup failed: {e}", ephemeral=True)
            return
        scope = "all" if days <= 0 else f"older than {days:g} day(s)"
        await interaction.followup.send(
            f"🧹 Deleted **{n}** transcript thread(s) ({scope}).", ephemeral=True
        )
        log.info("voice: /transcribe-cleanup removed %d thread(s) (scope=%s)", n, scope)

    log.info(
        "voice: /transcribe-join + /transcribe-leave + /transcribe-cleanup registered"
    )
    return True
