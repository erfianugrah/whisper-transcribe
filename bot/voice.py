"""Discord voice-call live transcription.

See docs/plans/2026-06-18-discord-voice-transcription.md for the full design.

This module is import-safe even when `discord-ext-voice-recv` is absent (e.g.
the test stub layer): the extension import is guarded and exposed via
`VOICE_RECV_AVAILABLE`. The pure audio helpers (`resample_48k_stereo_to_16k_mono`,
`mix_streams`, `SilenceInjector`) depend only on numpy and are unit-tested.

Phase 0 (this file's current scope): prove the receive path — connect with
`VoiceRecvClient`, attach a `BasicSink` that logs packets. No whisper-live
streaming yet (Phase 1). Everything is gated behind `VOICE_TRANSCRIBE_ENABLED`.
"""
from __future__ import annotations

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


# ── Phase 0: connect + packet-log (no streaming yet) ────────────────────────
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

    import discord

    @bot.tree.command(
        name="transcribe-join",
        description="Join your voice channel and live-transcribe it (Phase 0: log only)",
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
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)

        # Phase 0 sink: rate-limited per-user packet log to prove the path.
        _last_log: dict[int, float] = {}

        def _on_packet(speaker, data) -> None:
            try:
                uid = getattr(speaker, "id", 0) or 0
                now = time.monotonic()
                if now - _last_log.get(uid, 0.0) >= 2.0:
                    _last_log[uid] = now
                    pcm = getattr(data, "pcm", b"") or b""
                    log.info("voice: packet from %s (%d bytes pcm)", speaker, len(pcm))
            except Exception as e:  # never raise on the recv thread
                log.debug("voice: packet log error: %s", e)

        vc.listen(voice_recv.BasicSink(_on_packet))
        await channel.send(_CONSENT_NOTICE)
        await interaction.followup.send(
            f"Joined **{channel.name}** — transcription path active (Phase 0: logging packets).",
            ephemeral=True,
        )
        log.info("voice: joined %s (guild %s)", channel.name, interaction.guild.id)

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
        try:
            vc.stop_listening()
        except Exception:
            pass
        await vc.disconnect()
        await interaction.response.send_message("Left the voice channel.", ephemeral=True)
        log.info("voice: left voice channel (guild %s)", interaction.guild.id)

    log.info("voice: /transcribe-join + /transcribe-leave registered")
    return True
