"""Resident faster-whisper model for streaming transcription.

Keeps the model loaded permanently (the live service is dedicated; VRAM is
not shared with the batch service). Serialises GPU calls via asyncio.Lock so
multiple concurrent sessions share one model safely.
"""
from __future__ import annotations

import asyncio
import os
import re
from collections import deque

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

    # ── Streaming (LocalAgreement) ────────────────────────────────────────────
    def new_session(self, **kw) -> "OnlineSession":
        """Create a per-connection streaming session bound to this model."""
        return OnlineSession(self._model, **kw)

    async def session_process(self, session: "OnlineSession") -> tuple[str, str]:
        """Run one inference pass over the session's growing buffer on the GPU
        (serialised on the shared lock). Returns (committed_text, partial_text).
        """
        loop = asyncio.get_event_loop()
        async with self._lock:
            return await loop.run_in_executor(None, session.process)

    async def session_finish(self, session: "OnlineSession") -> str:
        """Flush the unconfirmed tail as final when the stream ends."""
        loop = asyncio.get_event_loop()
        async with self._lock:
            return await loop.run_in_executor(None, session.finish)


# ── LocalAgreement-2 streaming ──────────────────────────────────────────────────
# Faithful port of the UFAL `whisper_streaming` policy: re-transcribe a growing
# audio buffer; only commit words that two consecutive hypotheses agree on (the
# stable prefix). Transient hallucinations don't survive a second pass, so they
# are never committed. word_timestamps drive buffer trimming.

Word = tuple[float, float, str]  # (start, end, text) — text keeps its leading space


def _norm(w: str) -> str:
    """Normalise a word for agreement comparison: lowercase, strip surrounding
    punctuation/space. The committed output keeps the original casing/punct;
    only the equality test is normalised so capitalisation or a trailing comma
    drifting between passes doesn't block an otherwise-stable word."""
    return re.sub(r"[^\w']", "", w.lower())


class HypothesisBuffer:
    """Tracks the previous hypothesis tail and commits the longest common
    prefix shared with the newest hypothesis."""

    def __init__(self) -> None:
        self.committed_in_buffer: list[Word] = []
        self.buffer: list[Word] = []  # previous hypothesis, not yet agreed
        self.new: list[Word] = []
        self.last_committed_time: float = 0.0

    def insert(self, words: list[Word], offset: float) -> None:
        """Stage `words` (timestamps relative to buffer start) shifted by
        `offset` into absolute time, dropping anything already committed and
        de-duplicating words that overlap the committed tail."""
        shifted = [(a + offset, b + offset, t) for (a, b, t) in words]
        self.new = [w for w in shifted if w[0] > self.last_committed_time - 0.1]
        if not self.new:
            return
        a = self.new[0][0]
        if abs(a - self.last_committed_time) < 1.0 and self.committed_in_buffer:
            cn = len(self.committed_in_buffer)
            nn = len(self.new)
            # Drop the longest committed-tail / new-head n-gram overlap (n≤5)
            # so repeated words at the seam aren't emitted twice.
            for i in range(1, min(cn, nn, 5) + 1):
                c = " ".join(self.committed_in_buffer[-j][2] for j in range(i, 0, -1))
                tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                if c == tail:
                    del self.new[:i]
                    break

    def flush(self) -> list[Word]:
        """Commit the longest common prefix of the new vs previous hypothesis."""
        commit: list[Word] = []
        while self.new and self.buffer:
            if _norm(self.new[0][2]) == _norm(self.buffer[0][2]):
                w = self.new.pop(0)
                commit.append(w)
                self.last_committed_time = w[1]
                self.buffer.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, t: float) -> None:
        """Drop committed words ending at/before `t` after a buffer trim."""
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= t:
            self.committed_in_buffer.pop(0)


class OnlineSession:
    """One streaming connection. Accumulates audio, re-transcribes the growing
    buffer, commits stable words, and trims the buffer at committed boundaries.
    All `process`/`finish` calls run under the model lock (see
    StreamingTranscriber.session_*)."""

    def __init__(
        self,
        model,
        min_chunk_s: float = 1.0,
        trim_s: float = 8.0,
        tail_silence_s: float = 0.6,
        silence_rms: float = 0.006,
        min_silence_ms: int = 300,
    ) -> None:
        self._model = model
        self._min_samples = int(SAMPLE_RATE * min_chunk_s)
        self._trim_s = trim_s
        self._tail_silence_s = tail_silence_s
        self._silence_rms = silence_rms
        self._min_silence_ms = min_silence_ms
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0.0  # absolute time of audio[0]
        self.hyp = HypothesisBuffer()
        self.committed: list[Word] = []
        # Incoming PCM is enqueued from the WS receive coroutine (main thread)
        # and drained inside process() which runs in a worker thread. A deque
        # is thread-safe under the GIL, so all numpy-buffer mutation stays on
        # the worker thread — no torn reads/writes of self.audio.
        self._inbox: deque[bytes] = deque()

    def insert_audio(self, pcm_bytes: bytes) -> None:
        self._inbox.append(pcm_bytes)

    def _drain(self) -> None:
        if not self._inbox:
            return
        parts = []
        while self._inbox:
            parts.append(self._inbox.popleft())
        a = np.frombuffer(b"".join(parts), dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.append(self.audio, a)

    def _prompt(self) -> str | None:
        text = "".join(w[2] for w in self.committed)[-200:].strip()
        return text or None

    def _transcribe(self) -> list[Word]:
        segments, _info = self._model.transcribe(
            self.audio,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self._min_silence_ms},
            initial_prompt=self._prompt(),
        )
        words: list[Word] = []
        for seg in segments:
            sw = getattr(seg, "words", None)
            if sw:
                words.extend((w.start, w.end, w.word) for w in sw)
            elif seg.text.strip():
                words.append((seg.start, seg.end, seg.text))
        return words

    def process(self) -> tuple[str, str, bool]:
        """One pass. Returns (committed_text, partial_text, end_of_utterance).
        `end_of_utterance` is True when a trailing-silence pause closed an
        utterance — the caller uses it to insert a line break."""
        self._drain()
        if len(self.audio) < self._min_samples:
            return "", self._partial(), False
        self.hyp.insert(self._transcribe(), self.offset)
        committed = list(self.hyp.flush())
        silent = self._tail_is_silent()
        if silent and self.hyp.buffer:
            # End-of-utterance: a trailing-silence pause means whisper saw the
            # whole utterance, so the unconfirmed tail is safe to commit.
            committed.extend(self.hyp.buffer)
            self.hyp.buffer = []
        self.committed.extend(committed)
        if silent:
            self._finalize_reset()  # drop the utterance's audio; start fresh
        elif len(self.audio) / SAMPLE_RATE > self._trim_s and self.committed:
            self._trim(self.committed[-1][1])
        return "".join(w[2] for w in committed).strip(), self._partial(), silent

    def _tail_is_silent(self) -> bool:
        n = int(self._tail_silence_s * SAMPLE_RATE)
        if len(self.audio) < n:
            return False
        tail = self.audio[-n:]
        return float(np.sqrt(np.mean(tail * tail))) < self._silence_rms

    def _finalize_reset(self) -> None:
        """Utterance boundary: keep only a short trailing window and reset the
        agreement state so the next utterance starts clean."""
        keep = int(0.3 * SAMPLE_RATE)
        if len(self.audio) > keep:
            self.offset += (len(self.audio) - keep) / SAMPLE_RATE
            self.audio = self.audio[-keep:]
        self.hyp.buffer = []
        self.hyp.committed_in_buffer = []
        self.hyp.last_committed_time = self.offset

    def _trim(self, t: float) -> None:
        cut = int((t - self.offset) * SAMPLE_RATE)
        if 0 < cut < len(self.audio):
            self.audio = self.audio[cut:]
            self.offset = t
            self.hyp.pop_committed(t)

    def _partial(self) -> str:
        return "".join(w[2] for w in self.hyp.buffer).strip()

    def finish(self) -> str:
        """Commit the unconfirmed tail when the stream ends."""
        self._drain()
        rem = self.hyp.buffer
        self.hyp.buffer = []
        self.committed.extend(rem)
        return "".join(w[2] for w in rem).strip()
