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
