"""Pitch-based speaker gender estimation.

`estimate_speaker_genders` runs autocorrelation pitch detection on each
speaker's segments and labels them M/F based on median F0. `apply_gender_labels`
prefixes the resulting label onto each segment's `speaker` field.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("whisper-ui")


def estimate_speaker_genders(audio: np.ndarray, segments: list[dict],
                             sample_rate: int = 16000) -> dict[str, str]:
    """Estimate gender for each speaker based on median fundamental frequency (F0).

    Uses autocorrelation pitch detection on each speaker's audio segments.
    Male: median F0 < 165 Hz. Female: ≥ 165 Hz.
    Returns dict: {speaker_label: "M" | "F" | "?"}.
    """
    from scipy.signal import correlate  # local import — scipy is heavy

    speaker_samples: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        start_sample = int(seg.get("start", 0) * sample_rate)
        end_sample = int(seg.get("end", 0) * sample_rate)
        if end_sample <= start_sample:
            continue
        chunk = audio[start_sample:min(end_sample, len(audio))]
        if len(chunk) < sample_rate * 0.1:  # skip < 100 ms
            continue
        speaker_samples.setdefault(speaker, []).append(chunk)

    genders: dict[str, str] = {}
    for speaker, chunks in speaker_samples.items():
        pitches: list[float] = []
        for chunk in chunks[:10]:  # sample up to 10 segments per speaker
            frame_len = min(int(sample_rate * 0.05), len(chunk))
            mid = len(chunk) // 2
            frame = chunk[mid - frame_len // 2: mid + frame_len // 2]
            if len(frame) < 200:
                continue
            frame = frame - frame.mean()
            corr = correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]  # positive lags only
            min_lag = int(sample_rate / 500)   # max freq 500 Hz
            max_lag = int(sample_rate / 60)    # min freq 60 Hz
            if max_lag > len(corr):
                continue
            search = corr[min_lag:max_lag]
            if len(search) == 0:
                continue
            peak_idx = search.argmax() + min_lag
            if corr[peak_idx] > 0.2 * corr[0]:  # confidence threshold
                pitches.append(sample_rate / peak_idx)

        if pitches:
            median_f0 = float(np.median(pitches))
            genders[speaker] = "F" if median_f0 >= 165 else "M"
            log.debug(f"  Speaker {speaker}: median F0={median_f0:.0f} Hz → {genders[speaker]}")
        else:
            genders[speaker] = "?"
    return genders


def apply_gender_labels(segments: list[dict], genders: dict[str, str]) -> list[dict]:
    """Rename speaker labels with a gender prefix: SPEAKER_00 → M-SPEAKER_00."""
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker in genders and genders[speaker] != "?":
            new_label = f"{genders[speaker]}-{speaker}"
            seg["speaker"] = new_label
            for w in seg.get("words", []):
                if w.get("speaker") == speaker:
                    w["speaker"] = new_label
    return segments
