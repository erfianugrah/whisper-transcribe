"""Pure rendering helpers — timestamps, HTML/plain transcript output.

Stateful: `_speaker_index_map` is module-level and persists across calls so
speakers consistently get the same colour within a session. Call
`reset_speaker_colors()` at the start of a new transcription run.
"""

from __future__ import annotations

import html as _html


# ─── Timestamp formatters ─────────────────────────────────────────────────────

def format_timestamp_display(seconds: float) -> str:
    """Short timestamp for transcript display: MM:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ─── Speaker colour mapping ───────────────────────────────────────────────────

_speaker_index_map: dict[str, int] = {}


def reset_speaker_colors() -> None:
    """Clear the speaker→colour-index map. Call at start of a new run."""
    _speaker_index_map.clear()


def speaker_class(speaker_label: str) -> str:
    """Map a speaker label to a CSS class (spk-0 through spk-7, cycling)."""
    if speaker_label not in _speaker_index_map:
        _speaker_index_map[speaker_label] = len(_speaker_index_map) % 8
    return f"spk-{_speaker_index_map[speaker_label]}"


# ─── Transcript rendering ─────────────────────────────────────────────────────

def format_transcript_html(segments: list[dict], has_speakers: bool = False) -> str:
    """Convert segments to HTML with timestamps and optional speaker colours."""
    if not segments:
        return "<div class='transcript-empty'>No segments</div>"
    lines = []
    for seg in segments:
        ts = format_timestamp_display(seg.get("start", 0))
        text = _html.escape(seg.get("text", "").strip())
        ts_span = f"<span class='transcript-ts'>[{ts}]</span>"
        if has_speakers:
            speaker = seg.get("speaker", "?")
            cls = speaker_class(speaker)
            spk_span = (
                f"<span class='transcript-speaker {cls}'>"
                f"{_html.escape(speaker)}</span>"
            )
            lines.append(f"<div class='transcript-line'>{ts_span}{spk_span}{text}</div>")
        else:
            lines.append(f"<div class='transcript-line'>{ts_span}{text}</div>")
    return "\n".join(lines)


def format_transcript_plain(segments: list[dict], has_speakers: bool = False) -> str:
    """Convert segments to plain text for copy/export."""
    lines = []
    for seg in segments:
        ts = format_timestamp_display(seg.get("start", 0))
        text = seg.get("text", "").strip()
        if has_speakers:
            speaker = seg.get("speaker", "?")
            lines.append(f"[{ts}] [{speaker}] {text}")
        else:
            lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


def plain_html(text: str) -> str:
    """Wrap plain text in a pre-formatted div for intermediate progress views."""
    if not text:
        return ""
    escaped = _html.escape(text)
    return (
        f"<div style='white-space:pre-wrap;font-size:0.8rem;line-height:1.5'>"
        f"{escaped}</div>"
    )
