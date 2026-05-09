"""Pure segment manipulation — word-list to segment, sentence/speaker splits.

Operates on whisperX-style segment dicts:
    {"start": float, "end": float, "text": str, "speaker"?: str,
     "words": [{"word": str, "start": float, "end": float, "speaker"?: str}, ...]}
"""

from __future__ import annotations


# Unicode ranges for scripts that don't insert spaces between words.
# Used to decide whether to space-join word tokens or concatenate.
_NO_SPACE_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7A3),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),   # Halfwidth/fullwidth forms
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1000, 0x109F),   # Myanmar
    (0x0F00, 0x0FFF),   # Tibetan
    (0x1780, 0x17FF),   # Khmer
)


def is_no_space_script(s: str) -> bool:
    """True if `s` contains a character from a no-space-between-words script."""
    for ch in s:
        cp = ord(ch)
        for lo, hi in _NO_SPACE_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def words_to_segment(words: list[dict], speaker: str) -> dict:
    """Build a segment dict from a list of word dicts.

    Joining strategy is per-segment, decided by the first non-empty token:
    - Latin / Cyrillic / Arabic / etc. → space-join.
    - CJK / Thai / Lao / Khmer / etc. → concatenate (no separator).
    """
    raw_tokens = [w.get("word", "").strip() for w in words]
    tokens = [t for t in raw_tokens if t]
    sep = "" if tokens and is_no_space_script(tokens[0]) else " "
    text = sep.join(tokens)
    starts = [w["start"] for w in words if "start" in w]
    ends = [w["end"] for w in words if "end" in w]
    return {
        "start": starts[0] if starts else words[0].get("start", 0),
        "end": ends[-1] if ends else words[-1].get("end", 0),
        "text": text,
        "speaker": speaker,
        "words": words,
    }


def split_at_sentences(words: list[dict], speaker: str, max_words: int) -> list[dict]:
    """Split a word list at sentence-ending punctuation if it exceeds max_words."""
    if len(words) <= max_words:
        return [words_to_segment(words, speaker)]

    segments: list[dict] = []
    current: list[dict] = []
    sentence_enders = {".", "!", "?", "。", "！", "？"}
    for w in words:
        current.append(w)
        text = w.get("word", "").strip()
        # Split if we hit a sentence ender and have enough words
        if len(current) >= 8 and text and text[-1] in sentence_enders:
            segments.append(words_to_segment(current, speaker))
            current = []
    if current:
        segments.append(words_to_segment(current, speaker))
    return segments


def split_segments_by_speaker(segments: list[dict],
                              max_segment_words: int = 40) -> list[dict]:
    """Split segments where the speaker changes mid-segment (using word-level
    labels). Also splits long single-speaker segments at sentence boundaries.
    """
    new_segments: list[dict] = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue

        # Group consecutive words by speaker, propagating the last known
        # speaker to unlabeled words instead of inserting spurious "?" groups.
        seg_speaker = seg.get("speaker", "?")
        groups: list[tuple[str, list[dict]]] = []
        current_speaker = None
        current_words: list[dict] = []
        for w in words:
            speaker = w.get("speaker") or seg_speaker
            if speaker != current_speaker and current_words:
                groups.append((current_speaker, current_words))
                current_words = []
            current_speaker = speaker
            current_words.append(w)
        if current_words:
            groups.append((current_speaker, current_words))

        # Merge any remaining "?" groups into their neighbour
        merged: list[tuple[str, list[dict]]] = []
        for speaker, grp_words in groups:
            if speaker == "?" and merged:
                prev_speaker, prev_words = merged[-1]
                merged[-1] = (prev_speaker, prev_words + grp_words)
            else:
                merged.append((speaker, grp_words))
        groups = merged

        # Build new segments from groups; further-split long ones at sentences.
        for speaker, group_words in groups:
            new_segments.extend(split_at_sentences(group_words, speaker, max_segment_words))
    return new_segments
