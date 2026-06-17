import sys
import os
import re
import time
import logging
import tempfile
import traceback
import shutil
import json
import threading
import subprocess
import asyncio
import hashlib
import uuid

# -- Logging setup -------------------------------------------------------------
# Accepts "1", "true", "yes", "on" (any case) for opt-in; everything else
# (including unset) opts out of debug.
DEBUG_MODE = os.environ.get("DEBUG_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("whisper-ui")
log.setLevel(logging.DEBUG)

# Third-party DEBUG-level log spam -- these libraries emit per-chunk, per-frame,
# or per-request debug logs that drown out our own logs. Only suppress to WARNING.
for noisy in ("PIL", "python_multipart", "python_multipart.multipart",
              "multipart", "asyncio", "watchfiles",
              "httpcore", "httpcore.http11", "httpcore.connection",
              "httpx", "filelock", "faster_whisper",
              "matplotlib", "matplotlib.font_manager",
              "fsspec", "fsspec.local", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import warnings
# PyTorch performance hints -- not actionable in this context
warnings.filterwarnings("ignore", category=UserWarning, message="TensorFloat-32")
warnings.filterwarnings("ignore", category=UserWarning, message="std\\(\\): degrees of freedom")

if not DEBUG_MODE:
    for noisy in ("httpcore", "httpx", "urllib3", "gradio", "starlette"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
else:
    log.info("DEBUG_MODE is ON -- verbose third-party logging enabled")

log.info("=" * 60)
log.info("WhisperX Transcription UI starting")
log.info("=" * 60)

# -- Python / env info ---------------------------------------------------------
log.info(f"Python: {sys.version}")
log.info(f"CWD: {os.getcwd()}")
log.info(f"ENV NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unset')}")
log.info(f"ENV CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

# -- GPU detection -------------------------------------------------------------
import torch

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
GPU_INFO_STR = "CPU mode (no GPU detected)"

if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    # Use torch directly — no subprocess to nvidia-smi. The CUDA runtime is
    # already loaded (torch.cuda.is_available() returned True), so name +
    # total memory are free property reads.
    try:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_mb = props.total_memory // (1024 * 1024)
        GPU_INFO_STR = (
            f"{gpu_name}  |  {vram_mb // 1024} GB VRAM  |  "
            f"whisperX (faster-whisper + wav2vec2 alignment)  |  float16"
        )
        log.info(f"GPU 0: {gpu_name} ({vram_mb} MB VRAM)")
    except Exception as e:
        log.warning(f"Failed to read GPU properties via torch: {e}")
        GPU_INFO_STR = "CUDA GPU detected  |  whisperX  |  float16"
else:
    log.info("No CUDA GPU detected, falling back to CPU mode")

log.info(f"Selected device: {DEVICE}, compute_type: {COMPUTE_TYPE}")

# -- Import whisperx ----------------------------------------------------------
log.info("Importing whisperx...")
t0 = time.time()
import whisperx
log.info(f"whisperx imported in {time.time()-t0:.2f}s")

# -- Import gradio -------------------------------------------------------------
log.info("Importing gradio...")
t0 = time.time()
import gradio as gr
log.info(f"Gradio {gr.__version__} imported in {time.time()-t0:.2f}s")

# -- Diarization availability --------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
DIARIZATION_AVAILABLE = bool(HF_TOKEN)
if not HF_TOKEN:
    log.info("HF_TOKEN not set -- speaker diarization disabled")
else:
    log.info("HF_TOKEN set -- speaker diarization available")

# -- Model management ---------------------------------------------------------
whisper_model = None
current_model_key = None  # (model_name, hotwords, initial_prompt, suppress_numerals) for cache invalidation
diarize_model = None
# Bounded LRU for wav2vec2 alignment models. Each model is ~360 MB on GPU; on
# a multilingual server transcribing dozens of languages, an unbounded dict
# would steadily creep VRAM until the next idle unload. Cap is configurable.
import collections
ALIGN_MODEL_CACHE_SIZE = int(os.environ.get("ALIGN_MODEL_CACHE_SIZE", "4"))
align_model_cache: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()

# Idle unload: free VRAM after MODEL_IDLE_TIMEOUT seconds of no transcription.
# Set to 0 to disable unloading entirely (keeps WhisperX resident; saves the
# 5-15s cold-load on the next request — worth it when VRAM headroom allows).
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))  # 5 min default; 0 = never
_last_activity = time.time()
_idle_timer = None
_idle_lock = threading.Lock()


def _unload_models():
    """Release all GPU models to free VRAM.

    Safety: never unloads while a transcription is running. The timer fires
    `MODEL_IDLE_TIMEOUT+5` after the LAST `_reset_idle_timer` call (which
    happens at the start of a transcription); for a long-running transcription
    that exceeds the timeout, the timer would otherwise unload the model
    mid-job. The lock check below prevents that.
    """
    global whisper_model, current_model_key, diarize_model, align_model_cache
    with _idle_lock:
        if _transcription_lock.locked():
            # Reschedule for after MODEL_IDLE_TIMEOUT past the current job's end.
            # _reset_idle_timer is called in the transcription's `finally`, so
            # we just bail and trust that path to set up the next firing.
            log.debug("Idle timer fired but a transcription is running — skipping unload")
            return
        elapsed = time.time() - _last_activity
        if elapsed < MODEL_IDLE_TIMEOUT:
            return  # Activity happened since timer was set
        if whisper_model is None and diarize_model is None and not align_model_cache:
            return  # Nothing loaded
        log.info(f"Idle for {elapsed:.0f}s — unloading models to free VRAM")
        whisper_model = None
        current_model_key = None
        diarize_model = None
        align_model_cache = {}
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        log.info("Models unloaded, VRAM freed")


def _reset_idle_timer():
    """Reset the idle timer. Called at start AND end of transcription.

    No-op when MODEL_IDLE_TIMEOUT <= 0 — keeps models resident forever.
    """
    global _last_activity, _idle_timer
    _last_activity = time.time()
    if MODEL_IDLE_TIMEOUT <= 0:
        return
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(MODEL_IDLE_TIMEOUT + 5, _unload_models)
    _idle_timer.daemon = True
    _idle_timer.start()


def load_whisper(model_name, hotwords=None, initial_prompt=None, suppress_numerals=False):
    """Load (or reuse) a whisperX transcription model."""
    global whisper_model, current_model_key
    # Map friendly names to actual model IDs
    actual_name = model_name
    if model_name == "large":
        actual_name = "large-v3"
    elif model_name == "turbo":
        actual_name = "large-v3-turbo"

    cache_key = (actual_name, hotwords or "", initial_prompt or "", suppress_numerals)
    if cache_key != current_model_key:
        log.info(f"Loading whisperX model '{actual_name}' on {DEVICE} (compute_type={COMPUTE_TYPE})...")
        if hotwords:
            log.info(f"  Hotwords: {hotwords[:100]}...")
        if initial_prompt:
            log.info(f"  Initial prompt: {initial_prompt[:100]}...")
        if suppress_numerals:
            log.info(f"  Suppress numerals: enabled")
        t0 = time.time()
        asr_options = {}
        if hotwords:
            asr_options["hotwords"] = hotwords
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt
        if suppress_numerals:
            asr_options["suppress_numerals"] = True
        whisper_model = whisperx.load_model(
            actual_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None,
            asr_options=asr_options if asr_options else None,
        )
        elapsed = time.time() - t0
        log.info(f"  WhisperX model loaded in {elapsed:.2f}s")
        current_model_key = cache_key
    else:
        log.info(f"WhisperX model '{actual_name}' already loaded, reusing")
    return whisper_model


def load_align_model(language_code):
    """Load (or reuse) a wav2vec2 alignment model for a given language.

    Bounded by ALIGN_MODEL_CACHE_SIZE — when the cache is full, the
    least-recently-used entry is evicted to free its VRAM. CUDA frees lazily
    so we hint with empty_cache() after eviction.
    """
    if language_code in align_model_cache:
        # Move to MRU end
        align_model_cache.move_to_end(language_code)
        return align_model_cache[language_code]
    log.info(f"Loading alignment model for '{language_code}'...")
    t0 = time.time()
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=DEVICE,
    )
    align_model_cache[language_code] = (model_a, metadata)
    # Evict LRU if over capacity. Capacity ≤ 0 disables the bound.
    if ALIGN_MODEL_CACHE_SIZE > 0:
        while len(align_model_cache) > ALIGN_MODEL_CACHE_SIZE:
            evicted_lang, _evicted = align_model_cache.popitem(last=False)
            log.info(f"  Evicted alignment model for '{evicted_lang}' (LRU)")
        if DEVICE == "cuda" and len(align_model_cache) >= ALIGN_MODEL_CACHE_SIZE:
            torch.cuda.empty_cache()
    log.info(f"  Alignment model loaded in {time.time()-t0:.2f}s")
    return model_a, metadata


def load_diarization():
    """Load (or reuse) the whisperX diarization pipeline."""
    global diarize_model
    if diarize_model is not None:
        log.info("Diarization pipeline already loaded, reusing")
        return diarize_model
    if not HF_TOKEN:
        return None
    log.info("Loading whisperX diarization pipeline...")
    t0 = time.time()
    try:
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(
            token=HF_TOKEN,
            device=DEVICE,
        )
        log.info(f"  Diarization pipeline loaded in {time.time()-t0:.2f}s")
    except Exception as e:
        log.error(f"  Failed to load diarization pipeline: {e}")
        traceback.print_exc()
        diarize_model = None
    return diarize_model


# -- Temp file cleanup ---------------------------------------------------------
_previous_subtitle = None

# Gradio writes uploads under GRADIO_TEMP_DIR (default {tempdir}/gradio).
# Honour the env so users who relocate the temp dir get correct cleanup.
GRADIO_TMP_DIR = os.environ.get("GRADIO_TEMP_DIR") or os.path.join(
    tempfile.gettempdir(), "gradio"
)
# Trailing separator simplifies prefix matching in cleanup_upload below.
_GRADIO_TMP_PREFIX = os.path.normpath(GRADIO_TMP_DIR) + os.sep


def cleanup_upload(file_path):
    """Remove the uploaded file and its parent gradio temp dir if empty."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
        parent = os.path.dirname(file_path)
        if parent and (
            os.path.normpath(parent).startswith(_GRADIO_TMP_PREFIX.rstrip(os.sep))
        ) and not os.listdir(parent):
            os.rmdir(parent)
        log.info(f"Cleaned up upload: {file_path}")
    except Exception as e:
        log.warning(f"Failed to clean up {file_path}: {e}")


def cleanup_stale_gradio_tmp():
    """Remove old gradio temp directories on startup.

    Only touches entries older than `STALE_TMP_AGE_SECONDS` to avoid trashing
    in-flight uploads from another instance sharing the same tmp root.
    """
    gradio_tmp = GRADIO_TMP_DIR
    age_threshold = int(os.environ.get("STALE_TMP_AGE_SECONDS", "3600"))
    cutoff = time.time() - age_threshold
    if not os.path.isdir(gradio_tmp):
        return
    count = 0
    skipped = 0
    for entry in os.listdir(gradio_tmp):
        path = os.path.join(gradio_tmp, entry)
        try:
            if os.path.getmtime(path) > cutoff:
                skipped += 1
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            count += 1
        except Exception as e:
            log.warning(f"Failed to clean {path}: {e}")
    if count or skipped:
        log.info(f"Cleaned {count} stale gradio temp entries (skipped {skipped} recent)")


cleanup_stale_gradio_tmp()


# -- History -------------------------------------------------------------------
# Path is configurable so the same image can run with bind-mounted /data, a
# named volume, or a different writable mount.
HISTORY_FILE = os.environ.get("HISTORY_FILE", "/data/history.json")
_history_lock = threading.Lock()


def _load_history() -> list[dict]:
    """Load transcription history from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history_entry(entry: dict):
    """Append an entry to the history file (thread-safe).

    Concurrent UI + API requests would otherwise clobber each other on the
    read-modify-write cycle.
    """
    parent = os.path.dirname(HISTORY_FILE)
    with _history_lock:
        history = _load_history()
        history.insert(0, entry)
        # Keep last 50 entries
        history = history[:50]
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Failed to save history: {e}")


def _format_history_html() -> str:
    """Render history as an HTML table.

    All dynamic fields are passed through html.escape — filenames and
    user-supplied metadata could otherwise inject script.
    """
    # NB: html_module is imported below at top-level (see module imports).
    # Forward reference is fine — this function is only called from Gradio
    # event handlers, well after module load.
    history = _load_history()
    if not history:
        return "<div style='opacity:0.4;font-style:italic;padding:0.5rem'>No transcription history yet.</div>"
    esc = html_module.escape
    rows = []
    for entry in history[:20]:
        ts = esc(str(entry.get("timestamp", "?")))
        fname = esc(str(entry.get("filename", "?")))
        duration = esc(str(entry.get("duration_str", "?")))
        lang = esc(str(entry.get("language", "?")))
        speakers = entry.get("speakers", "")
        speed = esc(str(entry.get("speed", "")))
        spk_badge = f" <small>({esc(str(speakers))} spk)</small>" if speakers else ""
        rows.append(
            f"<tr><td style='opacity:0.5;white-space:nowrap'>{ts}</td>"
            f"<td>{fname}</td>"
            f"<td>{duration}</td>"
            f"<td>{lang}{spk_badge}</td>"
            f"<td>{speed}</td></tr>"
        )
    return f"""<table style='width:100%;font-size:0.78rem;border-collapse:collapse'>
<thead><tr style='opacity:0.5;text-align:left'><th>Time</th><th>File</th><th>Duration</th><th>Lang</th><th>Speed</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>"""


# -- Default batch size based on VRAM -----------------------------------------
# Heuristic auto-detection; override with WHISPER_DEFAULT_BATCH_SIZE env var
# (any positive int) to skip the heuristic entirely. Per-request batch_size
# from the UI / API still overrides this default.
_BATCH_OVERRIDE = os.environ.get("WHISPER_DEFAULT_BATCH_SIZE", "").strip()
if _BATCH_OVERRIDE:
    DEFAULT_BATCH_SIZE = int(_BATCH_OVERRIDE)
    log.info(f"Using WHISPER_DEFAULT_BATCH_SIZE override: {DEFAULT_BATCH_SIZE}")
else:
    DEFAULT_BATCH_SIZE = 4  # conservative CPU default
    if DEVICE == "cuda":
        try:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            # Tested on RTX 5090 / A100 / 4090 / 3090 / 3080 / 3060.
            # If you hit OOM, set WHISPER_DEFAULT_BATCH_SIZE explicitly.
            if vram_gb >= 24:
                DEFAULT_BATCH_SIZE = 64
            elif vram_gb >= 16:
                DEFAULT_BATCH_SIZE = 24
            elif vram_gb >= 10:
                DEFAULT_BATCH_SIZE = 16
            elif vram_gb >= 6:
                DEFAULT_BATCH_SIZE = 8
            else:
                DEFAULT_BATCH_SIZE = 4
            log.info(f"Auto-selected batch_size={DEFAULT_BATCH_SIZE} for {vram_gb:.0f} GB VRAM")
        except Exception as e:
            DEFAULT_BATCH_SIZE = 8
            log.warning(f"VRAM detection failed ({e}), defaulting batch_size={DEFAULT_BATCH_SIZE}")


# -- Gender estimation from pitch (F0) -----------------------------------------
import numpy as np


def estimate_speaker_genders(audio, segments, sample_rate=16000):
    """Estimate gender for each speaker based on median fundamental frequency (F0).

    Uses autocorrelation pitch detection on each speaker's audio segments.
    Male: median F0 < 165 Hz, Female: >= 165 Hz.
    Returns dict: {speaker_label: "M" | "F"}
    """
    from scipy.signal import correlate

    # Collect audio samples per speaker
    speaker_samples = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        start_sample = int(seg.get("start", 0) * sample_rate)
        end_sample = int(seg.get("end", 0) * sample_rate)
        if end_sample <= start_sample:
            continue
        chunk = audio[start_sample:min(end_sample, len(audio))]
        if len(chunk) < sample_rate * 0.1:  # skip < 100ms
            continue
        if speaker not in speaker_samples:
            speaker_samples[speaker] = []
        speaker_samples[speaker].append(chunk)

    # Estimate F0 per speaker using autocorrelation
    genders = {}
    for speaker, chunks in speaker_samples.items():
        pitches = []
        for chunk in chunks[:10]:  # sample up to 10 segments per speaker
            # Window: take a 50ms frame from the middle
            frame_len = min(int(sample_rate * 0.05), len(chunk))
            mid = len(chunk) // 2
            frame = chunk[mid - frame_len // 2 : mid + frame_len // 2]
            if len(frame) < 200:
                continue
            # Autocorrelation
            frame = frame - frame.mean()
            corr = correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]  # positive lags only
            # Find first peak after min_lag (max freq 500Hz)
            min_lag = int(sample_rate / 500)
            max_lag = int(sample_rate / 60)  # min freq 60Hz
            if max_lag > len(corr):
                continue
            search = corr[min_lag:max_lag]
            if len(search) == 0:
                continue
            peak_idx = search.argmax() + min_lag
            if corr[peak_idx] > 0.2 * corr[0]:  # confidence threshold
                f0 = sample_rate / peak_idx
                pitches.append(f0)

        if pitches:
            median_f0 = float(np.median(pitches))
            genders[speaker] = "F" if median_f0 >= 165 else "M"
            log.debug(f"  Speaker {speaker}: median F0={median_f0:.0f} Hz → {genders[speaker]}")
        else:
            genders[speaker] = "?"

    return genders


def apply_gender_labels(segments, genders):
    """Rename speaker labels to include gender prefix: SPEAKER_00 → M-SPEAKER_00."""
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker in genders and genders[speaker] != "?":
            new_label = f"{genders[speaker]}-{speaker}"
            seg["speaker"] = new_label
            for w in seg.get("words", []):
                if w.get("speaker") == speaker:
                    w["speaker"] = new_label
    return segments


# -- Post-processing: split segments at speaker boundaries --------------------
def split_segments_by_speaker(segments):
    """Split segments where the speaker changes mid-segment (using word-level labels).

    Also splits overly long single-speaker segments at sentence boundaries.
    """
    # Split single-speaker segments longer than this at sentence ends.
    # Configurable via env for users who want different segment granularity.
    MAX_SEGMENT_WORDS = int(os.environ.get("MAX_SEGMENT_WORDS", "40"))

    new_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue

        # Group consecutive words by speaker, propagating the last known
        # speaker to unlabeled words instead of inserting spurious "?" groups.
        seg_speaker = seg.get("speaker", "?")
        groups = []
        current_speaker = None
        current_words = []
        for w in words:
            speaker = w.get("speaker") or seg_speaker
            if speaker != current_speaker and current_words:
                groups.append((current_speaker, current_words))
                current_words = []
            current_speaker = speaker
            current_words.append(w)
        if current_words:
            groups.append((current_speaker, current_words))

        # Merge any remaining "?" groups into their neighbor
        merged = []
        for speaker, grp_words in groups:
            if speaker == "?" and merged:
                # Attach to previous speaker's group
                prev_speaker, prev_words = merged[-1]
                merged[-1] = (prev_speaker, prev_words + grp_words)
            else:
                merged.append((speaker, grp_words))
        groups = merged

        # Build new segments from groups
        for speaker, group_words in groups:
            # Further split long groups at sentence boundaries
            sub_segments = _split_at_sentences(group_words, speaker, MAX_SEGMENT_WORDS)
            new_segments.extend(sub_segments)

    return new_segments


def _split_at_sentences(words, speaker, max_words):
    """Split a word list at sentence-ending punctuation if it exceeds max_words."""
    if len(words) <= max_words:
        return [_words_to_segment(words, speaker)]

    segments = []
    current = []
    sentence_enders = {".", "!", "?", "。", "！", "？"}

    for w in words:
        current.append(w)
        text = w.get("word", "").strip()
        # Split if we hit a sentence ender and have enough words
        if len(current) >= 8 and text and text[-1] in sentence_enders:
            segments.append(_words_to_segment(current, speaker))
            current = []

    if current:
        segments.append(_words_to_segment(current, speaker))

    return segments


# Unicode ranges for scripts that don't use spaces between words.
# CJK Unified Ideographs (incl. Ext-A), Hiragana, Katakana, Hangul, Thai, Lao,
# Khmer, Myanmar, Tibetan. We don't insert spaces when joining words from any
# of these scripts — whisperX's word `text` field already carries the correct
# inter-character formatting from the model output.
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


def _is_no_space_script(s: str) -> bool:
    """True if the string contains any character from a no-space-between-words
    script. Used to decide whether to insert spaces when joining word tokens.
    """
    for ch in s:
        cp = ord(ch)
        for lo, hi in _NO_SPACE_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _words_to_segment(words, speaker):
    """Build a segment dict from a list of word dicts.

    Joining strategy:
    - For Latin / Cyrillic / Arabic / etc. (space-separated scripts), join
      stripped tokens with a single space.
    - For CJK / Thai / Lao / Khmer / etc. (no-space scripts), concatenate
      with no separator. whisperX's word `text` already includes appropriate
      inter-character formatting from the model output.
    Detection is per-segment: we sample the first non-empty token. Mixed-script
    segments fall back to space-joined (correct for the Latin parts; the
    CJK parts will retain whatever formatting the model emitted).
    """
    raw_tokens = [w.get("word", "").strip() for w in words]
    tokens = [t for t in raw_tokens if t]

    # Sample the first token to decide the join strategy.
    sep = "" if tokens and _is_no_space_script(tokens[0]) else " "
    text = sep.join(tokens)

    # Use word timestamps for precise boundaries
    starts = [w["start"] for w in words if "start" in w]
    ends = [w["end"] for w in words if "end" in w]
    return {
        "start": starts[0] if starts else words[0].get("start", 0),
        "end": ends[-1] if ends else words[-1].get("end", 0),
        "text": text,
        "speaker": speaker,
        "words": words,
    }


# -- Transcript HTML formatting ------------------------------------------------
import html as html_module

_speaker_index_map = {}


def _speaker_class(speaker_label):
    """Map a speaker label to a CSS class (spk-0 through spk-7, cycling)."""
    if speaker_label not in _speaker_index_map:
        _speaker_index_map[speaker_label] = len(_speaker_index_map) % 8
    return f"spk-{_speaker_index_map[speaker_label]}"


def format_transcript_html(segments, has_speakers=False):
    """Convert segments to HTML with timestamps and optional speaker colors."""
    if not segments:
        return "<div class='transcript-empty'>No segments</div>"
    lines = []
    for seg in segments:
        ts = format_timestamp_display(seg.get("start", 0))
        text = html_module.escape(seg.get("text", "").strip())
        ts_span = f"<span class='transcript-ts'>[{ts}]</span>"
        if has_speakers:
            speaker = seg.get("speaker", "?")
            cls = _speaker_class(speaker)
            spk_span = f"<span class='transcript-speaker {cls}'>{html_module.escape(speaker)}</span>"
            lines.append(f"<div class='transcript-line'>{ts_span}{spk_span}{text}</div>")
        else:
            lines.append(f"<div class='transcript-line'>{ts_span}{text}</div>")
    return "\n".join(lines)


def format_transcript_plain(segments, has_speakers=False):
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


# -- Cancel support ------------------------------------------------------------
_cancel_requested = {"value": False}

# -- Last transcription result (for speaker renaming) --------------------------
# Single-slot global. _transcription_lock prevents concurrent writes, but a
# Gradio user staring at the rename UI from a just-completed transcription
# will see _last_result clobbered if a different user starts a new
# transcription in the same browser instance. Acceptable — Gradio is a
# single-user-at-a-time UI here. Discord rename uses read_cache(video_id)
# instead and is unaffected.
_last_result = {"segments": [], "has_speakers": False, "format": "srt",
                "language": "", "duration": 0.0}

# -- Concurrency lock ----------------------------------------------------------
_transcription_lock = threading.Lock()


# -- Transcription (generator for live UI updates) ----------------------------
def transcribe(file, local_path, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords, initial_prompt, suppress_numerals):
    """Yields (status, transcript, subtitle_file) tuples for live progress."""
    global _previous_subtitle
    _cancel_requested["value"] = False
    request_id = f"req-{int(time.time()*1000) % 100000}"

    # Prevent concurrent transcriptions (would OOM on GPU)
    if not _transcription_lock.acquire(blocking=False):
        log.warning(f"[{request_id}] Rejected -- another transcription is in progress")
        yield "Busy — another transcription is already running", "", "", None
        return

    try:
        yield from _transcribe_inner(file, local_path, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords, initial_prompt, suppress_numerals, request_id)
    finally:
        _transcription_lock.release()
        _reset_idle_timer()  # Reset timer at end so unload countdown is fresh


def _transcribe_inner(file, local_path, model_name, language, output_format,
                      enable_diarization, min_speakers, max_speakers, batch_size,
                      hotwords, initial_prompt, suppress_numerals, request_id,
                      return_file=True, task="transcribe"):
    """Inner generator — runs under _transcription_lock.
    Yields 4-tuples: (status, html_view, plain_text, subtitle_file).

    `return_file=False` skips subtitle file generation (callers that only need
    the transcript text — e.g. the bot via /api/transcribe — avoid disk I/O
    and the leak window if the response is dropped).

    `task` is either "transcribe" (preserve source language) or "translate"
    (Whisper outputs English regardless of source). Translate is the right
    default for code-switched audio per CS-FLEURS (arXiv:2509.14161):
    translation BLEU barely degrades on mixed-language inputs while
    transcribe CER doubles. wav2vec2 alignment is skipped in translate
    mode (the aligner is source-language-specific; English transcript over
    non-English audio produces garbage word-level timestamps).
    """
    global _previous_subtitle

    _reset_idle_timer()  # Mark activity — prevent unload during transcription
    _speaker_index_map.clear()

    def _plain_html(text):
        """Wrap plain text in a pre-formatted div for intermediate progress."""
        if not text:
            return ""
        escaped = html_module.escape(text)
        return f"<div style='white-space:pre-wrap;font-size:0.8rem;line-height:1.5'>{escaped}</div>"

    # Prefer local path over uploaded file (for large files that can't be uploaded via browser)
    local_path_str = local_path.strip() if local_path else ""
    if local_path_str:
        if os.path.isfile(local_path_str):
            file = local_path_str
            log.info(f"[{request_id}] Using local path: {file}")
        else:
            log.error(f"[{request_id}] Local path not found: {local_path_str}")
            yield f"Error: file not found: {local_path_str}", "", "", None
            return

    log.info(f"[{request_id}] == New transcription request ==")
    log.info(f"[{request_id}] File: {file}")

    if file is None:
        log.warning(f"[{request_id}] No file provided")
        yield "No file uploaded or local path specified.", "", "", None
        return

    # File info
    file_size_str = ""
    try:
        file_size = os.path.getsize(file) / (1024 * 1024)
        file_size_str = f" ({file_size:.0f} MB)"
        log.info(f"[{request_id}] File size: {file_size:.1f} MB")
    except Exception:
        log.info(f"[{request_id}] Could not determine file size")

    batch_size = int(batch_size)
    hotwords_str = hotwords.strip() if hotwords else ""
    prompt_str = initial_prompt.strip() if initial_prompt else ""
    log.info(f"[{request_id}] Model: {model_name}")
    log.info(f"[{request_id}] Language: {language}")
    log.info(f"[{request_id}] Task: {task}")
    log.info(f"[{request_id}] Output format: {output_format}")
    log.info(f"[{request_id}] Diarization: {enable_diarization}")
    log.info(f"[{request_id}] Batch size: {batch_size}")
    log.info(f"[{request_id}] Suppress numerals: {suppress_numerals}")
    if hotwords_str:
        log.info(f"[{request_id}] Hotwords: {hotwords_str[:100]}")
    if prompt_str:
        log.info(f"[{request_id}] Initial prompt: {prompt_str[:100]}")
    log.info(f"[{request_id}] Device: {DEVICE}, compute_type: {COMPUTE_TYPE}")

    # -- Phase 1: Load whisperX model --
    yield f"Loading whisperX model '{model_name}'...", "", "", None
    log.info(f"[{request_id}] Loading whisperX model...")
    t0_model = time.time()
    m = load_whisper(
        model_name,
        hotwords=hotwords_str or None,
        initial_prompt=prompt_str or None,
        suppress_numerals=bool(suppress_numerals),
    )
    model_time = time.time() - t0_model
    log.info(f"[{request_id}] WhisperX model ready in {model_time:.2f}s")

    lang = None if (not language or language == "Auto-detect") else language

    if _cancel_requested["value"]:
        log.info(f"[{request_id}] Cancelled before audio load")
        yield "Cancelled", "", "", None
        return

    # -- Phase 2: Load audio --
    yield f"Loading audio{file_size_str}...", "", "", None

    # Verify the file still exists (Gradio temp files can vanish between yields)
    if not os.path.exists(file):
        log.error(f"[{request_id}] File no longer exists: {file}")
        yield f"Error: uploaded file no longer exists (may have been cleaned up)", "", "", None
        return

    # Create a safe temp path without spaces (ffmpeg subprocess can have issues
    # with spaces in paths). Use a hard link (instant) instead of copying GBs.
    ext = os.path.splitext(file)[1] or ".mkv"
    safe_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    safe_tmp.close()
    safe_path = safe_tmp.name
    try:
        os.remove(safe_path)  # remove empty file so we can link
        os.link(file, safe_path)  # hard link -- instant, no data copied
        log.info(f"[{request_id}] Hard-linked to safe temp path: {safe_path}")
    except OSError:
        # Cross-device link or unsupported -- fall back to copy
        try:
            log.info(f"[{request_id}] Hard link failed, copying to: {safe_path}")
            shutil.copy2(file, safe_path)
        except Exception as e:
            log.error(f"[{request_id}] Failed to copy file: {e}")
            yield f"Error copying file: {e}", "", "", None
            return

    log.info(f"[{request_id}] Loading audio...")
    t0_audio = time.time()
    try:
        audio = whisperx.load_audio(safe_path)
    except Exception as e:
        log.error(f"[{request_id}] Failed to load audio: {e}")
        traceback.print_exc()
        yield f"Error loading audio: {e}", "", "", None
        return
    finally:
        # Remove the safe copy now that audio is in memory
        try:
            os.remove(safe_path)
        except Exception:
            pass
    audio_duration = len(audio) / 16000  # whisperx loads at 16kHz
    log.info(f"[{request_id}] Audio loaded in {time.time()-t0_audio:.2f}s ({audio_duration:.0f}s / {audio_duration/60:.1f} min)")

    if _cancel_requested["value"]:
        log.info(f"[{request_id}] Cancelled before transcription")
        yield "Cancelled", "", "", None
        return

    # -- Phase 3: Transcribe (batched) --
    duration_str = f"{audio_duration/60:.1f} min" if audio_duration >= 60 else f"{audio_duration:.0f}s"
    yield f"Transcribing {duration_str} of audio (batch_size={batch_size})...", "", "", None
    log.info(f"[{request_id}] >> Starting batched transcription...")
    t0 = time.time()

    try:
        result = m.transcribe(
            audio,
            language=lang,
            batch_size=batch_size,
            task=task,
            print_progress=False,  # whisperX prints to stdout (not logger); keep off in containers
        )
    except Exception as e:
        log.error(f"[{request_id}] Transcription FAILED: {e}")
        traceback.print_exc()
        yield f"Error: {e}", "", "", None
        return

    transcribe_elapsed = time.time() - t0
    detected_lang = result.get("language", lang or "unknown")
    num_segments = len(result.get("segments", []))
    log.info(f"[{request_id}] [OK] Transcription complete")
    log.info(f"[{request_id}]   Time: {transcribe_elapsed:.1f}s")
    log.info(f"[{request_id}]   Audio duration: {audio_duration:.0f}s ({audio_duration/60:.1f} min)")
    log.info(f"[{request_id}]   Speed: {audio_duration/transcribe_elapsed:.1f}x realtime")
    log.info(f"[{request_id}]   Segments: {num_segments}")
    log.info(f"[{request_id}]   Detected language: {detected_lang}")

    # Show initial transcript before alignment
    speed = audio_duration / transcribe_elapsed if transcribe_elapsed > 0 else 0
    formatted_lines = []
    for seg in result.get("segments", []):
        ts = format_timestamp_display(seg.get("start", 0))
        formatted_lines.append(f"[{ts}] {seg.get('text', '').strip()}")
    yield f"Transcribed {num_segments} segments in {transcribe_elapsed:.1f}s ({speed:.1f}x realtime), aligning...", _plain_html("\n".join(formatted_lines)), "\n".join(formatted_lines), None

    if _cancel_requested["value"]:
        log.info(f"[{request_id}] Cancelled before alignment")
        yield "Cancelled (transcription complete, no alignment)", _plain_html("\n".join(formatted_lines)), "\n".join(formatted_lines), None
        return

    # -- Phase 4: Word-level alignment (wav2vec2) --
    # Skip alignment when:
    #   (a) task=translate — wav2vec2 aligners are source-language phoneme
    #       models. The transcript is English but the audio is the source
    #       language, so the aligner can't match. Word timestamps would be
    #       meaningless. Whisper segment timestamps survive translate mode
    #       and that's enough for chapter markers / summarisation.
    #   (b) detected_lang has no default aligner — whisperx's
    #       DEFAULT_ALIGN_MODELS_* cover ~40 languages; Whisper supports 100.
    #       For the uncovered 60 (e.g. yue Cantonese, id Indonesian)
    #       load_align_model would raise ValueError. Detect upfront so we
    #       log at INFO ("expected, by design") instead of WARNING.
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
    align_elapsed = 0.0
    alignment_ok = False
    skip_reason = None
    if task == "translate":
        skip_reason = "task=translate (English transcript over non-English audio)"
    elif (detected_lang not in DEFAULT_ALIGN_MODELS_TORCH
          and detected_lang not in DEFAULT_ALIGN_MODELS_HF):
        skip_reason = f"no default aligner for '{detected_lang}' (whisperx covers ~40 of Whisper's 100 languages)"

    if skip_reason:
        log.info(f"[{request_id}] Skipping alignment: {skip_reason}")
    else:
        log.info(f"[{request_id}] Running word-level alignment for '{detected_lang}'...")
        t0_align = time.time()
        try:
            model_a, metadata = load_align_model(detected_lang)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
                print_progress=False,  # whisperX prints to stdout (not logger); keep off in containers
            )
            align_elapsed = time.time() - t0_align
            alignment_ok = True
            log.info(f"[{request_id}]   Alignment complete in {align_elapsed:.1f}s")
        except Exception as e:
            align_elapsed = time.time() - t0_align
            log.warning(f"[{request_id}]   Alignment failed (proceeding without): {e}")
            # result still has segment-level timestamps, just not word-level

    # Rebuild display after alignment (timestamps may have been refined)
    formatted_lines = []
    for seg in result.get("segments", []):
        ts = format_timestamp_display(seg.get("start", 0))
        formatted_lines.append(f"[{ts}] {seg.get('text', '').strip()}")
    if alignment_ok:
        align_status = f"Aligned in {align_elapsed:.1f}s"
    elif skip_reason:
        align_status = f"Alignment skipped: {skip_reason} (segment-level timestamps only)"
    else:
        align_status = "Alignment failed (segment-level timestamps only)"
    yield f"{align_status} -- {num_segments} segments", _plain_html("\n".join(formatted_lines)), "\n".join(formatted_lines), None

    # -- Phase 5: Speaker diarization (optional) --
    # Note: diarization works without alignment (segment-level speaker labels),
    # but word-level speaker attribution requires aligned word timestamps.
    if enable_diarization and DIARIZATION_AVAILABLE and not alignment_ok:
        log.info(f"[{request_id}]   Alignment failed -- diarization will use segment-level assignment only (no per-word speakers)")
    if enable_diarization and DIARIZATION_AVAILABLE:
        yield "Loading diarization pipeline...", _plain_html("\n".join(formatted_lines)), "\n".join(formatted_lines), None
        log.info(f"[{request_id}] Running speaker diarization...")
        min_spk = int(min_speakers) if min_speakers and int(min_speakers) > 0 else None
        max_spk = int(max_speakers) if max_speakers and int(max_speakers) > 0 else None
        if min_spk or max_spk:
            log.info(f"[{request_id}]   Speaker constraints: min={min_spk}, max={max_spk}")
        t0_diar = time.time()
        try:
            dpipe = load_diarization()
            if dpipe is not None:
                yield "Running speaker diarization...", _plain_html("\n".join(formatted_lines)), "\n".join(formatted_lines), None
                diarize_segments = dpipe(audio, min_speakers=min_spk, max_speakers=max_spk)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                # Split segments at speaker boundaries (requires word-level timestamps)
                if alignment_ok:
                    original_count = len(result.get("segments", []))
                    result["segments"] = split_segments_by_speaker(result.get("segments", []))
                    new_count = len(result["segments"])
                    if new_count != original_count:
                        log.info(f"[{request_id}]   Split {original_count} -> {new_count} segments at speaker/sentence boundaries")
                else:
                    log.info(f"[{request_id}]   Skipping segment splitting (no word timestamps)")

                # Estimate gender from pitch and apply labels (after splitting so all segments get labeled)
                try:
                    genders = estimate_speaker_genders(audio, result.get("segments", []))
                    if genders:
                        result["segments"] = apply_gender_labels(result.get("segments", []), genders)
                        gender_summary = ", ".join(f"{k}={v}" for k, v in sorted(genders.items()))
                        log.info(f"[{request_id}]   Gender estimates: {gender_summary}")
                except Exception as e:
                    log.warning(f"[{request_id}]   Gender estimation failed (non-critical): {e}")

                # Rebuild formatted lines with speaker labels
                formatted_lines = []
                for seg in result.get("segments", []):
                    ts = format_timestamp_display(seg.get("start", 0))
                    speaker = seg.get("speaker", "?")
                    formatted_lines.append(f"[{ts}] [{speaker}] {seg.get('text', '').strip()}")

                num_speakers = len(set(
                    seg.get("speaker", "?") for seg in result.get("segments", [])
                ))
                diar_time = time.time() - t0_diar
                log.info(f"[{request_id}]   Diarization complete in {diar_time:.1f}s ({num_speakers} speakers)")
            else:
                log.warning(f"[{request_id}]   Diarization pipeline not available")
        except Exception as e:
            log.error(f"[{request_id}]   Diarization failed: {e}")
            traceback.print_exc()
    elif enable_diarization:
        log.warning(f"[{request_id}]   Diarization requested but HF_TOKEN not set")

    # Free the raw audio array -- no longer needed after alignment/diarization
    del audio

    transcript = "\n".join(formatted_lines)
    segments = result.get("segments", [])
    num_segments = len(segments)  # update after potential splitting
    log.info(f"[{request_id}]   Text length: {len(transcript)} chars")

    # -- Phase 6: Generate subtitle file --
    subtitle_file = None
    has_speakers = enable_diarization and any(seg.get("speaker") for seg in segments)

    if not return_file:
        log.info(f"[{request_id}] Skipping subtitle file generation (return_file=False)")
    elif output_format == "txt":
        yield "Generating txt file...", _plain_html(transcript), transcript, None
        log.info(f"[{request_id}] Generating txt file...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        tmp.write(transcript)
        tmp.close()
        subtitle_file = tmp.name
    elif output_format == "json":
        yield "Generating JSON file...", _plain_html(transcript), transcript, None
        log.info(f"[{request_id}] Generating JSON file with word-level timestamps...")
        json_data = {
            "language": detected_lang,
            "duration": round(audio_duration, 2),
            "segments": [],
        }
        for seg in segments:
            seg_out = {
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "text": seg.get("text", "").strip(),
            }
            if has_speakers:
                seg_out["speaker"] = seg.get("speaker", "?")
            # Include word-level timestamps when available
            words = seg.get("words", [])
            if words:
                seg_out["words"] = []
                for w in words:
                    word_out = {"word": w.get("word", "").strip()}
                    if "start" in w:
                        word_out["start"] = round(w["start"], 3)
                    if "end" in w:
                        word_out["end"] = round(w["end"], 3)
                    if "score" in w:
                        word_out["confidence"] = round(w["score"], 3)
                    if has_speakers and "speaker" in w:
                        word_out["speaker"] = w["speaker"]
                    seg_out["words"].append(word_out)
            json_data["segments"].append(seg_out)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        json.dump(json_data, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        subtitle_file = tmp.name
        log.info(f"[{request_id}] JSON file: {subtitle_file}")
    elif output_format in ("srt", "vtt"):
        yield f"Generating {output_format} file...", _plain_html(transcript), transcript, None
        log.info(f"[{request_id}] Generating {output_format} subtitle file...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}", mode="w")
        if output_format == "srt":
            for i, seg in enumerate(segments, 1):
                start_ts = format_timestamp_srt(seg.get("start", 0))
                end_ts = format_timestamp_srt(seg.get("end", 0))
                speaker_prefix = ""
                if has_speakers:
                    speaker_prefix = f"[{seg.get('speaker', '?')}] "
                tmp.write(f"{i}\n{start_ts} --> {end_ts}\n{speaker_prefix}{seg.get('text', '').strip()}\n\n")
        else:  # vtt
            tmp.write("WEBVTT\n\n")
            for seg in segments:
                start_ts = format_timestamp_vtt(seg.get("start", 0))
                end_ts = format_timestamp_vtt(seg.get("end", 0))
                speaker_prefix = ""
                if has_speakers:
                    speaker_prefix = f"[{seg.get('speaker', '?')}] "
                tmp.write(f"{start_ts} --> {end_ts}\n{speaker_prefix}{seg.get('text', '').strip()}\n\n")
        tmp.close()
        subtitle_file = tmp.name
        log.info(f"[{request_id}] Subtitle file: {subtitle_file}")

    total_time = model_time + transcribe_elapsed
    done_parts = [f"{num_segments} segments", f"{transcribe_elapsed:.0f}s ({speed:.1f}x realtime)", detected_lang]
    if has_speakers:
        speaker_count = len(set(seg.get("speaker", "?") for seg in segments))
        done_parts.append(f"{speaker_count} speakers")
    done_msg = f"Done -- {', '.join(done_parts)}"
    log.info(f"[{request_id}] == Request complete ({total_time:.1f}s total) ==")

    # Cleanup: only remove previous subtitle file (not the uploaded file --
    # the user may re-transcribe with different settings)
    if _previous_subtitle and os.path.exists(_previous_subtitle):
        try:
            os.remove(_previous_subtitle)
            log.info(f"[{request_id}] Cleaned previous subtitle: {_previous_subtitle}")
        except Exception:
            pass
    _previous_subtitle = subtitle_file

    # Store for speaker renaming
    _last_result["segments"] = segments
    _last_result["has_speakers"] = has_speakers
    _last_result["format"] = output_format
    _last_result["language"] = detected_lang
    _last_result["duration"] = audio_duration

    # Save to history
    import datetime
    _save_history_entry({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename": os.path.basename(file) if file else "unknown",
        "duration_str": duration_str,
        "language": detected_lang,
        "speakers": len(set(seg.get("speaker", "?") for seg in segments)) if has_speakers else "",
        "speed": f"{speed:.1f}x",
        "segments": num_segments,
    })

    _reset_idle_timer()  # Start countdown to unload models
    yield done_msg, format_transcript_html(segments, has_speakers), transcript, subtitle_file


# -- Timestamp formatting ------------------------------------------------------
def format_timestamp_display(seconds):
    """Short timestamp for transcript display: MM:SS or H:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_timestamp_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# -- Languages -----------------------------------------------------------------
LANGUAGES = [
    "Auto-detect", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt",
    "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th",
    "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
    "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is",
    "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa",
]

# -- Scan /media for local files -----------------------------------------------
# -- yt-dlp helpers (shared by UI and HTTP API) --------------------------------

# yt-dlp error patterns that will not recover on retry. Matched against
# stderr; the API layer maps these to HTTP 422 so clients (the bot) can
# short-circuit their retry loops.
_PERMANENT_YT_DLP_PATTERNS = (
    "Sign in to confirm your age",            # age-gated, needs cookies
    "Private video",
    "Video unavailable",
    "This video is unavailable",
    "members-only content",
    "members only video",
    "This video has been removed",
    "blocked it on copyright grounds",
    "blocked it in your country",
    "country and is unavailable",
    "Premieres in",                            # not yet released
    "This live event will begin",              # future stream
    "Sign in to confirm you're not a bot",    # IP/account-flagged
    "Join this channel to get access",         # paid membership
    "Video is not available",
    # Platform-specific "no media in this URL" cases — link triggered the
    # bot (post matched VIDEO_URL_PATTERN) but the destination has no
    # downloadable media. Retrying re-hits the same empty result.
    "No video could be found in this tweet",  # x.com / twitter
    "No video could be found in this",        # generic catch
    "No video formats found",
    "no video formats found",                  # case variants
    "Unsupported URL",                          # yt-dlp doesn't know the host
    "is not a valid URL",
    "There's no video in this post",          # instagram / threads
    "No media found",
    "Post does not contain any media",
)


def _is_permanent_yt_dlp_error(stderr: str) -> bool:
    return any(p in stderr for p in _PERMANENT_YT_DLP_PATTERNS)


def _extract_comments(meta: dict, output_dir: str, video_id: str) -> list[dict]:
    """Pull comments from yt-dlp's --print-json blob; fall back to the
    .info.json file on disk when --print-json truncated them.

    yt-dlp's stdout JSON sometimes drops the `comments` array when it's
    large (the runtime concats can hit buffer limits). The info.json file
    on disk has the full payload. Either way the field is `comments`.

    Returns a list of dicts with normalised keys:
      text, author, like_count, is_pinned, is_favorited,
      author_is_uploader, parent, time_text
    """
    raw = meta.get("comments")
    if not raw:
        # Probe the info.json file
        info_path = os.path.join(output_dir, f"{video_id}.info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                raw = info.get("comments") or []
            except (OSError, json.JSONDecodeError) as e:
                log.warning(f"[API] couldn't read info.json comments: {e}")
                raw = []
        else:
            raw = []

    if not isinstance(raw, list):
        return []

    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        out.append({
            "text": (c.get("text") or "").strip(),
            "author": c.get("author") or "[unknown]",
            "like_count": int(c.get("like_count") or 0),
            "is_pinned": bool(c.get("is_pinned")),
            "is_favorited": bool(c.get("is_favorited")),  # creator-hearted
            "author_is_uploader": bool(c.get("author_is_uploader")),
            "parent": c.get("parent") or "root",
            "time_text": c.get("_time_text") or "",
        })
    return out


def _yt_dlp_auth_args() -> list[str]:
    """Build cookie/auth args for yt-dlp from environment.

    YT_DLP_COOKIES_FILE          — path to Netscape cookies.txt readable by
                                   the container (mount it in via compose).
    YT_DLP_COOKIES_FROM_BROWSER  — passed straight to --cookies-from-browser.
                                   Only useful if the browser is installed in
                                   the container, which the default image
                                   does not do.

    Operator note: handle YouTube's age-gate / "are you a bot" challenges by
    creating a throwaway YouTube account, exporting its cookies once, and
    mounting the file. Per-user cookie capture from Discord is not supported
    (full account credentials over Discord DMs is a security non-starter).
    """
    args: list[str] = []
    cookies_file = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    if cookies_file:
        if os.path.isfile(cookies_file):
            args.extend(["--cookies", cookies_file])
        else:
            log.warning(
                f"YT_DLP_COOKIES_FILE set to {cookies_file!r} but file not found "
                f"— age-restricted videos will fail"
            )
    cookies_browser = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_browser:
        args.extend(["--cookies-from-browser", cookies_browser])
    return args


# -- /media filesystem scanning ------------------------------------------------

MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "/media")
MEDIA_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
                    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".ts", ".flv"}


# Result cache for scan_media_files. os.walk over a large media library
# (TV show backups, podcast archives) is slow; cache for a short TTL so the UI
# refresh is fast and the explicit Refresh button still works (it bypasses
# the cache by passing force=True).
_MEDIA_SCAN_TTL = int(os.environ.get("MEDIA_SCAN_TTL", "60"))
_media_scan_cache: dict = {"at": 0.0, "result": []}


def scan_media_files(force: bool = False) -> list[tuple[str, str]]:
    """Walk MEDIA_ROOT and return (display, full_path) tuples sorted newest-first.

    Cached for MEDIA_SCAN_TTL seconds (default 60). Pass force=True to bypass.
    """
    now = time.time()
    if not force and (now - _media_scan_cache["at"]) < _MEDIA_SCAN_TTL:
        return _media_scan_cache["result"]
    if not os.path.isdir(MEDIA_ROOT):
        _media_scan_cache.update(at=now, result=[])
        return []
    found = []
    for root, _dirs, files in os.walk(MEDIA_ROOT):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in MEDIA_EXTENSIONS:
                full_path = os.path.join(root, fname)
                found.append(full_path)
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    result = [(os.path.basename(p), p) for p in found]
    _media_scan_cache.update(at=now, result=result)
    return result


# -- Custom CSS ----------------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
/* Header */
.header-wrap {
    text-align: center;
    padding: 0.5rem 0 0.5rem;
}
.header-wrap h2 {
    margin: 0 0 4px;
    font-weight: 700;
    font-size: 1.4rem;
}
.header-wrap .gpu {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72rem;
    opacity: 0.5;
}
/* Hide Gradio footer */
footer { display: none !important; }
/* Fix file upload label — keep it inside the box at the top */
.gradio-container .file-preview,
.gradio-container [data-testid="file"] {
    position: relative;
}
.gradio-container [data-testid="file"] label {
    position: static !important;
    transform: none !important;
    margin-bottom: 0.25rem;
}
/* Constrain media dropdown height + scroll */
#media-select ul[role="listbox"],
#media-select .options {
    max-height: 280px !important;
    overflow-y: auto !important;
}
/* Refresh button — icon-only, compact, aligned with dropdown */
#refresh-media-btn {
    max-width: 40px;
    min-width: 40px;
    align-self: flex-end;
    margin-bottom: 0;
}
#refresh-media-btn button {
    min-width: 40px !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
}
/* Status bar styling */
#status-bar textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500;
}
/* Transcript area */
#transcript-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}
/* Output actions row */
.output-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}
#copy-transcript-btn {
    max-width: 150px;
}
#copy-transcript-btn button {
    font-size: 0.8rem;
    padding: 4px 14px;
}
/* Tighter accordion spacing */
.accordion-compact .label-wrap {
    padding: 8px 12px !important;
}
/* Cancel button */
#cancel-btn button {
    border-color: var(--color-red-500) !important;
    color: var(--color-red-500) !important;
}
#cancel-btn button:hover {
    background: var(--color-red-500) !important;
    color: white !important;
}
/* Transcript HTML viewer — force containment */
#transcript-html {
    max-height: 450px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    background: var(--background-fill-secondary);
}
/* Ensure Gradio's wrapper div doesn't expand beyond the inner constraint */
#transcript-html > div {
    max-height: 430px !important;
    overflow-y: auto !important;
}
.transcript-empty {
    opacity: 0.4;
    font-style: italic;
}
.transcript-line {
    margin-bottom: 0.3rem;
    padding: 2px 0;
}
.transcript-ts {
    opacity: 0.45;
    font-size: 0.75rem;
    margin-right: 0.5rem;
}
.transcript-speaker {
    font-weight: 600;
    font-size: 0.75rem;
    padding: 1px 5px;
    border-radius: 3px;
    margin-right: 0.4rem;
}
/* Speaker color palette */
.spk-0 { color: #f97316; background: rgba(249,115,22,0.12); }
.spk-1 { color: #3b82f6; background: rgba(59,130,246,0.12); }
.spk-2 { color: #10b981; background: rgba(16,185,129,0.12); }
.spk-3 { color: #a855f7; background: rgba(168,85,247,0.12); }
.spk-4 { color: #ec4899; background: rgba(236,72,153,0.12); }
.spk-5 { color: #eab308; background: rgba(234,179,8,0.12); }
.spk-6 { color: #06b6d4; background: rgba(6,182,212,0.12); }
.spk-7 { color: #f43f5e; background: rgba(244,63,94,0.12); }
/* Search box */
#transcript-search {
    margin-bottom: 0.25rem;
}
#transcript-search input {
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
}
"""

# -- Gradio UI -----------------------------------------------------------------
log.info("Building Gradio UI...")

THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="zinc",
    font=["Inter", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "Fira Code", "monospace"],
)

# whisper-live sidecar (mic streaming for the Live tab). Read at module level
# so the Gradio handler can reference it as a global at call time.
LIVE_SERVICE_URL = os.environ.get("LIVE_SERVICE_URL", "http://localhost:7861")

with gr.Blocks(title="WhisperX Transcription") as demo:
    demo.queue()

    gr.HTML(f"""
        <div class="header-wrap">
            <h2>WhisperX Transcription</h2>
            <div class="gpu">{GPU_INFO_STR}</div>
        </div>
    """)

    # -- Input: File source (tabbed) --
    with gr.Tabs():
        with gr.Tab("Upload"):
            file_input = gr.File(
                label="Drop or click to upload audio/video",
                file_types=["audio", "video"],
                height=100,
            )
        with gr.Tab("Local file"):
            with gr.Row():
                local_path_input = gr.Dropdown(
                    choices=scan_media_files(),
                    value=None,
                    label="Select from /media",
                    allow_custom_value=True,
                    scale=12,
                    elem_id="media-select",
                )
                refresh_media_btn = gr.Button("↻", scale=1, size="sm", variant="secondary", elem_id="refresh-media-btn")
        with gr.Tab("YouTube"):
            with gr.Row():
                yt_url_input = gr.Textbox(
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                    max_lines=1,
                    scale=12,
                    show_label=False,
                    container=False,
                )
                yt_fetch_btn = gr.Button("Fetch", scale=1, size="sm", variant="primary", elem_id="yt-fetch-btn")
        with gr.Tab("Live"):
            gr.Markdown(
                "Transcribe from microphone in real time. Click **Record**, "
                "speak, and the transcript accumulates below. **Clear** resets it."
            )
            live_audio_input = gr.Audio(
                streaming=True,
                sources=["microphone"],
                label="Microphone",
            )
            live_transcript_box = gr.Textbox(
                label="Live Transcript",
                lines=12,
                interactive=False,
                placeholder="Transcript will appear here as you speak…",
            )
            live_state = gr.State({"buffer": b"", "transcript": ""})
            live_clear_btn = gr.Button("Clear", variant="secondary")

    def _live_chunk(audio_chunk, state):
        """Per-mic-chunk callback. Accumulates PCM in state; flushes to
        whisper-live every ~10 s (matches server CHUNK_SECONDS). Uses stdlib
        urllib (no `requests` dependency — not in the whisper image).
        Gradio runs handlers in a thread pool, so a blocking call is fine."""
        import json as _json
        import urllib.parse
        import urllib.request

        if audio_chunk is None:
            return state["transcript"], state
        sr, arr = audio_chunk
        if arr.dtype != np.int16:
            arr = (arr.clip(-1.0, 1.0) * 32767).astype(np.int16)
        if sr != 16000:
            from scipy.signal import resample_poly
            arr = resample_poly(arr, 16000, sr).astype(np.int16)
        state["buffer"] += arr.tobytes()

        THRESHOLD = 16000 * 2 * 10  # 10 s of 16 kHz int16
        if len(state["buffer"]) < THRESHOLD:
            return state["transcript"], state

        try:
            qs = urllib.parse.urlencode({"context": state["transcript"][-300:]})
            req = urllib.request.Request(
                f"{LIVE_SERVICE_URL}/transcribe-chunk?{qs}",
                data=bytes(state["buffer"]),
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = _json.loads(resp.read().decode())
            for seg in payload.get("segments", []):
                state["transcript"] += seg["text"] + " "
        except Exception as exc:
            log.warning(f"[live] chunk POST failed: {exc}")
        state["buffer"] = b""
        return state["transcript"], state

    def _refresh_media():
        # Manual refresh bypasses the TTL cache.
        return gr.update(choices=scan_media_files(force=True))

    refresh_media_btn.click(fn=_refresh_media, outputs=[local_path_input])

    live_audio_input.stream(
        fn=_live_chunk,
        inputs=[live_audio_input, live_state],
        outputs=[live_transcript_box, live_state],
        show_progress=False,
    )
    live_clear_btn.click(
        fn=lambda: ("", {"buffer": b"", "transcript": ""}),
        outputs=[live_transcript_box, live_state],
    )

    # yt_fetch_btn event wiring is registered later (after status_text is defined)

    # -- Core settings (always visible) --
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large", "turbo"],
            value="turbo",
            label="Model",
            scale=2,
        )
        lang_dropdown = gr.Dropdown(
            choices=LANGUAGES,
            value="Auto-detect",
            label="Language",
            scale=2,
        )
        format_dropdown = gr.Dropdown(
            choices=["txt", "srt", "vtt", "json"],
            value="srt",
            label="Format",
            scale=1,
        )
        batch_slider = gr.Slider(
            minimum=1,
            maximum=64,
            step=1,
            value=DEFAULT_BATCH_SIZE,
            label="Batch size",
            scale=2,
        )

    # -- Speaker diarization --
    with gr.Accordion("Speaker diarization", open=False):
        if not DIARIZATION_AVAILABLE:
            gr.Markdown(
                "<small style='opacity:0.6'>Disabled — set <code>HF_TOKEN</code> env var to enable.</small>"
            )
        with gr.Row():
            diarize_checkbox = gr.Checkbox(
                label="Enable (identify speakers)",
                value=False,
                interactive=DIARIZATION_AVAILABLE,
                scale=2,
            )
            min_speakers_input = gr.Number(
                value=0,
                label="Min speakers",
                info="0 = auto",
                minimum=0,
                maximum=20,
                precision=0,
                interactive=DIARIZATION_AVAILABLE,
                scale=1,
            )
            max_speakers_input = gr.Number(
                value=0,
                label="Max speakers",
                info="0 = auto",
                minimum=0,
                maximum=20,
                precision=0,
                interactive=DIARIZATION_AVAILABLE,
                scale=1,
            )

    # -- Advanced options (collapsed by default) --
    with gr.Accordion("Advanced options", open=False):
        hotwords_input = gr.Textbox(
            label="Hotwords",
            placeholder="proper nouns, product names, technical terms the model might mishear",
            lines=1,
            max_lines=1,
        )
        initial_prompt_input = gr.Textbox(
            label="Initial prompt",
            placeholder="Context hint for the first transcription window",
            lines=1,
            max_lines=2,
        )
        suppress_numerals_input = gr.Checkbox(
            label="Suppress numerals (spell out numbers — improves alignment)",
            value=False,
        )

    # -- Action buttons --
    with gr.Row():
        transcribe_btn = gr.Button(
            "Transcribe",
            variant="primary",
            interactive=True,
            scale=4,
        )
        cancel_btn = gr.Button(
            "Cancel",
            variant="stop",
            scale=1,
            elem_id="cancel-btn",
        )

    # -- Output section --
    status_text = gr.Textbox(
        label="Status",
        lines=1,
        max_lines=1,
        interactive=False,
        placeholder="Ready",
        elem_id="status-bar",
    )
    search_box = gr.Textbox(
        placeholder="Search transcript...",
        lines=1,
        max_lines=1,
        elem_id="transcript-search",
        show_label=False,
        container=False,
    )
    search_box.input(
        fn=None,
        inputs=[search_box],
        js="""(query) => {
            const container = document.querySelector('#transcript-html');
            if (!container) return;
            const lines = container.querySelectorAll('.transcript-line');
            const q = (query || '').toLowerCase().trim();
            lines.forEach(el => {
                if (!q) {
                    el.style.display = '';
                    el.querySelectorAll('mark').forEach(m => m.replaceWith(m.textContent));
                } else {
                    const text = el.textContent.toLowerCase();
                    if (text.includes(q)) {
                        el.style.display = '';
                    } else {
                        el.style.display = 'none';
                    }
                }
            });
        }""",
    )
    output_html = gr.HTML(
        value="<div id='transcript-view' class='transcript-empty'>Transcript will appear here...</div>",
        elem_id="transcript-html",
    )
    # Hidden textbox holds plain text for copy/download
    output_text = gr.Textbox(visible=False, elem_id="transcript-raw")

    # State: store segments for speaker renaming
    segments_state = gr.State(value=[])
    has_speakers_state = gr.State(value=False)

    # Speaker rename section
    with gr.Accordion("Rename speakers", open=False, visible=False) as speaker_accordion:
        gr.Markdown("<small>Rename detected speakers and click Apply to update transcript and subtitle file.</small>")
        speaker_rename_input = gr.Textbox(
            label="Speaker names (one per line: SPEAKER_00=Alice)",
            placeholder="SPEAKER_00=Alice\nSPEAKER_01=Bob",
            lines=4,
            max_lines=8,
        )
        apply_rename_btn = gr.Button("Apply renames", variant="secondary", size="sm")

    with gr.Row():
        output_file = gr.File(label="Download", height=50, interactive=False, scale=3)
        copy_btn = gr.Button(
            "Copy transcript",
            size="sm",
            variant="secondary",
            elem_id="copy-transcript-btn",
            scale=1,
        )
    copy_btn.click(
        fn=None,
        inputs=[output_text],
        outputs=[],
        js="""(text) => {
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.querySelector('#copy-transcript-btn button');
                if (btn) { const o = btn.textContent; btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = o, 1500); }
            });
        }""",
    )

    # -- Speaker rename logic --
    def _show_speaker_rename(status, *_):
        """After transcription, show the rename UI if speakers were detected."""
        if not _last_result["has_speakers"]:
            return gr.update(visible=False), ""
        speakers = sorted(set(seg.get("speaker", "?") for seg in _last_result["segments"]))
        prefill = "\n".join(f"{s}={s}" for s in speakers)
        return gr.update(visible=True, open=True), prefill

    def _apply_speaker_renames(rename_text):
        """Apply speaker name mappings and regenerate outputs."""
        renames = {}
        for line in rename_text.strip().split("\n"):
            if "=" in line:
                old, new = line.split("=", 1)
                old, new = old.strip(), new.strip()
                if old and new:
                    renames[old] = new
        if not renames:
            return gr.update(), gr.update(), gr.update()

        segments = _last_result["segments"]
        # Apply renames to segments
        for seg in segments:
            if seg.get("speaker") in renames:
                seg["speaker"] = renames[seg["speaker"]]
            for w in seg.get("words", []):
                if w.get("speaker") in renames:
                    w["speaker"] = renames[w["speaker"]]

        has_speakers = _last_result["has_speakers"]
        html = format_transcript_html(segments, has_speakers)
        plain = format_transcript_plain(segments, has_speakers)

        # Regenerate subtitle file
        output_format = _last_result["format"]
        subtitle_file = None
        if output_format == "txt":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
            tmp.write(plain)
            tmp.close()
            subtitle_file = tmp.name
        elif output_format in ("srt", "vtt"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}", mode="w")
            if output_format == "srt":
                for i, seg in enumerate(segments, 1):
                    start_ts = format_timestamp_srt(seg.get("start", 0))
                    end_ts = format_timestamp_srt(seg.get("end", 0))
                    speaker_prefix = f"[{seg.get('speaker', '?')}] " if has_speakers else ""
                    tmp.write(f"{i}\n{start_ts} --> {end_ts}\n{speaker_prefix}{seg.get('text', '').strip()}\n\n")
            else:
                tmp.write("WEBVTT\n\n")
                for seg in segments:
                    start_ts = format_timestamp_vtt(seg.get("start", 0))
                    end_ts = format_timestamp_vtt(seg.get("end", 0))
                    speaker_prefix = f"[{seg.get('speaker', '?')}] " if has_speakers else ""
                    tmp.write(f"{start_ts} --> {end_ts}\n{speaker_prefix}{seg.get('text', '').strip()}\n\n")
            tmp.close()
            subtitle_file = tmp.name
        elif output_format == "json":
            # Match the schema produced by the initial JSON gen in phase 6
            # (language, duration, per-word timestamps + confidence).
            json_data = {
                "language": _last_result.get("language", ""),
                "duration": round(_last_result.get("duration", 0.0), 2),
                "segments": [],
            }
            for seg in segments:
                seg_out = {
                    "start": round(seg.get("start", 0), 3),
                    "end": round(seg.get("end", 0), 3),
                    "text": seg.get("text", "").strip(),
                }
                if has_speakers:
                    seg_out["speaker"] = seg.get("speaker", "?")
                words = seg.get("words", [])
                if words:
                    seg_out["words"] = []
                    for w in words:
                        word_out = {"word": w.get("word", "").strip()}
                        if "start" in w:
                            word_out["start"] = round(w["start"], 3)
                        if "end" in w:
                            word_out["end"] = round(w["end"], 3)
                        if "score" in w:
                            word_out["confidence"] = round(w["score"], 3)
                        if has_speakers and "speaker" in w:
                            word_out["speaker"] = w["speaker"]
                        seg_out["words"].append(word_out)
                json_data["segments"].append(seg_out)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            json.dump(json_data, tmp, ensure_ascii=False, indent=2)
            tmp.close()
            subtitle_file = tmp.name

        return html, plain, subtitle_file

    apply_rename_btn.click(
        fn=_apply_speaker_renames,
        inputs=[speaker_rename_input],
        outputs=[output_html, output_text, output_file],
    )

    # Track the previous upload so we can clean it up when a new file arrives
    _previous_upload = {"path": None}

    def on_file_change(f):
        # Clean up the previous upload when a new file is uploaded
        prev = _previous_upload["path"]
        if prev and prev != f:
            cleanup_upload(prev)
        _previous_upload["path"] = f

        if f is not None:
            try:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                fname = os.path.basename(f)
                log.info(f"File uploaded: {fname} ({size_mb:.1f} MB)")
            except Exception:
                log.info(f"File uploaded: {f}")
        else:
            log.info("File cleared")
        return gr.update(interactive=f is not None)

    file_input.change(
        fn=on_file_change,
        inputs=[file_input],
        outputs=[transcribe_btn],
    )

    all_inputs = [file_input, local_path_input, model_dropdown, lang_dropdown, format_dropdown, diarize_checkbox, min_speakers_input, max_speakers_input, batch_slider, hotwords_input, initial_prompt_input, suppress_numerals_input]
    all_outputs = [status_text, output_html, output_text, output_file]

    notification_js = """(status) => {
        if (!status || !status.startsWith("Done --")) return;
        if ("Notification" in window && Notification.permission === "granted") {
            new Notification("Transcription Complete", {
                body: status,
                icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙</text></svg>"
            });
        }
    }"""

    # Cancel handler
    def _request_cancel():
        _cancel_requested["value"] = True
        return "Cancelling..."

    cancel_btn.click(fn=_request_cancel, outputs=[status_text])

    # Request notification permission on first upload
    file_input.upload(
        fn=None,
        js="""() => {
            if ("Notification" in window && Notification.permission === "default") {
                Notification.requestPermission();
            }
        }""",
    )
    # Disable the transcribe button while a job is running so the user
    # can't accidentally start a second one (the lock would reject it with
    # "Busy" but the UX is poor).
    def _disable_transcribe():
        return gr.update(interactive=False, value="Transcribing...")

    def _enable_transcribe():
        return gr.update(interactive=True, value="Transcribe")

    # Auto-transcribe on upload
    upload_event = (
        file_input.upload(fn=_disable_transcribe, outputs=[transcribe_btn])
        .then(fn=transcribe, inputs=all_inputs, outputs=all_outputs)
        .then(fn=_enable_transcribe, outputs=[transcribe_btn])
    )
    upload_event.then(
        fn=None,
        inputs=[status_text],
        js=notification_js,
    ).then(
        fn=_show_speaker_rename,
        inputs=[status_text],
        outputs=[speaker_accordion, speaker_rename_input],
    )

    # Manual re-transcribe button (for changing settings on an already-uploaded file)
    transcribe_event = (
        transcribe_btn.click(fn=_disable_transcribe, outputs=[transcribe_btn])
        .then(fn=transcribe, inputs=all_inputs, outputs=all_outputs)
        .then(fn=_enable_transcribe, outputs=[transcribe_btn])
    )
    transcribe_event.then(
        fn=None,
        inputs=[status_text],
        js=notification_js,
    ).then(
        fn=_show_speaker_rename,
        inputs=[status_text],
        outputs=[speaker_accordion, speaker_rename_input],
    )

    # Cancel aborts running transcription events AND re-enables the button
    cancel_btn.click(fn=None, cancels=[upload_event, transcribe_event])
    cancel_btn.click(fn=_enable_transcribe, outputs=[transcribe_btn])

    # -- YouTube fetch handler (registered here so status_text exists) --
    def _yt_download(url):
        """Download YouTube audio. Returns (path, status_msg)."""
        url = (url or "").strip()
        if not url:
            return "", "Enter a YouTube URL first"
        output_dir = tempfile.mkdtemp(prefix="yt-dlp-")
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp", "--no-playlist",
            "--remote-components", "ejs:github",
            *_yt_dlp_auth_args(),
            "-f", "bestaudio",
            "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
            "--print-json",
            "-o", output_template,
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return "", "yt-dlp timed out after 600s"
        except Exception as e:
            return "", f"yt-dlp error: {e}"
        if result.returncode != 0:
            err = result.stderr.strip()[:200]
            if _is_permanent_yt_dlp_error(result.stderr):
                err = f"unrecoverable: {err}"
            return "", f"yt-dlp failed: {err}"
        try:
            meta = json.loads(result.stdout.strip().split("\n")[-1])
        except Exception:
            return "", "Failed to parse yt-dlp output"
        video_id = meta.get("id", "unknown")
        filename = meta.get("requested_downloads", [{}])[0].get("filepath", "")
        if not filename or not os.path.isfile(filename):
            # Probe output_dir for the actual file (any extension).
            candidates = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith(video_id) and not f.endswith(".info.json")
                   and not f.endswith(".part")
            ]
            if candidates:
                filename = max(candidates, key=os.path.getsize)
            else:
                return "", f"yt-dlp produced no file in {output_dir}"
        title = meta.get("title", "unknown")
        duration = meta.get("duration", 0)
        log.info(f"[UI] YT download: {filename} ({duration}s)")
        return filename, f"Downloaded '{title}' ({duration}s)"

    def _yt_fetch_and_transcribe(url, _file, _local, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords, initial_prompt, suppress_numerals):
        """Download YT audio then transcribe (same as clicking Fetch + Transcribe)."""
        yield "Downloading from YouTube...", "", "", None
        path, msg = _yt_download(url)
        if not path:
            yield msg, "", "", None
            return
        yield msg + " — transcribing...", "", "", None
        # Run transcription with downloaded file as local_path
        for result in transcribe(None, path, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords, initial_prompt, suppress_numerals):
            yield result

    yt_fetch_btn.click(
        fn=_yt_fetch_and_transcribe,
        inputs=[yt_url_input] + all_inputs,
        outputs=all_outputs,
    )

    # -- History section --
    with gr.Accordion("History", open=False):
        history_html = gr.HTML(value=_format_history_html())
        refresh_history_btn = gr.Button("Refresh", size="sm", variant="secondary")
        refresh_history_btn.click(fn=_format_history_html, outputs=[history_html])

    # Also refresh history after transcription completes
    upload_event.then(fn=_format_history_html, outputs=[history_html])
    transcribe_event.then(fn=_format_history_html, outputs=[history_html])

# ─── Job queue (Valkey-backed) ───────────────────────────────────────────────
# Server-side queue + shared transcript cache. Backs /api/jobs* — any
# consumer (Discord bot, MCP, Gradio UI, ad-hoc curl) submits a job and
# polls for results instead of fighting over a single sync lock with 409s.
#
# Schema (Valkey keys):
#   queue:waiting       LIST   job_ids in FIFO order
#   jobs:active         SET    job_ids currently running (size ≤ WORKER_CONCURRENCY)
#   jobs:{job_id}       HASH   {id, status, payload, result, error, ...}
#   jobs:recent         LIST   bounded to last JOB_RECENT_LIMIT terminal jobs
#   transcripts:{key}   STRING JSON-encoded shared transcript cache
#
# Cache key: sha1(file_bytes) + model + language + diarize. Different
# decode settings produce different transcripts, so they don't collide.
# Filename is NOT part of the key — same audio under different paths
# shares cache (helpful when yt-dlp temp dirs vary across runs).
#
# Resilience: if valkey is unreachable we set `_queue_available=False`,
# /api/jobs* return 503, and /api/transcribe (sync) still works as a
# fallback. The lock-based path is therefore preserved as a hot spare.

VALKEY_URL = os.environ.get("VALKEY_URL", "redis://valkey:6379/0")
TRANSCRIPT_CACHE_TTL = int(os.environ.get("TRANSCRIPT_CACHE_TTL", "604800"))   # 7 days
JOB_TTL = int(os.environ.get("JOB_TTL", "3600"))                               # 1h after terminal
JOB_RECENT_LIMIT = int(os.environ.get("JOB_RECENT_LIMIT", "100"))
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "1"))            # single GPU = 1

_valkey = None
_valkey_async = None
_queue_available = False


def _init_valkey() -> bool:
    """Connect to Valkey. Mutates module-level clients + `_queue_available`.
    Returns True on success, False on failure (logged). Called by the
    lifespan startup hook and by tests via the same path.
    """
    global _valkey, _valkey_async, _queue_available
    try:
        import valkey as _vk
        import valkey.asyncio as _vk_async
        _valkey = _vk.from_url(VALKEY_URL, decode_responses=True)
        _valkey.ping()
        _valkey_async = _vk_async.from_url(VALKEY_URL, decode_responses=True)
        _queue_available = True
        log.info(f"Valkey connected at {VALKEY_URL}")
        return True
    except Exception as e:
        log.warning(
            f"Valkey unavailable ({VALKEY_URL}): {e} — "
            f"/api/jobs disabled; /api/transcribe sync fallback only"
        )
        _queue_available = False
        return False


# ── Job state ────────────────────────────────────────────────────────────────


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"
_JOB_TERMINAL = {JOB_STATUS_DONE, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}


class PermanentJobError(Exception):
    """Marker — job failed in a way that won't recover on retry (missing
    input file, oversized input). Stored as `permanent: 1` on the job hash
    so clients can fail fast instead of resubmitting."""


def _new_job_id() -> str:
    """Opaque job id. `wbx_` prefix ties to whisper-transcribe origin."""
    return f"wbx_{uuid.uuid4().hex[:12]}"


def _job_key(job_id: str) -> str:
    return f"jobs:{job_id}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _deserialize_job(data: dict) -> dict:
    """Decode JSON-encoded fields embedded in a job hash."""
    out = dict(data)
    for k in ("payload", "result"):
        if k in out and out[k]:
            try:
                out[k] = json.loads(out[k])
            except Exception:
                pass
    # permanent is stored as "1"/"0" string; coerce to bool for clients.
    if "permanent" in out:
        out["permanent"] = out["permanent"] == "1"
    return out


async def _enqueue_job(payload: dict, consumer: str = "unknown") -> dict:
    """Create + enqueue a transcription job. Returns the initial state dict.

    The caller does NOT need to pre-validate the payload: the worker
    re-checks on dequeue (file may have moved between submit and run).
    Validation here is intentionally minimal to keep submit fast.
    """
    if not _queue_available:
        raise RuntimeError("queue backend unavailable")
    job_id = _new_job_id()
    now = _now_iso()
    state = {
        "id": job_id,
        "status": JOB_STATUS_QUEUED,
        "consumer": consumer,
        "submitted_at": now,
        "payload": json.dumps(payload),
    }
    # Atomic: enqueue + state-hash write happen together so polls never see
    # a job in the LIST without its hash (or vice versa).
    pipe = _valkey_async.pipeline()
    pipe.hset(_job_key(job_id), mapping=state)
    pipe.rpush("queue:waiting", job_id)
    await pipe.execute()
    log.info(f"[queue] enqueued {job_id} from {consumer}: {payload.get('file_path', '?')}")
    return {**state, "payload": payload}


async def _read_job(job_id: str) -> dict | None:
    """Fetch a job's current state. None if not found / queue down."""
    if not _queue_available:
        return None
    data = await _valkey_async.hgetall(_job_key(job_id))
    if not data:
        return None
    return _deserialize_job(data)


async def _job_position(job_id: str) -> int | None:
    """Position in queue (1-indexed, 1 = next). None if not queued."""
    if not _queue_available:
        return None
    # LPOS is O(N) but N is bounded by the queue depth which we keep small.
    # decode_responses=True means LPOS returns int or None directly.
    try:
        idx = await _valkey_async.lpos("queue:waiting", job_id)
        return None if idx is None else idx + 1
    except Exception:
        return None


async def _cancel_job(job_id: str) -> tuple[bool, str]:
    """Cancel a queued job. Returns (success, reason).

    Running jobs can't be cancelled — whisperX has no checkpoint protocol,
    so interrupting mid-decode would leak GPU memory. Caller can
    fire-and-forget on their side if they no longer want the result.
    """
    if not _queue_available:
        return False, "queue backend unavailable"
    job = await _read_job(job_id)
    if not job:
        return False, "not found"
    status = job.get("status")
    if status in _JOB_TERMINAL:
        return False, f"already {status}"
    if status == JOB_STATUS_RUNNING:
        return False, "cannot cancel in-flight job"
    # Status is queued — remove from LIST and mark cancelled. LREM is O(N)
    # but again N is small.
    removed = await _valkey_async.lrem("queue:waiting", 1, job_id)
    if removed:
        await _valkey_async.hset(_job_key(job_id), mapping={
            "status": JOB_STATUS_CANCELLED,
            "completed_at": _now_iso(),
        })
        await _valkey_async.expire(_job_key(job_id), JOB_TTL)
        return True, "cancelled"
    return False, "race: worker picked it up between read and cancel"


async def _queue_info() -> dict:
    """Snapshot of queue state for /api/queue."""
    if not _queue_available:
        return {"depth": 0, "active": [], "recent": [], "available": False}
    depth = await _valkey_async.llen("queue:waiting")
    active_ids = await _valkey_async.smembers("jobs:active")
    recent_ids = await _valkey_async.lrange("jobs:recent", 0, 19)
    # Fetch hashes in parallel — gather is significantly faster than a
    # sequential await loop when there are several active/recent jobs.
    active_jobs = await asyncio.gather(*(_read_job(jid) for jid in active_ids))
    recent_jobs = await asyncio.gather(*(_read_job(jid) for jid in recent_ids))
    return {
        "depth": depth,
        "active": [j for j in active_jobs if j],
        "recent": [j for j in recent_jobs if j],
        "available": True,
    }


# ── Transcript cache ─────────────────────────────────────────────────────────


def _sha1_file(path: str) -> str:
    """Streaming sha1 of a file. 1 MiB blocks — keeps memory flat for the
    multi-GB audio files common with long-form video downloads."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _transcript_cache_key(file_path: str, model: str, language: str,
                          diarize: bool, task: str = "transcribe") -> str:
    """Cache key includes decode settings. Same audio under different paths
    shares cache; same audio with different settings does not.

    `task` is part of the key because transcribe and translate produce
    different transcripts from the same audio (Whisper task=translate
    outputs English regardless of source). They must not collide.

    Caller is responsible for offloading to a thread — this reads the
    entire file synchronously (multi-GB audio = multi-second I/O block).
    See _execute_transcription for the to_thread wrap.
    """
    file_hash = _sha1_file(file_path)
    return f"transcripts:{file_hash}:{model}:{language}:{int(bool(diarize))}:{task}"


def _decode_first_seconds(file_path: str, seconds: int = 30, sr: int = 16000) -> "np.ndarray":
    """Decode only the first `seconds` of audio via ffmpeg `-t`. Much faster
    than whisperx.load_audio for long files — a 4hr WAV decodes in ~5-10s
    end-to-end, but we only need 30s of it for LID. Cuts that to ~200ms.

    Returns a numpy float32 array (matches whisperx.load_audio's contract).
    """
    import numpy as np
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-t", str(seconds),     # cap input duration BEFORE -i for faster seek/exit
        "-i", file_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0


def _quick_detect_language_sync(file_path: str) -> tuple[str, float]:
    """30s LID pre-pass via faster-whisper. Returns (lang_code, confidence).

    Cheap relative to a full transcription — single encoder forward pass on
    the first 30s of audio plus the 99-language token softmax. Used by the
    translate="auto" heuristic in _execute_transcription to decide between
    task=transcribe (English source) and task=translate (non-English source,
    including code-switched audio where the dominant language is non-English).

    Reuses the currently-loaded whisper model — if none is loaded, loads
    turbo (cheap, supports all 100 languages including yue Cantonese).
    Decodes only the first 30s of audio via ffmpeg's `-t` flag — for a 4hr
    video this is ~200ms instead of ~10s of full-file decoding.

    Returns ("unknown", 0.0) on any failure; caller treats that as "fall
    back to language=None whisperx auto-detect" (the pre-existing behaviour).
    """
    try:
        # Ensure a model is loaded; reuse whatever's already resident.
        if whisper_model is None:
            load_whisper("turbo")
        first_30s = _decode_first_seconds(file_path, seconds=30)
        # faster-whisper signature: (audio=None, features=None, vad_filter,
        # vad_parameters, language_detection_segments, language_detection_threshold)
        # → (lang, prob, all_probs)
        lang, prob, _all = whisper_model.model.detect_language(first_30s)
        return lang, float(prob)
    except Exception as e:
        log.warning(f"[quick-LID] failed: {e}; falling back to whisperx auto-detect")
        return "unknown", 0.0


async def _quick_detect_language(file_path: str) -> tuple[str, float]:
    """Async wrapper around _quick_detect_language_sync. Offloaded to a
    thread because both audio I/O and model inference are blocking."""
    return await asyncio.to_thread(_quick_detect_language_sync, file_path)


def _decide_task(translate: object, language: str | None,
                 quick_lang: str | None = None, quick_conf: float = 0.0) -> str:
    """Resolve the `translate` payload field to a Whisper `task` value.

    `translate` is one of: True, False, or "auto" (default). Resolution:
      - True  → "translate" unconditionally
      - False → "transcribe" unconditionally
      - "auto":
          - If `language` was explicitly set (not "Auto-detect"/None/"") →
            respect it, use "transcribe". User who picked a specific
            language wants that language's output.
          - Else if quick-LID confidence < 0.5 → "translate". Threshold
            matches faster-whisper's `language_detection_threshold` default;
            below that, LID is "not sure" — code-switched audio often
            produces low-confidence single-language LID, and translate is
            the safer default for the summarisation use case (CS-FLEURS:
            BLEU barely degrades on CS audio).
          - Else if high-confidence English → "transcribe" (no point
            translating English to English).
          - Else (high-confidence non-English) → "translate".
    """
    if translate is True:
        return "translate"
    if translate is False:
        return "transcribe"
    # "auto" (or anything else falsy/unknown — treat as auto)
    explicit = language and language not in ("Auto-detect", "auto", "")
    if explicit:
        return "transcribe"
    # Confidence check FIRST so that a low-confidence "en" (often CS audio
    # leaning English) gets translate, not transcribe.
    if quick_lang == "unknown" or quick_conf < 0.5:
        return "translate"
    if quick_lang == "en":
        return "transcribe"
    return "translate"


async def _cache_get_transcript(key: str) -> dict | None:
    if not _queue_available:
        return None
    try:
        blob = await _valkey_async.get(key)
    except Exception as e:
        log.warning(f"transcript cache read failed: {e}")
        return None
    if not blob:
        return None
    try:
        return json.loads(blob)
    except Exception:
        return None


async def _cache_set_transcript(key: str, result: dict) -> None:
    if not _queue_available:
        return
    try:
        await _valkey_async.set(key, json.dumps(result), ex=TRANSCRIPT_CACHE_TTL)
    except Exception as e:
        log.warning(f"transcript cache write failed: {e}")


# ── Worker loop ──────────────────────────────────────────────────────────────


_worker_tasks: list = []


async def _job_worker(worker_id: int):
    """Single-worker loop. BLPOP from queue:waiting → run → repeat. One
    instance per GPU. A poisoned job logs + records failure but never
    crashes the worker.
    """
    # valkey-py raises valkey.exceptions.TimeoutError (a subclass of
    # the built-in TimeoutError) when its temporary socket read-deadline
    # for BLPOP fires before the server returns nil-on-empty. On a
    # completely idle queue this is normal behaviour, NOT an error — we
    # catch it specifically and continue silently. The wider `Exception`
    # catch below is reserved for actual problems (valkey down, protocol
    # error, poisoned response).
    import valkey.exceptions as _vk_exc  # local import — symbol only used here
    log.info(f"[worker {worker_id}] started")
    while True:
        try:
            popped = await _valkey_async.blpop("queue:waiting", timeout=5)
            if not popped:
                continue
            _key, job_id = popped
            await _run_one_job(job_id)
        except asyncio.CancelledError:
            log.info(f"[worker {worker_id}] cancelled")
            raise
        except (_vk_exc.TimeoutError, TimeoutError):
            # Expected on idle queue — the client's socket read-deadline
            # for the BLPOP fired. Just loop and BLPOP again.
            continue
        except Exception as e:
            log.error(f"[worker {worker_id}] outer loop error: {e}")
            traceback.print_exc()
            await asyncio.sleep(2)  # tiny backoff so we don't tight-loop on the same error


async def _run_one_job(job_id: str):
    """Dispatch one job end-to-end. Handles cache lookup, state transitions,
    failure recording, file cleanup. Never raises — failures are recorded
    on the job hash so clients see them via /api/jobs/{id}."""
    job = await _read_job(job_id)
    if not job:
        log.warning(f"[worker] {job_id} popped but hash missing — dropping")
        return
    # Already-terminal guard. Two paths can land us here:
    #   1. Cancelled between BLPOP and this read (rare — cancel's LREM would
    #      normally fail because we already popped, but the race exists).
    #   2. Duplicate enqueue from crash recovery scaling out beyond one
    #      whisper container (we don't today, but defense-in-depth is cheap).
    # Either way: skip silently, the result is already authoritative.
    if job.get("status") in _JOB_TERMINAL:
        log.info(f"[worker] {job_id} already {job.get('status')} — skipping")
        return
    payload = job.get("payload") or {}
    await _valkey_async.sadd("jobs:active", job_id)
    await _valkey_async.hset(_job_key(job_id), mapping={
        "status": JOB_STATUS_RUNNING,
        "started_at": _now_iso(),
    })
    log.info(f"[worker] running {job_id}: {payload.get('file_path', '?')}")
    try:
        result = await _execute_transcription(payload)
        await _valkey_async.hset(_job_key(job_id), mapping={
            "status": JOB_STATUS_DONE,
            "result": json.dumps(result),
            "completed_at": _now_iso(),
        })
        log.info(f"[worker] done {job_id}: {str(result.get('status', ''))[:80]}")
    except Exception as e:
        log.error(f"[worker] failed {job_id}: {e}")
        traceback.print_exc()
        permanent = isinstance(e, PermanentJobError)
        await _valkey_async.hset(_job_key(job_id), mapping={
            "status": JOB_STATUS_FAILED,
            "error": str(e),
            "permanent": "1" if permanent else "0",
            "completed_at": _now_iso(),
        })
    finally:
        await _valkey_async.srem("jobs:active", job_id)
        await _valkey_async.lpush("jobs:recent", job_id)
        await _valkey_async.ltrim("jobs:recent", 0, JOB_RECENT_LIMIT - 1)
        await _valkey_async.expire(_job_key(job_id), JOB_TTL)


async def _execute_transcription(payload: dict) -> dict:
    """Run the actual whisper transcription, with cache-hit short-circuit."""
    file_path = (payload.get("file_path") or "").strip()
    if not file_path or not os.path.isfile(file_path):
        raise PermanentJobError(f"file not found: {file_path}")

    model = payload.get("model", "turbo")
    language = payload.get("language", "Auto-detect")
    diarize = bool(payload.get("diarize", False))

    # Resolve the `translate` payload field → Whisper `task`. The "auto"
    # path runs a cheap 30s LID pre-pass and translates when source is
    # non-English (the summarisation use case). Explicit True/False
    # respects the caller's intent and skips the pre-pass.
    translate = payload.get("translate", "auto")
    quick_lang, quick_conf = "unknown", 0.0
    if translate == "auto" and not (language and language not in ("Auto-detect", "auto", "")):
        # Only run the pre-pass when we actually need its output.
        quick_lang, quick_conf = await _quick_detect_language(file_path)
        log.info(f"[worker] quick-LID: lang={quick_lang} conf={quick_conf:.2f}")
    task = _decide_task(translate, language, quick_lang, quick_conf)
    log.info(f"[worker] task={task} (translate={translate!r}, language={language!r})")

    # Cache lookup before whisper runs — different consumers transcribing
    # the same video share results. Hash the file in a thread; for multi-GB
    # audio sync I/O would block the event loop for several seconds and
    # stall /api/status / /api/queue polls from other clients.
    #
    # `refresh=true` on the payload bypasses the lookup (user explicitly
    # asked for a fresh run via /summarize refresh:true or curl). We still
    # WRITE the result to the cache on success so subsequent non-refresh
    # runs benefit.
    cache_key = await asyncio.to_thread(
        _transcript_cache_key, file_path, model, language, diarize, task
    )
    refresh = bool(payload.get("refresh", False))
    cached = None if refresh else await _cache_get_transcript(cache_key)
    if refresh:
        log.info(f"[worker] cache bypass (refresh=true) for {cache_key[:40]}...")
    if cached:
        log.info(f"[worker] transcript cache hit: {cache_key[:40]}...")
        # subtitle_file is intentionally NOT cached (ephemeral path). Null
        # it on cache hit so callers don't try to read a deleted file. If
        # they need an SRT they can re-run with `cleanup=false` on a fresh
        # file or generate one client-side from the transcript text.
        result = {**cached, "cached": True, "subtitle_file": None, "task": task}
        if payload.get("cleanup"):
            _cleanup_payload_file(file_path)
        return result

    return_file = bool(payload.get("return_file", True))

    # Run whisper in a thread so the asyncio loop stays responsive
    # (status endpoints, queue polls, healthcheck). The transcription
    # itself is CPU/GPU bound and synchronous.
    result = await asyncio.to_thread(
        _run_transcription,
        file_path,
        model_name=model,
        language=language,
        output_format=payload.get("format", "txt"),
        enable_diarization=diarize,
        min_speakers=payload.get("min_speakers", 0),
        max_speakers=payload.get("max_speakers", 0),
        batch_size=payload.get("batch_size"),
        hotwords=payload.get("hotwords", ""),
        initial_prompt=payload.get("initial_prompt", ""),
        suppress_numerals=payload.get("suppress_numerals", False),
        return_file=return_file,
        task=task,
    )
    result["cached"] = False

    # Cache successful runs only. Whisper returns "Error: ..." on internal
    # failures (CUDA OOM, prompt overflow) — don't poison cache with those.
    # Cache the reproducible fields only — subtitle_file is an ephemeral
    # /tmp path that won't exist on the next cache hit. `task` is part of
    # the cache key so transcribe/translate don't collide; storing it on
    # the value too lets clients see which task produced a cache hit.
    status = result.get("status", "") or ""
    if status and not status.lower().startswith("error"):
        cacheable = {
            "status": status,
            "transcript": result.get("transcript", ""),
            "task": task,
        }
        await _cache_set_transcript(cache_key, cacheable)

    # Reclaim the subtitle file when caller said it didn't want one. Same
    # rationale as the legacy /api/transcribe path: avoid the leak window if
    # the response is dropped + skip the disk write on a per-call basis.
    if not return_file:
        subtitle_file = result.get("subtitle_file")
        if subtitle_file and os.path.isfile(subtitle_file):
            try:
                os.remove(subtitle_file)
                log.info(f"[worker] cleaned subtitle file (return_file=false): {subtitle_file}")
            except OSError as e:
                log.warning(f"[worker] subtitle cleanup failed: {e}")

    if payload.get("cleanup"):
        _cleanup_payload_file(file_path)

    return result


def _cleanup_payload_file(file_path: str) -> None:
    """Remove the source file + its yt-dlp parent dir if empty. Synchronous
    — file system, not network — so safe to call from async paths."""
    if not os.path.isfile(file_path):
        return
    try:
        parent = os.path.dirname(file_path)
        os.remove(file_path)
        if parent.startswith("/tmp/yt-dlp-") and not os.listdir(parent):
            os.rmdir(parent)
        log.info(f"[worker] cleaned up {file_path}")
    except Exception as e:
        log.warning(f"[worker] cleanup failed for {file_path}: {e}")


async def _recover_active_jobs():
    """On startup, re-queue any job that was running at last shutdown.

    Single-worker single-GPU = safe to retry; whisperx is idempotent.
    Without this, a crash mid-transcription leaves the job stuck in
    'running' forever and clients poll it indefinitely.
    """
    if not _queue_available:
        return
    active = await _valkey_async.smembers("jobs:active")
    if not active:
        return
    log.warning(f"[recovery] {len(active)} job(s) running at last shutdown — re-queueing")
    for job_id in active:
        await _valkey_async.srem("jobs:active", job_id)
        await _valkey_async.hset(_job_key(job_id), mapping={"status": JOB_STATUS_QUEUED})
        await _valkey_async.hdel(_job_key(job_id), "started_at")
        # LPUSH (not RPUSH) so recovered jobs run first — they were already
        # at the front before the crash, fairness-wise.
        await _valkey_async.lpush("queue:waiting", job_id)


async def _worker_startup():
    """Lifespan startup. Connects to valkey, recovers stale active jobs,
    spawns WORKER_CONCURRENCY worker tasks."""
    if not _init_valkey():
        return
    await _recover_active_jobs()
    for i in range(max(1, WORKER_CONCURRENCY)):
        task = asyncio.create_task(_job_worker(i))
        _worker_tasks.append(task)
    log.info(f"[queue] {len(_worker_tasks)} worker(s) started")


async def _worker_shutdown():
    """Lifespan shutdown. Cancels workers; in-flight job (if any) will be
    re-queued by _recover_active_jobs on the next start."""
    for task in _worker_tasks:
        task.cancel()
    for task in _worker_tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    _worker_tasks.clear()


# -- HTTP API (for MCP server / programmatic access) --------------------------
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def api_status(request: Request):
    """GET /api/status — GPU info, ready state, capabilities.

    `busy` reflects whether a transcription is currently running. Clients
    (e.g. the Discord bot) poll this before submitting to /api/transcribe
    so they can wait for the GPU instead of burning retry budget against
    repeated 409s. The 409 contract on /api/transcribe is unchanged — this
    is purely an opportunistic pre-flight check (race-safe because callers
    still handle 409 on the actual submit).
    """
    return JSONResponse({
        "status": "ready",
        "busy": _transcription_lock.locked(),
        "gpu": GPU_INFO_STR,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "diarization_available": DIARIZATION_AVAILABLE,
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "vision": {
            "available": True,    # endpoint always exists; VLM call may fail
            "model": LLM_VISION_MODEL,
            "api_url": LLM_VISION_API_URL,
            "fps_interval": VLM_FPS_INTERVAL,
            "max_frames": VLM_MAX_FRAMES,
        },
    })


async def api_yt_download(request: Request):
    """POST /api/yt-download — download audio (and optionally video) via yt-dlp.

    Body fields:
      url:               str   (required) — YouTube/etc. URL
      format:            str   — yt-dlp format spec (default depends on keep_video)
      keep_video:        bool  (default false) — when true, downloads a low-res
                         video stream alongside the audio so /api/describe can
                         extract frames later. When false, audio-extracted to WAV
                         (smaller; the legacy default for transcription-only).
      playlist:          bool  (default false) — process as playlist
      max_height:        int   (default VIDEO_DOWNLOAD_MAX_HEIGHT, currently 480)
                         — cap on video resolution when keep_video=true. Frames are
                         downscaled to VLM_FRAME_WIDTH for inference anyway, so
                         higher resolution wastes bandwidth.
      include_comments:  bool  (default false) — when true, also fetches the
                         video's top YouTube comments (yt-dlp `--get-comments`)
                         and includes them in the response. Adds ~5-30s to the
                         download depending on `comments_max`. Each comment is:
                         {text, author, like_count, is_pinned, is_favorited,
                          author_is_uploader, parent, time_text}.
      comments_max:      int   (default 100) — cap on comments fetched. yt-dlp
                         pulls top-level comments first, then dives into
                         replies up to this total.
      comments_sort:     str   (default "top") — yt-dlp comment sort order:
                         "top" (default, by likes) or "new".

    Returns (single video): {"filename", "title", "duration", optionally "comments": [...]}
    Returns (playlist):     {"items": [{...}, ...], "count": N}
    """
    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    keep_video = bool(body.get("keep_video", False))
    playlist = body.get("playlist", False)
    max_height = int(body.get("max_height", os.environ.get("VIDEO_DOWNLOAD_MAX_HEIGHT", "480")))
    include_comments = bool(body.get("include_comments", False))
    comments_max = int(body.get("comments_max", 100))
    comments_sort = body.get("comments_sort", "top")
    output_dir = tempfile.mkdtemp(prefix="yt-dlp-")

    # Format spec depends on whether the caller will need video frames later.
    # keep_video=false (default, legacy): pure audio → WAV via --extract-audio.
    # keep_video=true (VLM-enabled bot): low-res video + best audio in a single
    # container; whisperX's ffmpeg pipeline transcodes the audio in-memory
    # at transcribe time, and /api/describe extracts frames from the same file.
    if keep_video:
        default_fmt = (f"bestvideo[height<={max_height}]+bestaudio/"
                       f"best[height<={max_height}]/best")
    else:
        default_fmt = "bestaudio"
    fmt = body.get("format", default_fmt)

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        # Allow yt-dlp to auto-fetch the JS challenge solver script from GitHub.
        # Required for YouTube Music / signature-protected streams as of 2026.
        "--remote-components", "ejs:github",
        *_yt_dlp_auth_args(),
        "-f", fmt,
    ]
    if not keep_video:
        cmd.extend([
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
        ])
    if include_comments:
        # `--write-info-json` ensures the comments end up in <id>.info.json
        # on disk (yt-dlp embeds them when --get-comments is set). We parse
        # that file below — `--print-json` output sometimes truncates large
        # comment arrays in stdout.
        cmd.extend([
            "--write-info-json",
            "--get-comments",
            "--extractor-args",
            f"youtube:max_comments={comments_max};comment_sort={comments_sort}",
        ])
    cmd.extend([
        "--print-json",
        "-o", output_template,
        url,
    ])
    if not playlist:
        cmd.insert(1, "--no-playlist")
    log.info(f"[API] yt-dlp download: {url} (playlist={playlist}, "
             f"comments={include_comments and comments_max or 0})")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            log.error(f"[API] yt-dlp failed: {result.stderr[:500]}")
            # 422 (Unprocessable Entity) for permanent errors — clients
            # (the bot) treat 4xx as PermanentError and skip retries.
            status_code = 422 if _is_permanent_yt_dlp_error(result.stderr) else 500
            return JSONResponse(
                {"error": f"yt-dlp failed: {result.stderr[:300]}",
                 "permanent": status_code == 422},
                status_code=status_code,
            )

        # Parse output — one JSON object per line per video. yt-dlp's
        # `requested_downloads[0].filepath` is the canonical answer when
        # populated, but for some formats (notably merged video+audio with
        # post-processing) it can be missing or stale. Falls back to
        # probing `output_dir` for the actual file — extension-agnostic so
        # both .wav (audio-only) and .mp4 (keep_video=true) work.
        json_lines = [l for l in result.stdout.strip().split("\n") if l.strip().startswith("{")]
        items = []
        for line in json_lines:
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = meta.get("id", "unknown")
            filename = meta.get("requested_downloads", [{}])[0].get("filepath", "")
            if not filename or not os.path.isfile(filename):
                # Find any file matching the video id in output_dir,
                # regardless of extension.
                candidates = [
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.startswith(video_id) and not f.endswith(".info.json")
                       and not f.endswith(".part")
                ]
                if candidates:
                    # Prefer largest (the actual media file vs. e.g. thumbnails)
                    filename = max(candidates, key=os.path.getsize)
                    log.info(f"[API] yt-dlp metadata missing filepath; "
                             f"resolved {video_id} → {filename}")
                else:
                    log.warning(f"[API] no file found in {output_dir} for {video_id}")
                    continue
            item = {
                "filename": filename,
                "title": meta.get("title", "unknown"),
                "duration": meta.get("duration", 0),
                # Livestream signal — yt-dlp sets `was_live: true` for VOD'd
                # streams, `live_status` is one of "not_live"|"is_live"|
                # "was_live"|"post_live" etc. Bot uses this to skip VLM
                # enrichment on livestreams (long quiet stretches are normal
                # gameplay/music, not "silent video" in the VLM sense).
                "was_live": bool(meta.get("was_live", False)),
                "live_status": meta.get("live_status", "") or "",
            }
            if include_comments:
                item["comments"] = _extract_comments(meta, output_dir, video_id)
            items.append(item)

        if not items:
            return JSONResponse({"error": "yt-dlp produced no output"}, status_code=500)

        # Single video: return flat response for backward compat
        if len(items) == 1:
            item = items[0]
            log.info(f"[API] Downloaded: {item['filename']} ({item['duration']}s)")
            return JSONResponse(item)

        # Playlist: return list
        log.info(f"[API] Downloaded playlist: {len(items)} items")
        return JSONResponse({"items": items, "count": len(items)})
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "yt-dlp timed out"}, status_code=504)
    except Exception as e:
        log.error(f"[API] yt-dlp error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _run_transcription(file_path, model_name="turbo", language="Auto-detect",
                       output_format="txt", enable_diarization=False,
                       min_speakers=0, max_speakers=0, batch_size=None,
                       hotwords="", initial_prompt="", suppress_numerals=False,
                       return_file=True, task="transcribe"):
    """Run transcription synchronously, return final result dict.

    `task`: "transcribe" preserves the source language; "translate" produces
    English regardless of source. See _transcribe_inner for the multilingual
    rationale.
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    # Consume the generator to get the final result
    last_status, last_html, last_text, last_file = "", "", "", None
    for status, html, text, subtitle_file in _transcribe_inner(
        file_path, "", model_name, language, output_format,
        enable_diarization, min_speakers, max_speakers, batch_size,
        hotwords, initial_prompt, suppress_numerals,
        f"api-{int(time.time()*1000) % 100000}",
        return_file=return_file,
        task=task,
    ):
        last_status, last_html, last_text, last_file = status, html, text, subtitle_file
    return {
        "status": last_status,
        "transcript": last_text,
        "subtitle_file": last_file,
        "task": task,
    }


async def api_transcribe(request: Request):
    """POST /api/transcribe — transcribe a local file.

    DEPRECATED: prefer POST /api/jobs (async, queued, persistent). This
    endpoint is preserved for backwards compatibility with ad-hoc curl
    users; the Discord bot and MCP server have migrated to /api/jobs.
    Default behaviour now returns 202 + job_id (same as /api/jobs).
    Pass `wait: true` in the body for legacy sync behaviour.

    Body fields:
      file_path:     str   (required) path on the whisper container's filesystem
      model:         str   (default "turbo")
      language:      str   (default "Auto-detect")
      format:        str   (default "txt") txt|srt|vtt|json
      diarize:       bool  (default false)
      min_speakers:  int   (default 0 = auto)
      max_speakers:  int   (default 0 = auto)
      batch_size:    int   (default = VRAM-derived)
      hotwords:      str
      initial_prompt: str
      suppress_numerals: bool
      return_file:   bool  (default true) — set false to skip subtitle file
                     generation when caller only needs the transcript text
                     (avoids disk I/O and a leak window if the response is
                     dropped by the client).
      cleanup:       bool  (default false) remove file_path + its parent
                     yt-dlp tmp dir on completion (success or failure).
      wait:          bool  (default false) — legacy sync mode. When true,
                     blocks until the job completes and returns the result
                     inline (same shape as before). When false (default),
                     returns 202 + {job_id} and caller polls /api/jobs/{id}.

    Returns:
      wait=true:  {"status": "...", "transcript": "...", "subtitle_file": "..."}
      wait=false: 202 + {"job_id": "...", "status": "queued", "position": N}
    """
    body = await request.json()
    file_path = (body.get("file_path") or "").strip()
    if not file_path or not os.path.isfile(file_path):
        return JSONResponse({"error": f"file not found: {file_path}"}, status_code=400)

    wait = bool(body.get("wait", False))

    # Queue path: when valkey is reachable, route through the queue so all
    # consumers serialise on the same FIFO instead of fighting over a lock.
    if _queue_available:
        # Build the payload from the legacy /api/transcribe body shape. The
        # worker accepts the same keys, so this is a passthrough.
        consumer = (body.get("consumer") or
                    request.headers.get("x-consumer") or
                    "api-transcribe")
        try:
            state = await _enqueue_job(body, consumer=consumer)
        except Exception as e:
            log.error(f"[API] enqueue failed: {e}")
            return JSONResponse({"error": f"enqueue failed: {e}"}, status_code=500)
        job_id = state["id"]
        if not wait:
            position = await _job_position(job_id)
            return JSONResponse({
                "job_id": job_id,
                "status": JOB_STATUS_QUEUED,
                "position": position,
            }, status_code=202)
        # Sync mode: poll until terminal, then return the result inline.
        return await _wait_and_return_job(job_id)

    # Fallback: legacy lock-based sync path when queue is unavailable.
    # Preserves the 409 contract for ad-hoc curl users who hit a whisper
    # service running without valkey.
    return await _legacy_sync_transcribe(body, file_path)


async def _wait_and_return_job(job_id: str) -> JSONResponse:
    """Block on a job until terminal, return its result in the legacy
    /api/transcribe response shape. Used by `wait=true` callers."""
    while True:
        job = await _read_job(job_id)
        if not job:
            return JSONResponse({"error": "job vanished"}, status_code=500)
        status = job.get("status")
        if status == JOB_STATUS_DONE:
            return JSONResponse(job.get("result") or {})
        if status == JOB_STATUS_FAILED:
            permanent = job.get("permanent", False)
            return JSONResponse(
                {"error": job.get("error", "unknown"), "permanent": permanent},
                status_code=500,
            )
        if status == JOB_STATUS_CANCELLED:
            return JSONResponse({"error": "cancelled"}, status_code=499)
        await asyncio.sleep(1)


async def _legacy_sync_transcribe(body: dict, file_path: str) -> JSONResponse:
    """Pre-queue lock-based path. Kept for the case where valkey is down —
    whisper degrades to single-consumer-at-a-time but still functions."""
    if not _transcription_lock.acquire(blocking=False):
        return JSONResponse({"error": "busy — another transcription is running"}, status_code=409)

    return_file = bool(body.get("return_file", True))
    result_subtitle: str | None = None

    # Resolve translate → task (same heuristic as the queue path). The
    # legacy path doesn't share the transcript cache, so the quick-LID is
    # purely informational; we still run it to honour the auto behaviour.
    translate = body.get("translate", "auto")
    language = body.get("language", "Auto-detect")
    quick_lang, quick_conf = "unknown", 0.0
    if translate == "auto" and not (language and language not in ("Auto-detect", "auto", "")):
        quick_lang, quick_conf = await _quick_detect_language(file_path)
    task = _decide_task(translate, language, quick_lang, quick_conf)

    try:
        result = await asyncio.to_thread(
            _run_transcription,
            file_path,
            model_name=body.get("model", "turbo"),
            language=language,
            output_format=body.get("format", "txt"),
            enable_diarization=body.get("diarize", False),
            min_speakers=body.get("min_speakers", 0),
            max_speakers=body.get("max_speakers", 0),
            batch_size=body.get("batch_size"),
            hotwords=body.get("hotwords", ""),
            initial_prompt=body.get("initial_prompt", ""),
            suppress_numerals=body.get("suppress_numerals", False),
            return_file=return_file,
            task=task,
        )
        result_subtitle = result.get("subtitle_file")
        return JSONResponse(result)
    except Exception as e:
        log.error(f"[API] Transcription error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _transcription_lock.release()
        _reset_idle_timer()

        if not return_file and result_subtitle and os.path.isfile(result_subtitle):
            try:
                os.remove(result_subtitle)
                log.info(f"[API] Cleaned subtitle file (return_file=false): {result_subtitle}")
            except OSError as e:
                log.warning(f"[API] Subtitle cleanup failed: {e}")

        if body.get("cleanup") and os.path.isfile(file_path):
            try:
                parent = os.path.dirname(file_path)
                os.remove(file_path)
                if parent.startswith("/tmp/yt-dlp-") and not os.listdir(parent):
                    os.rmdir(parent)
                log.info(f"[API] Cleaned up temp file: {file_path}")
            except Exception as e:
                log.warning(f"[API] Cleanup failed for {file_path}: {e}")


# ── /api/jobs* (server-side queue) ───────────────────────────────────────────


async def api_jobs_submit(request: Request):
    """POST /api/jobs — submit a transcription job. Returns 202 + job_id.

    Same body shape as POST /api/transcribe. The submitter doesn't block;
    poll GET /api/jobs/{id} for status and result. Lets multiple consumers
    (Discord bot, MCP, Gradio UI, curl) serialise on a single FIFO instead
    of each implementing their own busy-wait against 409s.

    Optional `consumer` field (or `X-Consumer` header) tags the job for
    /api/queue visibility. Free-form string.
    """
    if not _queue_available:
        return JSONResponse(
            {"error": "queue backend unavailable; use /api/transcribe with wait=true"},
            status_code=503,
        )
    body = await request.json()
    file_path = (body.get("file_path") or "").strip()
    if not file_path:
        return JSONResponse({"error": "file_path is required"}, status_code=400)
    # We intentionally do NOT check os.path.isfile here — the worker re-checks
    # at dequeue time. Submission can race with download cleanup and we'd
    # rather let the worker emit a PermanentJobError than fail the submit.

    consumer = (body.get("consumer") or
                request.headers.get("x-consumer") or
                "unknown")
    try:
        state = await _enqueue_job(body, consumer=consumer)
    except Exception as e:
        return JSONResponse({"error": f"enqueue failed: {e}"}, status_code=500)
    position = await _job_position(state["id"])
    return JSONResponse({
        "job_id": state["id"],
        "status": state["status"],
        "submitted_at": state["submitted_at"],
        "position": position,
    }, status_code=202)


async def api_jobs_get(request: Request):
    """GET /api/jobs/{job_id} — fetch job state + result.

    Response shape varies by status:
      queued:    {status, position, submitted_at}
      running:   {status, started_at}
      done:      {status, result, completed_at}
      failed:    {status, error, permanent, completed_at}
      cancelled: {status, completed_at}
    """
    if not _queue_available:
        return JSONResponse({"error": "queue backend unavailable"}, status_code=503)
    job_id = request.path_params["job_id"]
    job = await _read_job(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    # Add live position for queued jobs (depth-dependent, can't be cached).
    if job.get("status") == JOB_STATUS_QUEUED:
        job["position"] = await _job_position(job_id)
    return JSONResponse(job)


async def api_jobs_cancel(request: Request):
    """DELETE /api/jobs/{job_id} — cancel a queued job.

    Only queued jobs can be cancelled — in-flight transcription has no
    safe interruption point in whisperX. Callers who no longer want a
    running job's result can just stop polling.
    """
    if not _queue_available:
        return JSONResponse({"error": "queue backend unavailable"}, status_code=503)
    job_id = request.path_params["job_id"]
    ok, reason = await _cancel_job(job_id)
    if ok:
        return JSONResponse({"ok": True, "job_id": job_id, "status": JOB_STATUS_CANCELLED})
    if reason == "not found":
        return JSONResponse({"error": reason}, status_code=404)
    if reason.startswith("already") or reason == "cannot cancel in-flight job":
        return JSONResponse({"error": reason}, status_code=409)
    return JSONResponse({"error": reason}, status_code=500)


async def api_queue_info(request: Request):
    """GET /api/queue — queue depth, active jobs, recent terminal jobs.

    Operators (and the Discord bot's `/status` slash command) use this
    to show users their position in line. Bounded — `recent` is the
    last 20 terminal jobs, `active` is the workers currently busy
    (1 today, more when WORKER_CONCURRENCY > 1).
    """
    return JSONResponse(await _queue_info())


# ─── VLM frame description ────────────────────────────────────────────────────
# Vision-language fallback for videos without speech (music videos, silent
# gameplay, ASMR, etc.). The bot detects low speech density and calls
# /api/describe to get timestamped frame descriptions, which feed into the
# existing summarize pipeline as if they were a transcript.
#
# Frame extraction lives here (whisper service has ffmpeg). The VLM call
# goes to the same llm-compose proxy the bot uses for text — whisper just
# needs to be on the llm network too (see compose.yaml).

LLM_VISION_API_URL = os.environ.get(
    "LLM_VISION_API_URL",
    os.environ.get("LLM_API_URL", "http://model_proxy:11434/v1"),
)
LLM_VISION_MODEL = os.environ.get("LLM_VISION_MODEL", "Qwen2.5-VL-7B-Instruct")
VLM_FPS_INTERVAL = float(os.environ.get("VLM_FPS_INTERVAL", "10"))  # seconds between frames
VLM_MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "60"))         # cap per video
VLM_FRAME_WIDTH = int(os.environ.get("VLM_FRAME_WIDTH", "512"))      # downscale for inference
VLM_FRAME_TIMEOUT = int(os.environ.get("VLM_FRAME_TIMEOUT", "120"))  # per-frame VLM call timeout
# Bounded parallelism for /api/describe. The VLM proxy serves one model on
# one GPU; too high a value queues at the proxy without speedup, but small
# values (2-4) overlap network/serialization and image encoding with model
# inference. 4 chosen empirically for a single-GPU llm-compose setup.
VLM_FRAME_CONCURRENCY = int(os.environ.get("VLM_FRAME_CONCURRENCY", "4"))

VLM_FRAME_PROMPT = os.environ.get(
    "VLM_FRAME_PROMPT",
    "Describe what is happening in this video frame in 1-2 sentences. "
    "Focus on visible action, setting, and key objects. Be concise and "
    "factual; do not speculate about content not visible.",
)

# ─── Scene-clustering pipeline ───────────────────────────────────────────────
# Frame-by-frame VLM on a static-shot video produces 60 near-identical
# descriptions, which the downstream LLM either loops on or quotes verbatim.
# We pre-cluster frames into scenes BEFORE the LLM sees them so the input
# is structurally compact.
#
# Pipeline:
#   ffmpeg scdet (scene boundaries)
#     → adaptive frame sampling (1-3 per scene depending on length)
#     → VLM per sampled frame (existing)
#     → Jaccard clustering (catches scenes scdet missed)
#     → LLM synthesis per cluster with >1 frame (text-only summary)
#     → return scenes list with time ranges + synthesized descriptions

LLM_TEXT_API_URL = os.environ.get(
    "LLM_TEXT_API_URL",
    os.environ.get("LLM_API_URL", LLM_VISION_API_URL),
)
LLM_SYNTHESIS_MODEL = os.environ.get(
    "LLM_SYNTHESIS_MODEL",
    os.environ.get("LLM_MODEL", LLM_VISION_MODEL),
)
LLM_SYNTHESIS_TIMEOUT = int(os.environ.get("LLM_SYNTHESIS_TIMEOUT", "60"))

# scdet threshold: 0.0-1.0, higher = stricter (only big scene changes detected).
# 0.3 is the ffmpeg default-ish — picks up most cuts, ignores subtle pans.
SCENE_DETECT_THRESHOLD = float(os.environ.get("SCENE_DETECT_THRESHOLD", "0.3"))

# Hardware-accelerated video decode for ffmpeg scene-detect + frame-extract.
# Empty string = software decode (default).
# "cuda" = NVDEC via -hwaccel cuda. Requires `capabilities: [gpu, video]` in
# compose (injects libnvcuvid). Supports VP9/H.264/HEVC/AV1 on Ada/Blackwell.
#
# Why disabled by default: synthetic benchmarks showed software decode often
# matches or beats NVDEC because (a) modern CPU h264/VP9 software decode is
# already fast, (b) the `scene` filter forces every decoded frame back to
# CPU memory via hwdownload, making PCIe transfer the bottleneck instead of
# decode. NVDEC's real advantage is high-bitrate 1080p+ content where decode
# IS the bottleneck — measure on your actual workload before enabling.
#
# Set FFMPEG_HWACCEL=cuda in env after verifying NVDEC actually helps the
# specific codec / resolution / duration you're processing.
FFMPEG_HWACCEL = os.environ.get("FFMPEG_HWACCEL", "").strip()


def _hwaccel_args() -> list[str]:
    """Return `-hwaccel <name>` args, or empty list when disabled.
    Always placed BEFORE `-i` so it applies to the input demuxer."""
    return ["-hwaccel", FFMPEG_HWACCEL] if FFMPEG_HWACCEL else []
# Adaptive sampling: target ~1 frame per N seconds within a scene.
# Short scenes (≤30s) → 1 sample. Medium (30-120s) → 2. Long (>120s) → 3.
SCENE_SAMPLES_SHORT = 1
SCENE_SAMPLES_MEDIUM = 2
SCENE_SAMPLES_LONG = 3
# Cluster similarity threshold (Jaccard on word sets). Catches scenes that
# scdet split mistakenly OR cases where VLM produced near-identical
# descriptions for distinct frames (e.g. static dialogue scene + reverse
# shot, or "Nebula and X" / "Nebula and Y" on a static cosmic backdrop).
# Lower = more aggressive clustering. 0.25 reliably merges paraphrased
# descriptions of the same static scene; raise toward 0.5 if you find
# distinct scenes are getting merged.
CLUSTER_SIMILARITY_THRESHOLD = float(
    os.environ.get("CLUSTER_SIMILARITY_THRESHOLD", "0.25")
)

# Duration-aware scene-count target. A 5-minute static music video and a
# 3-hour podcast can't both target "10 scenes" — the music video deserves
# 2-3, the podcast deserves 30+. Target scales with content length:
#   ~1 scene per N seconds (SCENE_SECONDS_PER_TARGET, default 300 = 5min)
#   bounded by SCENES_MIN (floor) and SCENES_MAX_ABSOLUTE (ceiling).
#
# Used as a SOFT cap: clusters within 1.5x target are left alone (content
# was genuinely varied). Clusters above 1.5x are merged down to target
# (likely paraphrased static-content micro-scenes that survived Jaccard).
SCENE_SECONDS_PER_TARGET = float(
    os.environ.get("SCENE_SECONDS_PER_TARGET", "300")
)
SCENES_MIN = int(os.environ.get("SCENES_MIN", "2"))
SCENES_MAX_ABSOLUTE = int(os.environ.get("SCENES_MAX_ABSOLUTE", "60"))
SCENES_CAP_TOLERANCE = float(os.environ.get("SCENES_CAP_TOLERANCE", "1.5"))


def _target_scene_count(duration: float) -> int:
    """Reasonable target scene count for a video of `duration` seconds.

    Linear scaling at 1 scene per SCENE_SECONDS_PER_TARGET, clamped to
    [SCENES_MIN, SCENES_MAX_ABSOLUTE]. Examples with defaults (300s,
    min=2, ceiling=60):
      30s    → 2  (floor)
      5min   → 2  (floor; one scene per 5min == 1, clamped up)
      15min  → 3
      1hr    → 12
      3hr    → 36
      5hr+   → 60 (ceiling)
    """
    if duration <= 0:
        return SCENES_MIN
    raw = max(1, int(duration / SCENE_SECONDS_PER_TARGET))
    return max(SCENES_MIN, min(raw, SCENES_MAX_ABSOLUTE))

_VLM_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "are", "from", "have",
    "has", "was", "were", "been", "being", "into", "their", "they",
    "which", "what", "when", "where", "while", "would", "could", "should",
    "but", "not", "any", "all", "one", "two", "out", "use", "used",
    # Domain-frequent VLM phrasings
    "video", "frame", "shows", "appears", "image", "scene", "depicts",
    "person", "people", "background", "foreground", "right", "left", "top",
    "bottom", "side", "center", "front", "behind", "around",
})


def _vlm_word_set(text: str) -> set[str]:
    """Word set for Jaccard similarity. Strips stopwords + short tokens."""
    words = re.findall(r"[a-z']{3,}", text.lower())
    return {w for w in words if w not in _VLM_STOPWORDS}


def _vlm_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _detect_scene_boundaries(file_path: str, duration: float) -> list[float]:
    """Use ffmpeg scdet to find scene-change timestamps. Returns a list
    `[0.0, t1, t2, ..., duration]` — boundaries inclusive of start/end.

    Returns `[0.0, duration]` (one big scene) on failure or when no
    scene changes are detected.
    """
    if duration <= 0:
        return [0.0]
    try:
        # scdet emits "lavfi.scene_score=<float>" on stderr; pts of scene
        # changes captured by `showinfo` filter.
        result = subprocess.run(
            ["ffmpeg", *_hwaccel_args(), "-i", file_path,
             "-vf", f"select='gt(scene,{SCENE_DETECT_THRESHOLD})',showinfo",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        log.warning("[VLM] scdet timed out; using single-scene fallback")
        return [0.0, duration]
    # Stderr contains `pts_time:<float>` for each selected frame
    times = [0.0]
    for match in re.finditer(r"pts_time:([\d.]+)", result.stderr):
        try:
            t = float(match.group(1))
        except ValueError:
            continue
        # Filter out near-duplicates (scdet can fire twice on the same cut)
        if not times or t - times[-1] > 0.5:
            times.append(t)
    if times[-1] < duration - 0.5:
        times.append(duration)
    log.info(f"[VLM] scdet found {len(times)-1} scenes "
             f"(threshold={SCENE_DETECT_THRESHOLD})")
    return times


def _adaptive_sample_timestamps(
    boundaries: list[float], max_total: int,
) -> list[tuple[float, int]]:
    """Pick sample timestamps for each scene.

    Returns list of (timestamp, scene_index) pairs. Short scenes get 1
    sample at midpoint; longer scenes get 2-3 evenly spaced.

    Respects `max_total` cap by proportionally downsampling if needed:
    long videos with many scenes get fewer samples per scene.
    """
    if len(boundaries) < 2:
        return []
    samples: list[tuple[float, int]] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        length = end - start
        if length <= 30:
            n = SCENE_SAMPLES_SHORT
        elif length <= 120:
            n = SCENE_SAMPLES_MEDIUM
        else:
            n = SCENE_SAMPLES_LONG
        for j in range(n):
            # j+0.5 puts samples at 1/(2n), 3/(2n), ... — centered, no edges
            t = start + length * (j + 0.5) / n
            samples.append((t, i))
    # Cap at max_total: drop evenly across scenes if over budget
    if len(samples) > max_total:
        step = len(samples) / max_total
        samples = [samples[int(k * step)] for k in range(max_total)]
    return samples


def _extract_frames_at_timestamps(
    file_path: str, out_dir: str, timestamps: list[float], width: int,
) -> list[tuple[str, float]]:
    """Extract one frame per timestamp. Returns [(path, timestamp), ...]
    in order. Uses one ffmpeg call per frame with `-ss` for precise seek.
    Slower than batch fps-filter extraction but lets us target exact
    moments instead of uniform intervals.
    """
    out: list[tuple[str, float]] = []
    for i, ts in enumerate(timestamps):
        path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
        cmd = [
            "ffmpeg", "-y", *_hwaccel_args(),
            "-ss", f"{ts:.3f}", "-i", file_path,
            "-frames:v", "1", "-vf", f"scale={width}:-1",
            "-q:v", "5", "-loglevel", "error", path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(path):
            out.append((path, ts))
        else:
            stderr = result.stderr or ""
            if any(p in stderr for p in _FFMPEG_PERMANENT_PATTERNS):
                raise NoVideoStreamError(
                    f"frame extraction failed at {ts}s: {stderr[:200]}"
                )
            log.warning(f"[VLM] frame extract failed at {ts}s: {stderr[:100]}")
    return out


def _cap_scenes(clusters: list[list[dict]], duration: float) -> list[list[dict]]:
    """Soft-cap cluster count using a duration-scaled target.

    A 5-minute music video gets ~2-3 scenes; a 3-hour podcast gets
    ~30+. The target comes from `_target_scene_count(duration)`.

    Tolerance: clusters within `SCENES_CAP_TOLERANCE` × target are
    left alone (content was genuinely varied — e.g. a trailer with
    rapid cuts deserves its scenes). Above tolerance, greedy adjacent
    merge until at target.

    Preserves temporal order. Singletons / short scenes get absorbed
    first.
    """
    target = _target_scene_count(duration)
    if len(clusters) <= target * SCENES_CAP_TOLERANCE:
        return clusters
    result = [list(c) for c in clusters]
    while len(result) > target:
        # Find adjacent pair to merge: smallest combined size, then
        # smallest combined time span. Singletons get absorbed first;
        # short scenes get absorbed before long ones.
        best_idx = 0
        best_score = (
            len(result[0]) + len(result[1]),
            result[1][-1]["timestamp"] - result[0][0]["timestamp"],
        )
        for i in range(1, len(result) - 1):
            size = len(result[i]) + len(result[i + 1])
            span = result[i + 1][-1]["timestamp"] - result[i][0]["timestamp"]
            score = (size, span)
            if score < best_score:
                best_score = score
                best_idx = i
        result[best_idx].extend(result[best_idx + 1])
        del result[best_idx + 1]
    return result


def _cluster_descriptions(descriptions: list[dict],
                          threshold: float = None) -> list[list[dict]]:
    """Group consecutive descriptions by similarity. Returns list of
    clusters; each cluster is a list of `{timestamp, text}` dicts in
    temporal order. A cluster represents one "scene" semantically.

    Greedy merge: each new description joins the current cluster iff its
    word set has Jaccard ≥ `threshold` with the cluster's representative
    (the most recent frame's words). Otherwise it starts a new cluster.
    """
    if not descriptions:
        return []
    t = threshold if threshold is not None else CLUSTER_SIMILARITY_THRESHOLD
    clusters: list[list[dict]] = []
    current: list[dict] = [descriptions[0]]
    current_words = _vlm_word_set(descriptions[0].get("text") or "")
    for desc in descriptions[1:]:
        words = _vlm_word_set(desc.get("text") or "")
        if _vlm_jaccard(words, current_words) >= t:
            current.append(desc)
            # Merge words for accumulating cluster vocabulary
            current_words = current_words | words
        else:
            clusters.append(current)
            current = [desc]
            current_words = words
    if current:
        clusters.append(current)
    return clusters


def _synthesize_cluster(cluster: list[dict]) -> tuple[str, str]:
    """Combine N frame descriptions from one scene into a single
    1-2 sentence summary via the text LLM, plus aggregate OCR text.

    Returns (synthesized_description, deduped_ocr_text).

    The OCR text from all frames in the cluster is deduped (since the
    same on-screen text often appears across multiple frames of a static
    shot) and returned alongside the description. The LLM synthesis is
    given BOTH the descriptions AND the OCR so it can resolve VLM
    vagueness against on-screen ground truth (e.g. VLM says "title card
    with text" but OCR says "STAR WARS THEME - shitty flute version").

    Falls back to the longest VLM description if the synthesis LLM call
    fails.
    """
    if not cluster:
        return "", ""
    # Aggregate OCR across all frames (dedup'd)
    ocr_seen: set[str] = set()
    ocr_parts: list[str] = []
    for c in cluster:
        ocr = (c.get("ocr") or "").strip()
        if not ocr:
            continue
        # Split on " | " separator we used in _ocr_frame
        for snippet in ocr.split(" | "):
            s = snippet.strip()
            if s and s.lower() not in ocr_seen:
                ocr_seen.add(s.lower())
                ocr_parts.append(s)
    ocr_combined = " | ".join(ocr_parts)

    descriptions = [c.get("text", "").strip() for c in cluster if c.get("text")]
    if len(cluster) == 1:
        return descriptions[0] if descriptions else "", ocr_combined

    # Fallback: longest description (likely richest)
    fallback = max(descriptions, key=len, default="")

    # Skip synthesis when descriptions are extremely similar AND no OCR
    # to integrate — the longest description already covers it.
    if len(set(descriptions)) <= 1 and not ocr_combined:
        return fallback, ocr_combined

    ocr_section = (
        f"\n\nOn-screen text detected via OCR (use these as ground truth — "
        f"prefer OCR text over any text the descriptions guess at):\n"
        f"{ocr_combined}"
    ) if ocr_combined else ""

    prompt = (
        f"Below are {len(descriptions)} short descriptions of consecutive "
        f"frames from the SAME scene in a video. Synthesize them into ONE "
        f"coherent description (1-2 sentences) capturing what's in the "
        f"scene (people, objects, setting) and any notable change or "
        f"action across the frames. Use the OCR text when present to "
        f"anchor specific titles / names / captions. Output ONLY the "
        f"synthesized description — no preamble, no list, no markdown.\n\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
        + ocr_section
    )
    import base64
    import urllib.request
    import urllib.error
    payload = {
        "model": LLM_SYNTHESIS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 250,
    }
    req = urllib.request.Request(
        f"{LLM_TEXT_API_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_SYNTHESIS_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        if text:
            return text, ocr_combined
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning(f"[VLM] synthesis failed (using longest-frame fallback): {e}")
    return fallback, ocr_combined


def _ffprobe_duration(file_path: str) -> float:
    """Get duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return float(result.stdout.strip() or 0)
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


class NoVideoStreamError(Exception):
    """The input file contains no video stream — frame extraction is
    impossible regardless of retries. Mapped to HTTP 422 (permanent) so
    clients short-circuit retry loops."""


# ffmpeg stderr fragments that indicate a permanent extraction failure
# (missing video stream, unsupported codec, corrupt header, etc.).
_FFMPEG_PERMANENT_PATTERNS = (
    "Output file does not contain any stream",
    "does not contain any stream",
    "Stream specifier",
    "no video streams",
    "Invalid data found when processing input",
    "Cannot find a matching stream",
)


def _extract_frames(file_path: str, out_dir: str,
                    fps_interval: float, max_frames: int,
                    width: int) -> list[str]:
    """Extract frames at regular intervals via ffmpeg.

    Returns a sorted list of frame file paths. The interval auto-stretches
    for long videos so we never exceed `max_frames` regardless of duration.

    Raises:
        NoVideoStreamError: input has no video track (e.g. an audio-only
            yt-dlp download). Caller should map to HTTP 422.
        RuntimeError: any other ffmpeg failure (transient — caller may retry).
    """
    duration = _ffprobe_duration(file_path)
    # Auto-stretch interval to fit max_frames over the whole video.
    effective_interval = fps_interval
    if duration > 0:
        effective_interval = max(fps_interval, duration / max_frames)
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", *_hwaccel_args(),
        "-i", file_path,
        "-vf", f"fps=1/{effective_interval},scale={width}:-1",
        "-frames:v", str(max_frames),
        "-q:v", "5",          # JPEG quality 1-31, lower=better; 5 ≈ 85% jpeg
        "-loglevel", "error",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr or ""
        if any(p in stderr for p in _FFMPEG_PERMANENT_PATTERNS):
            raise NoVideoStreamError(
                f"input file has no video stream — likely audio-only "
                f"download. ffmpeg said: {stderr[:200]}"
            )
        raise RuntimeError(f"ffmpeg frame extraction failed: {stderr[:300]}")
    frames = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    return frames


# ─── OCR (on-screen text extraction) ─────────────────────────────────────────
# Catches text the VLM misses: titles, lyrics, credits, captions, brand
# names. The VLM is asked to describe what it sees; it's terrible at
# transcribing text faithfully. EasyOCR is purpose-built for this.
#
# Lazy-loaded — the model weights (~600 MB) only download on first call.
# When VLM_OCR_ENABLED=0, this whole subsystem skips and frame descriptions
# rely solely on the VLM (slightly faster, weaker for text-heavy content).

VLM_OCR_ENABLED = os.environ.get("VLM_OCR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VLM_OCR_LANGUAGES = os.environ.get("VLM_OCR_LANGUAGES", "en").split(",")
# Drop OCR snippets shorter than this — single-character noise / artifacts
# pollute output without adding signal.
VLM_OCR_MIN_CHARS = int(os.environ.get("VLM_OCR_MIN_CHARS", "3"))
# Confidence floor (0-1). EasyOCR returns confidence per detection;
# below this we discard.
VLM_OCR_MIN_CONFIDENCE = float(os.environ.get("VLM_OCR_MIN_CONFIDENCE", "0.5"))

_ocr_reader = None  # singleton; lazy-init on first call
_ocr_reader_lock = threading.Lock()


def _get_ocr_reader():
    """Lazy-init EasyOCR reader. First call downloads model weights into
    the HF cache (~600 MB); subsequent calls reuse the loaded model.
    """
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    with _ocr_reader_lock:
        if _ocr_reader is None:
            log.info("[OCR] initialising EasyOCR (langs=%s, gpu=%s)...",
                     VLM_OCR_LANGUAGES, DEVICE == "cuda")
            try:
                import easyocr
                _ocr_reader = easyocr.Reader(
                    VLM_OCR_LANGUAGES,
                    gpu=(DEVICE == "cuda"),
                    verbose=False,
                )
                log.info("[OCR] ready")
            except Exception as e:
                log.warning(f"[OCR] init failed: {e}; OCR disabled for this run")
                _ocr_reader = False  # sentinel: don't retry
    return _ocr_reader


def _ocr_frame(frame_path: str) -> str:
    """Extract on-screen text from a single frame. Returns the joined
    text (newline-separated) or empty string if OCR is disabled / no
    text detected / OCR fails.

    Filters:
      - Confidence below VLM_OCR_MIN_CONFIDENCE (default 0.5)
      - Text shorter than VLM_OCR_MIN_CHARS (default 3)
    """
    if not VLM_OCR_ENABLED:
        return ""
    reader = _get_ocr_reader()
    if not reader:  # init failed
        return ""
    try:
        # readtext returns list of (bbox, text, confidence)
        results = reader.readtext(frame_path, detail=1, paragraph=False)
    except Exception as e:
        log.warning(f"[OCR] readtext failed for {frame_path}: {e}")
        return ""
    if not results:
        return ""
    snippets = []
    for _bbox, text, conf in results:
        text = (text or "").strip()
        if not text or len(text) < VLM_OCR_MIN_CHARS:
            continue
        if conf < VLM_OCR_MIN_CONFIDENCE:
            continue
        snippets.append(text)
    return " | ".join(snippets)


def _describe_frame(frame_path: str, vlm_model: str, prompt: str) -> str:
    """Send one frame to the VLM and return its description.

    Uses urllib (stdlib) — no extra deps. Synchronous; called from a
    worker thread via run_in_executor.
    """
    import base64
    import urllib.request
    import urllib.error

    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": vlm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        f"{LLM_VISION_API_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=VLM_FRAME_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"VLM HTTP {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"VLM call failed: {e}")
    return data["choices"][0]["message"]["content"].strip()


class VLMNotConfiguredError(Exception):
    """Raised when the LLM proxy is up but the model is missing vision
    capability (e.g. llama-server loaded without mmproj). Surfaces as a
    permanent 422 with an actionable error message — no amount of retries
    will fix it; the operator has to configure the model differently."""


# Patterns in VLM error responses that mean the deployed model can't accept
# images. Per-frame retries can't help — we surface immediately.
_VLM_NOT_CONFIGURED_PATTERNS = (
    "image input is not supported",
    "no mmproj",
    "mmproj is required",
    "vision is not enabled",
    "model does not support vision",
    "model does not support images",
    "multimodal not enabled",
)


def _is_vlm_not_configured(err: str) -> bool:
    err = (err or "").lower()
    return any(p in err for p in _VLM_NOT_CONFIGURED_PATTERNS)


def _describe_video(file_path: str, fps_interval: float, max_frames: int,
                    vlm_model: str, prompt: str) -> dict:
    """Scene-clustered VLM description pipeline.

    Pipeline:
      1. ffmpeg scdet → scene boundaries
      2. Adaptive sampling: 1-3 frames per scene depending on length
      3. VLM in parallel (existing concurrency machinery)
      4. Jaccard cluster (catches paraphrased near-duplicates that
         scdet split or that VLM described inconsistently)
      5. LLM synthesis per cluster with >1 frame

    Returns:
    {
        "duration": float,
        "frame_count": int,             # frames actually VLM'd
        "successful_frames": int,
        "model": str,
        "scenes": [                     # NEW — primary output for callers
            {
                "start": float,
                "end": float,
                "frame_count": int,
                "description": str,     # synthesized when frames>1
            },
            ...
        ],
        "descriptions": [               # raw per-frame, backward-compat
            {"timestamp": float, "text": str}, ...
        ],
    }

    Backward compat: `descriptions` still present so older bot versions
    keep working. `scenes` is the recommended consumer.

    Raises:
        VLMNotConfiguredError: VLM model rejects images (no mmproj loaded).
        NoVideoStreamError: input file has no video track.
    """
    import concurrent.futures
    import threading

    duration = _ffprobe_duration(file_path)
    frames_dir = tempfile.mkdtemp(prefix="vlm-frames-")
    try:
        # 1. Scene boundaries via ffmpeg scdet
        boundaries = _detect_scene_boundaries(file_path, duration)
        # 2. Adaptive sampling
        samples = _adaptive_sample_timestamps(boundaries, max_frames)
        if not samples:
            # Fallback: uniform sampling for cases where scdet found nothing
            # (rare, but defensive).
            log.info("[VLM] scdet found no scenes; falling back to uniform sampling")
            effective_interval = max(fps_interval, duration / max_frames) \
                if duration > 0 else fps_interval
            n = min(max_frames, max(1, int(duration / effective_interval)))
            samples = [(i * effective_interval, 0) for i in range(n)]

        sample_ts = [t for t, _ in samples]
        scene_of_sample = [s for _, s in samples]

        frame_files = _extract_frames_at_timestamps(
            file_path, frames_dir, sample_ts, VLM_FRAME_WIDTH,
        )
        if not frame_files:
            raise RuntimeError("ffmpeg produced no frames")
        log.info(f"[VLM] extracted {len(frame_files)} frames "
                 f"across {len(boundaries)-1} scenes")

        # 3. VLM in parallel
        workers = max(1, VLM_FRAME_CONCURRENCY)
        results: list[dict | None] = [None] * len(frame_files)
        success_count = 0
        first_failure: str | None = None
        not_configured_err: str | None = None
        cancel = threading.Event()
        lock = threading.Lock()

        def _worker(i: int, fp: str, timestamp: float) -> None:
            nonlocal success_count, first_failure, not_configured_err
            if cancel.is_set():
                results[i] = {"timestamp": timestamp,
                              "text": "[frame description unavailable]",
                              "ocr": ""}
                return
            try:
                text = _describe_frame(fp, vlm_model, prompt)
                with lock:
                    success_count += 1
            except Exception as e:
                err = str(e)
                if _is_vlm_not_configured(err):
                    with lock:
                        if not_configured_err is None:
                            not_configured_err = err
                    cancel.set()
                    text = "[frame description unavailable]"
                else:
                    log.warning(
                        f"[VLM] Frame {i} ({timestamp:.0f}s) failed: {err}"
                    )
                    with lock:
                        if first_failure is None:
                            first_failure = err
                    text = "[frame description unavailable]"
            # Run OCR in parallel with VLM where possible. EasyOCR isn't
            # thread-safe across distinct readers but the singleton reader
            # serialises internally; calling per-frame from multiple workers
            # is safe-but-slow. Acceptable for our small batch sizes.
            ocr = _ocr_frame(fp)
            results[i] = {"timestamp": timestamp, "text": text, "ocr": ocr}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_worker, i, fp, ts)
                for i, (fp, ts) in enumerate(frame_files)
            ]
            concurrent.futures.wait(futures)

        if not_configured_err:
            raise VLMNotConfiguredError(
                f"VLM model '{vlm_model}' on the LLM proxy rejected the "
                f"image input. Likely cause: the model is loaded "
                f"WITHOUT an mmproj (multimodal projector) file. "
                f"Fix: download the corresponding mmproj GGUF (same "
                f"HF repo as the main model) and pass it to "
                f"llama-server with --mmproj. Underlying error: "
                f"{not_configured_err[:200]}"
            )

        descriptions = [r for r in results if r is not None]
        if success_count == 0 and frame_files:
            raise RuntimeError(
                f"All {len(frame_files)} frame descriptions failed. "
                f"First error: {first_failure}"
            )
        log.info(f"[VLM] Described {success_count}/{len(frame_files)} frames")

        # 4. Cluster by description similarity (catches paraphrased near-dups
        # that scdet would have split + cases where VLM described otherwise-
        # identical frames differently). Drops frames that failed VLM.
        clean = [d for d in descriptions
                 if d.get("text") and d["text"] != "[frame description unavailable]"]
        clusters = _cluster_descriptions(clean)
        pre_cap = len(clusters)
        clusters = _cap_scenes(clusters, duration)
        if len(clusters) != pre_cap:
            log.info(f"[VLM] clustered {len(clean)} frames → {pre_cap} scenes "
                     f"→ capped to {len(clusters)} "
                     f"(target {_target_scene_count(duration)} "
                     f"@ duration {duration:.0f}s)")
        else:
            log.info(f"[VLM] clustered {len(clean)} frames → {len(clusters)} scenes")

        # 5. Synthesize each cluster (parallel; cheap text-only LLM calls).
        # _synthesize_cluster returns (description, ocr_text) — OCR is the
        # cluster-aggregated, deduped on-screen text from all frames in
        # the cluster, used by the bot to anchor specific names/titles
        # the VLM couldn't read.
        scene_outputs: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_synthesize_cluster, c) for c in clusters]
            syntheses = [f.result() for f in futures]
        for cluster, (synth, ocr) in zip(clusters, syntheses):
            scene_outputs.append({
                "start": cluster[0]["timestamp"],
                "end": cluster[-1]["timestamp"],
                "frame_count": len(cluster),
                "description": synth,
                "ocr": ocr,
            })

        return {
            "duration": duration,
            "frame_count": len(descriptions),
            "successful_frames": success_count,
            "model": vlm_model,
            "scenes": scene_outputs,
            "descriptions": descriptions,  # backward-compat
        }
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


async def api_describe(request: Request):
    """POST /api/describe — generate VLM frame descriptions for a video.

    Body fields:
      file_path:     str    (required) path on the whisper container
      fps_interval:  float  (default VLM_FPS_INTERVAL — seconds between frames)
      max_frames:    int    (default VLM_MAX_FRAMES — caps total frames,
                             interval auto-stretches for long videos)
      model:         str    (default LLM_VISION_MODEL)
      prompt:        str    (default VLM_FRAME_PROMPT)
      cleanup:       bool   (default false) remove file_path after.

    Returns: {"duration": ..., "descriptions": [{"timestamp": s, "text": "..."}, ...],
              "frame_count": N, "interval_seconds": s, "model": "..."}
    """
    body = await request.json()
    file_path = body.get("file_path", "").strip()
    if not file_path or not os.path.isfile(file_path):
        return JSONResponse({"error": f"file not found: {file_path}"}, status_code=400)

    fps_interval = float(body.get("fps_interval", VLM_FPS_INTERVAL))
    max_frames = int(body.get("max_frames", VLM_MAX_FRAMES))
    vlm_model = body.get("model", LLM_VISION_MODEL)
    prompt = body.get("prompt", VLM_FRAME_PROMPT)

    log.info(f"[API] /describe: {file_path} fps_interval={fps_interval} "
             f"max_frames={max_frames} model={vlm_model}")

    try:
        import asyncio
        result = await asyncio.to_thread(
            _describe_video, file_path, fps_interval, max_frames,
            vlm_model, prompt,
        )
        return JSONResponse(result)
    except NoVideoStreamError as e:
        # Permanent: no video stream means the bot downloaded audio-only.
        # Surface as 422 + permanent:true so the bot's retry loop short-
        # circuits. The bot fix is to pass keep_video=true on yt-download.
        log.error(f"[API] describe permanent error: {e}")
        return JSONResponse(
            {"error": str(e), "permanent": True},
            status_code=422,
        )
    except VLMNotConfiguredError as e:
        # Permanent: the LLM proxy is up but the model can't accept images.
        # Re-trying produces the same error every time.
        log.error(f"[API] describe permanent error (VLM config): {e}")
        return JSONResponse(
            {"error": str(e), "permanent": True, "kind": "vlm_not_configured"},
            status_code=422,
        )
    except Exception as e:
        log.error(f"[API] describe error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if body.get("cleanup") and os.path.isfile(file_path):
            try:
                parent = os.path.dirname(file_path)
                os.remove(file_path)
                if parent.startswith("/tmp/yt-dlp-") and not os.listdir(parent):
                    os.rmdir(parent)
                log.info(f"[API] Cleaned up after describe: {file_path}")
            except Exception as e:
                log.warning(f"[API] Cleanup failed: {e}")


# ─── Single-image OCR + VLM endpoint ─────────────────────────────────────────
# Used by the Discord bot's image-attachment summary flow. Mirrors the
# pipeline that /api/describe runs per-frame, but for one user-uploaded
# still image. Unlike /api/describe (which takes a path on a shared volume),
# this endpoint accepts the image as multipart bytes so callers in other
# containers don't need a shared mount.

# Cap per-request upload size. Discord allows attachments up to 25MB
# (50MB for Nitro boosters); 32MB is a comfortable ceiling that still
# catches abuse / accidental video uploads.
IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", str(32 * 1024 * 1024)))

# Per-image VLM prompt. Distinct from VLM_FRAME_PROMPT (which is tuned for
# 1-of-60 video frames where the model should describe action) — for a
# single still we want a more comprehensive, self-contained description.
IMAGE_VLM_PROMPT = os.environ.get(
    "IMAGE_VLM_PROMPT",
    "Describe this image in 2-4 sentences. Include the subject, setting, "
    "any people or objects, the overall mood, and notable visual details. "
    "If there is significant text visible (signs, captions, document "
    "contents, UI), note that text is present but don't try to transcribe "
    "it verbatim — OCR handles that separately. Be concrete, no hedging.",
)


_ALLOWED_IMAGE_CONTENT_TYPES = (
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/gif", "image/bmp", "image/tiff",
)


async def api_image(request: Request):
    """POST /api/image — OCR + VLM describe a single uploaded image.

    Accepts multipart/form-data with field `file` (the image bytes).
    Optional form fields:
      model:  VLM model id (default LLM_VISION_MODEL)
      prompt: VLM prompt override (default IMAGE_VLM_PROMPT)
      ocr:    "0" to skip OCR pass (default "1")
      vlm:    "0" to skip VLM pass (default "1")

    Returns:
      {
        "ocr": "...",           # joined OCR snippets, " | "-separated
        "description": "...",   # VLM scene description
        "width": int, "height": int,
        "model": str,            # VLM model id used
        "bytes": int,            # original upload size
      }

    422 + permanent:true on VLM-not-configured (mmproj missing on the
    deployed model). 400 on missing file / oversized upload / bad mime.
    """
    # Starlette parses multipart lazily; the form() call buffers the body.
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse(
            {"error": f"multipart parse failed: {e}"}, status_code=400,
        )

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse(
            {"error": "missing 'file' field (multipart upload required)"},
            status_code=400,
        )

    content_type = (getattr(upload, "content_type", "") or "").lower()
    # Be lenient — Discord CDN sometimes serves images with generic
    # application/octet-stream. Accept anything that LOOKS like an image
    # extension OR carries an image/* content type.
    filename = (getattr(upload, "filename", "") or "upload").lower()
    looks_like_image = (
        content_type.startswith("image/")
        or filename.endswith((".jpg", ".jpeg", ".png", ".webp",
                              ".gif", ".bmp", ".tif", ".tiff"))
    )
    if not looks_like_image:
        return JSONResponse(
            {"error": f"unsupported content-type '{content_type}' and "
                      f"non-image filename '{filename}'"},
            status_code=400,
        )

    data = await upload.read()
    if not data:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    if len(data) > IMAGE_MAX_BYTES:
        return JSONResponse(
            {"error": f"image too large: {len(data)} bytes > "
                      f"IMAGE_MAX_BYTES ({IMAGE_MAX_BYTES})"},
            status_code=400,
        )

    # Persist to a temp file so _ocr_frame / _describe_frame (which both
    # take a path) can read it. Suffix follows the original filename so
    # ffmpeg / PIL / EasyOCR / VLM all see a sensible extension.
    suffix = os.path.splitext(filename)[1] or ".jpg"
    if suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp",
                              ".gif", ".bmp", ".tif", ".tiff"):
        suffix = ".jpg"
    tmpdir = tempfile.mkdtemp(prefix="image-upload-")
    tmp_path = os.path.join(tmpdir, f"img{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)

        do_ocr = form.get("ocr", "1") not in ("0", "false", "no", "off")
        do_vlm = form.get("vlm", "1") not in ("0", "false", "no", "off")
        vlm_model = (form.get("model") or LLM_VISION_MODEL).strip()
        prompt = (form.get("prompt") or IMAGE_VLM_PROMPT).strip()

        log.info(
            "[API] /image: filename=%s bytes=%d ct=%s ocr=%s vlm=%s model=%s",
            filename, len(data), content_type, do_ocr, do_vlm, vlm_model,
        )

        # Normalise to JPEG via PIL before the VLM call. llama.cpp's
        # multimodal image loader is strict about format/MIME matching
        # the data URL prefix _describe_frame() hard-codes as image/jpeg,
        # so a PNG / WebP / GIF / AVIF straight from Discord CDN gets
        # rejected with "Failed to load image or audio file" (HTTP 400)
        # even though the bytes are a valid image. Re-encode to baseline
        # JPEG (RGB, no alpha) and the VLM accepts every format PIL can
        # decode. PIL is a transitive dep via whisperx so no new install.
        #
        # OCR still runs against the original — EasyOCR is happy with
        # any format and PNG/lossless input may even be slightly more
        # legible than a JPEG round-trip.
        width = height = 0
        vlm_path = tmp_path  # default: feed the original if normalisation fails
        try:
            from PIL import Image as _PILImage
            from PIL import ImageFile as _PILImageFile
            # Tolerate slightly truncated images — PIL otherwise refuses
            # to .load() anything missing its EOF marker, even when 99%
            # of the IDAT data is intact. Discord's CDN occasionally
            # serves images with the last few bytes missing (observed
            # 2026-05-26 on real Discord attachments); the VLM rejects
            # the original PNG outright but happily accepts whatever
            # PIL re-encodes after a best-effort decode.
            _PILImageFile.LOAD_TRUNCATED_IMAGES = True
            with _PILImage.open(tmp_path) as im:
                # Force .load() under LOAD_TRUNCATED_IMAGES so partial
                # data gets converted to whatever black/transparent fill
                # PIL uses for missing pixels — better than the VLM 400.
                im.load()
                width, height = im.size
                # Flatten transparency against white — the VLM sees a
                # natural background instead of black where alpha used
                # to be (matches how browsers/Discord display the image).
                if im.mode in ("RGBA", "LA", "P"):
                    bg = _PILImage.new("RGB", im.size, (255, 255, 255))
                    rgba = im.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                    rgb = bg
                elif im.mode != "RGB":
                    rgb = im.convert("RGB")
                else:
                    rgb = im
                # Animated GIFs / multi-frame TIFFs — PIL gives the first
                # frame by default, which is what we want for the VLM
                # describe pass (single still). The OCR pass on the
                # original handles each frame internally via EasyOCR.
                jpeg_path = os.path.join(tmpdir, "img-vlm.jpg")
                rgb.save(jpeg_path, "JPEG", quality=90, optimize=True)
                vlm_path = jpeg_path
                log.info(
                    "[API] /image: normalised to JPEG for VLM "
                    "(orig=%dx%d %s → jpg %d bytes)",
                    width, height, im.format, os.path.getsize(jpeg_path),
                )
        except Exception as e:
            log.warning(
                "[API] /image: PIL normalise failed (%s) — feeding original "
                "to VLM; expect VLM HTTP 400 if format isn't JPEG.", e,
            )

        ocr_text = ""
        description = ""

        # Run OCR + VLM in parallel — both are blocking sync calls so
        # gather them in worker threads. OCR is fast (~1-2s); VLM is
        # the long pole (5-30s depending on model + image complexity).
        async def _run_ocr() -> str:
            if not do_ocr:
                return ""
            try:
                return await asyncio.to_thread(_ocr_frame, tmp_path)
            except Exception as e:
                log.warning("[API] /image: OCR failed: %s", e)
                return ""

        async def _run_vlm() -> str:
            if not do_vlm:
                return ""
            try:
                return await asyncio.to_thread(
                    _describe_frame, vlm_path, vlm_model, prompt,
                )
            except RuntimeError as e:
                # Surface VLM-not-configured as a permanent 422 — matches
                # /api/describe semantics so the bot doesn't waste retries.
                if _is_vlm_not_configured(str(e)):
                    raise VLMNotConfiguredError(str(e)) from e
                raise

        try:
            ocr_text, description = await asyncio.gather(_run_ocr(), _run_vlm())
        except VLMNotConfiguredError as e:
            log.error("[API] /image permanent error (VLM config): %s", e)
            return JSONResponse(
                {"error": str(e), "permanent": True,
                 "kind": "vlm_not_configured"},
                status_code=422,
            )
        except Exception as e:
            log.error("[API] /image VLM error: %s", e)
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse({
            "ocr": ocr_text or "",
            "description": description or "",
            "width": width,
            "height": height,
            "model": vlm_model,
            "bytes": len(data),
        })
    finally:
        # Best-effort cleanup; ignore failures (next /tmp sweep handles it).
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                os.rmdir(tmpdir)
        except Exception as e:
            log.warning("[API] /image cleanup failed: %s", e)


async def api_cleanup(request: Request):
    """POST /api/cleanup — best-effort delete a yt-dlp temp file.

    Body: {"file_path": "/tmp/yt-dlp-xxx/yyy.wav"}

    Used by clients that pass cleanup=false to /api/transcribe (e.g. when
    they may follow up with /api/describe) and need to reclaim the file
    afterwards.

    Safety: only deletes paths under /tmp/yt-dlp-* to prevent abuse.
    """
    body = await request.json()
    file_path = body.get("file_path", "").strip()
    if not file_path:
        return JSONResponse({"error": "file_path required"}, status_code=400)
    # Restrict to known-safe prefix
    if not file_path.startswith("/tmp/yt-dlp-"):
        return JSONResponse(
            {"error": "only /tmp/yt-dlp-* paths can be cleaned"},
            status_code=400,
        )
    if not os.path.isfile(file_path):
        # idempotent — already gone is fine
        return JSONResponse({"ok": True, "already_gone": True})
    try:
        parent = os.path.dirname(file_path)
        os.remove(file_path)
        if parent.startswith("/tmp/yt-dlp-") and not os.listdir(parent):
            os.rmdir(parent)
        log.info(f"[API] /cleanup removed {file_path}")
        return JSONResponse({"ok": True})
    except Exception as e:
        log.warning(f"[API] /cleanup failed for {file_path}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


API_ROUTES = [
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/yt-download", api_yt_download, methods=["POST"]),
    Route("/api/transcribe", api_transcribe, methods=["POST"]),
    Route("/api/describe", api_describe, methods=["POST"]),
    Route("/api/image", api_image, methods=["POST"]),
    Route("/api/cleanup", api_cleanup, methods=["POST"]),
    # Server-side job queue (Valkey-backed). New default — bot/MCP use these.
    Route("/api/jobs", api_jobs_submit, methods=["POST"]),
    Route("/api/jobs/{job_id}", api_jobs_get, methods=["GET"]),
    Route("/api/jobs/{job_id}", api_jobs_cancel, methods=["DELETE"]),
    Route("/api/queue", api_queue_info, methods=["GET"]),
]


# -- Launch --------------------------------------------------------------------
log.info("Launching Gradio on 0.0.0.0:7860...")
try:
    # Serve with custom Starlette app: API routes + Gradio mounted at /
    import uvicorn
    from starlette.applications import Starlette
    from contextlib import asynccontextmanager

    # Lifespan: bring up the job queue worker before accepting traffic,
    # tear it down cleanly on SIGTERM. If valkey is unreachable, the worker
    # tasks aren't started and /api/jobs returns 503 — /api/transcribe with
    # wait=true still works via the legacy lock path.
    #
    # `on_startup`/`on_shutdown` kwargs were removed in modern Starlette
    # (≥0.35). Use the lifespan context-manager pattern instead.
    @asynccontextmanager
    async def _lifespan(app):
        await _worker_startup()
        try:
            yield
        finally:
            await _worker_shutdown()

    app = Starlette(routes=API_ROUTES, lifespan=_lifespan)
    app = gr.mount_gradio_app(app, demo, path="/", theme=THEME, css=CSS, js="""
() => {
    const observer = new MutationObserver(() => {
        const el = document.querySelector('#transcript-html');
        if (el) el.scrollTop = el.scrollHeight;
    });
    const init = () => {
        const el = document.querySelector('#transcript-html');
        if (el) {
            observer.observe(el, { childList: true, subtree: true, characterData: true });
        } else {
            setTimeout(init, 500);
        }
    };
    init();
}
""")
    log.info(f"Mounted {len(API_ROUTES)} API routes: {[r.path for r in API_ROUTES]}")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
except Exception as e:
    log.error(f"Failed to launch: {e}")
    traceback.print_exc()
    sys.exit(1)
