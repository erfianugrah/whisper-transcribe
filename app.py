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

# Idle unload: free VRAM after MODEL_IDLE_TIMEOUT seconds of no transcription
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))  # 5 min default
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
    """Reset the idle timer. Called at start AND end of transcription."""
    global _last_activity, _idle_timer
    _last_activity = time.time()
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
                      return_file=True):
    """Inner generator — runs under _transcription_lock.
    Yields 4-tuples: (status, html_view, plain_text, subtitle_file).

    `return_file=False` skips subtitle file generation (callers that only need
    the transcript text — e.g. the bot via /api/transcribe — avoid disk I/O
    and the leak window if the response is dropped).
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
    log.info(f"[{request_id}] Running word-level alignment for '{detected_lang}'...")
    t0_align = time.time()
    alignment_ok = False
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
    align_status = f"Aligned in {align_elapsed:.1f}s" if alignment_ok else "Alignment failed (segment-level timestamps only)"
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

    def _refresh_media():
        # Manual refresh bypasses the TTL cache.
        return gr.update(choices=scan_media_files(force=True))

    refresh_media_btn.click(fn=_refresh_media, outputs=[local_path_input])

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

# -- HTTP API (for MCP server / programmatic access) --------------------------
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def api_status(request: Request):
    """GET /api/status — GPU info, ready state, capabilities."""
    return JSONResponse({
        "status": "ready",
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
                       return_file=True):
    """Run transcription synchronously, return final result dict."""
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
    ):
        last_status, last_html, last_text, last_file = status, html, text, subtitle_file
    return {
        "status": last_status,
        "transcript": last_text,
        "subtitle_file": last_file,
    }


async def api_transcribe(request: Request):
    """POST /api/transcribe — transcribe a local file.

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

    Returns: {"status": "...", "transcript": "...", "subtitle_file": "..."}
    `subtitle_file` is null when return_file=false.
    """
    body = await request.json()
    file_path = body.get("file_path", "").strip()
    if not file_path or not os.path.isfile(file_path):
        return JSONResponse({"error": f"file not found: {file_path}"}, status_code=400)

    if not _transcription_lock.acquire(blocking=False):
        return JSONResponse({"error": "busy — another transcription is running"}, status_code=409)

    return_file = bool(body.get("return_file", True))
    result_subtitle: str | None = None  # captured for error-path cleanup

    try:
        import asyncio
        result = await asyncio.to_thread(
            _run_transcription,
            file_path,
            model_name=body.get("model", "turbo"),
            language=body.get("language", "Auto-detect"),
            output_format=body.get("format", "txt"),
            enable_diarization=body.get("diarize", False),
            min_speakers=body.get("min_speakers", 0),
            max_speakers=body.get("max_speakers", 0),
            batch_size=body.get("batch_size"),
            hotwords=body.get("hotwords", ""),
            initial_prompt=body.get("initial_prompt", ""),
            suppress_numerals=body.get("suppress_numerals", False),
            return_file=return_file,
        )
        result_subtitle = result.get("subtitle_file")
        return JSONResponse(result)
    except Exception as e:
        log.error(f"[API] Transcription error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        _transcription_lock.release()
        _reset_idle_timer()  # reset countdown after API job ends

        # Reclaim the subtitle file when caller said it didn't want one
        # (we may still have generated one if return_file flipped between
        # run start and cleanup, or to handle older clients explicitly).
        if not return_file and result_subtitle and os.path.isfile(result_subtitle):
            try:
                os.remove(result_subtitle)
                log.info(f"[API] Cleaned subtitle file (return_file=false): {result_subtitle}")
            except OSError as e:
                log.warning(f"[API] Subtitle cleanup failed: {e}")

        # Cleanup temp source file if requested
        if body.get("cleanup") and os.path.isfile(file_path):
            try:
                parent = os.path.dirname(file_path)
                os.remove(file_path)
                # Remove parent if it's a yt-dlp temp dir and now empty
                if parent.startswith("/tmp/yt-dlp-") and not os.listdir(parent):
                    os.rmdir(parent)
                log.info(f"[API] Cleaned up temp file: {file_path}")
            except Exception as e:
                log.warning(f"[API] Cleanup failed for {file_path}: {e}")


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
            ["ffmpeg", "-i", file_path,
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
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", file_path,
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


def _synthesize_cluster(cluster: list[dict]) -> str:
    """Combine N frame descriptions from one scene into a single
    1-2 sentence summary via the text LLM. Falls back to the longest
    frame's description if the LLM call fails.
    """
    if not cluster:
        return ""
    if len(cluster) == 1:
        return (cluster[0].get("text") or "").strip()

    descriptions = [c.get("text", "").strip() for c in cluster if c.get("text")]
    # Fallback: longest description (likely richest)
    fallback = max(descriptions, key=len, default="")

    # Skip synthesis when descriptions are extremely similar (avoid cost)
    if len(set(descriptions)) <= 1:
        return fallback

    prompt = (
        f"Below are {len(descriptions)} short descriptions of consecutive "
        f"frames from the SAME scene in a video. Synthesize them into ONE "
        f"coherent description (1-2 sentences) capturing what's in the "
        f"scene (people, objects, setting) and any notable change or "
        f"action across the frames. Output ONLY the synthesized "
        f"description — no preamble, no list, no markdown.\n\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    )
    import base64
    import urllib.request
    import urllib.error
    payload = {
        "model": LLM_SYNTHESIS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
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
            return text
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning(f"[VLM] synthesis failed (using longest-frame fallback): {e}")
    return fallback


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
        "ffmpeg", "-y",
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
                              "text": "[frame description unavailable]"}
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
            results[i] = {"timestamp": timestamp, "text": text}

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
        log.info(f"[VLM] clustered {len(clean)} frames → {len(clusters)} scenes")

        # 5. Synthesize each cluster (parallel; cheap text-only LLM calls)
        scene_outputs: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_synthesize_cluster, c) for c in clusters]
            syntheses = [f.result() for f in futures]
        for cluster, synth in zip(clusters, syntheses):
            scene_outputs.append({
                "start": cluster[0]["timestamp"],
                "end": cluster[-1]["timestamp"],
                "frame_count": len(cluster),
                "description": synth,
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
    Route("/api/cleanup", api_cleanup, methods=["POST"]),
]


# -- Launch --------------------------------------------------------------------
log.info("Launching Gradio on 0.0.0.0:7860...")
try:
    # Serve with custom Starlette app: API routes + Gradio mounted at /
    import uvicorn
    from starlette.applications import Starlette

    app = Starlette(routes=API_ROUTES)
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
