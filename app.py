import sys
import os
import time
import logging
import tempfile
import traceback
import shutil
import json

# -- Logging setup -------------------------------------------------------------
DEBUG_MODE = os.environ.get("DEBUG_MODE", "1") == "1"

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
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_name = parts[0]
            vram_mb = int(parts[1])
            GPU_INFO_STR = f"{gpu_name}  |  {vram_mb // 1024} GB VRAM  |  whisperX (faster-whisper + wav2vec2 alignment)  |  float16"
            log.info(f"GPU 0: {gpu_name} ({vram_mb} MB VRAM)")
        else:
            GPU_INFO_STR = f"CUDA GPU detected  |  whisperX  |  float16"
    except Exception:
        GPU_INFO_STR = f"CUDA GPU detected  |  whisperX  |  float16"
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
align_model_cache = {}  # keyed by language code


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
    """Load (or reuse) a wav2vec2 alignment model for a given language."""
    if language_code in align_model_cache:
        return align_model_cache[language_code]
    log.info(f"Loading alignment model for '{language_code}'...")
    t0 = time.time()
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=DEVICE,
    )
    align_model_cache[language_code] = (model_a, metadata)
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


def cleanup_upload(file_path):
    """Remove the uploaded file and its parent gradio temp dir if empty."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
        parent = os.path.dirname(file_path)
        if parent and parent.startswith("/tmp/gradio/") and not os.listdir(parent):
            os.rmdir(parent)
        log.info(f"Cleaned up upload: {file_path}")
    except Exception as e:
        log.warning(f"Failed to clean up {file_path}: {e}")


def cleanup_stale_gradio_tmp():
    """Remove old /tmp/gradio/ directories on startup."""
    gradio_tmp = "/tmp/gradio"
    if not os.path.isdir(gradio_tmp):
        return
    count = 0
    for entry in os.listdir(gradio_tmp):
        path = os.path.join(gradio_tmp, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            count += 1
        except Exception as e:
            log.warning(f"Failed to clean {path}: {e}")
    if count:
        log.info(f"Cleaned {count} stale gradio temp dirs")


cleanup_stale_gradio_tmp()


# -- Default batch size based on VRAM -----------------------------------------
DEFAULT_BATCH_SIZE = 4  # conservative CPU default
if DEVICE == "cuda":
    try:
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
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


# -- Post-processing: split segments at speaker boundaries --------------------
def split_segments_by_speaker(segments):
    """Split segments where the speaker changes mid-segment (using word-level labels).

    Also splits overly long single-speaker segments at sentence boundaries.
    """
    MAX_SEGMENT_WORDS = 40  # split segments longer than this at sentence ends

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


def _words_to_segment(words, speaker):
    """Build a segment dict from a list of word dicts."""
    text = " ".join(w.get("word", "").strip() for w in words if w.get("word", "").strip())
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
_last_result = {"segments": [], "has_speakers": False, "format": "srt"}

# -- Concurrency lock ----------------------------------------------------------
import threading
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


def _transcribe_inner(file, local_path, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords, initial_prompt, suppress_numerals, request_id):
    """Inner generator — runs under _transcription_lock.
    Yields 4-tuples: (status, html_view, plain_text, subtitle_file)
    """
    global _previous_subtitle

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
            print_progress=True,
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
            print_progress=True,
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

    if output_format == "txt":
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
MEDIA_ROOT = "/media"
MEDIA_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
                    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".ts", ".flv"}


def scan_media_files() -> list[str]:
    """Walk MEDIA_ROOT and return paths to audio/video files, sorted by mtime (newest first)."""
    if not os.path.isdir(MEDIA_ROOT):
        return []
    found = []
    for root, _dirs, files in os.walk(MEDIA_ROOT):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in MEDIA_EXTENSIONS:
                found.append(os.path.join(root, fname))
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found


# -- Custom CSS ----------------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
/* Header */
.header-wrap {
    text-align: center;
    padding: 0.5rem 0 0.25rem;
}
.header-wrap h2 {
    margin: 0 0 2px;
    font-weight: 700;
    font-size: 1.4rem;
}
.header-wrap .sub {
    opacity: 0.4;
    font-size: 0.75rem;
}
.header-wrap .gpu {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.72rem;
    opacity: 0.5;
    margin-top: 3px;
}
/* Hide Gradio footer */
footer { display: none !important; }
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
/* Transcript HTML viewer */
#transcript-html {
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    background: var(--background-fill-secondary);
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
            <div class="sub">faster-whisper + wav2vec2 alignment + pyannote diarization</div>
            <div class="gpu">{GPU_INFO_STR}</div>
        </div>
    """)

    # -- Input: File source --
    with gr.Group():
        file_input = gr.File(
            label="Upload audio/video (transcription starts automatically)",
            file_types=["audio", "video"],
            height=120,
        )
        with gr.Row():
            local_path_input = gr.Dropdown(
                choices=scan_media_files(),
                value=None,
                label="Or select from /media",
                allow_custom_value=True,
                scale=9,
            )
            refresh_media_btn = gr.Button("↻", scale=1, size="sm", variant="secondary")

    def _refresh_media():
        return gr.update(choices=scan_media_files())

    refresh_media_btn.click(fn=_refresh_media, outputs=[local_path_input])

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
    with gr.Accordion("Speaker diarization", open=DIARIZATION_AVAILABLE):
        if not DIARIZATION_AVAILABLE:
            gr.Markdown(
                "<small style='opacity:0.6'>Disabled — set <code>HF_TOKEN</code> env var to enable.</small>"
            )
        with gr.Row():
            diarize_checkbox = gr.Checkbox(
                label="Enable diarization",
                value=False,
                interactive=DIARIZATION_AVAILABLE,
                scale=2,
            )
            min_speakers_input = gr.Number(
                value=0,
                label="Min speakers (0 = auto)",
                minimum=0,
                maximum=20,
                precision=0,
                interactive=DIARIZATION_AVAILABLE,
                scale=1,
            )
            max_speakers_input = gr.Number(
                value=0,
                label="Max speakers (0 = auto)",
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
            json_data = {"segments": []}
            for seg in segments:
                seg_out = {"start": round(seg.get("start", 0), 3), "end": round(seg.get("end", 0), 3), "text": seg.get("text", "").strip()}
                if has_speakers:
                    seg_out["speaker"] = seg.get("speaker", "?")
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
    # Auto-transcribe on upload
    upload_event = file_input.upload(
        fn=transcribe,
        inputs=all_inputs,
        outputs=all_outputs,
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
    transcribe_event = transcribe_btn.click(
        fn=transcribe,
        inputs=all_inputs,
        outputs=all_outputs,
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

    # Cancel aborts running transcription events
    cancel_btn.click(fn=None, cancels=[upload_event, transcribe_event])

# -- Launch --------------------------------------------------------------------
log.info("Launching Gradio on 0.0.0.0:7860...")
try:
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CSS, js="""
() => {
    // Auto-scroll transcript HTML to bottom as content streams in
    const observer = new MutationObserver((mutations) => {
        const el = document.querySelector('#transcript-html');
        if (el) el.scrollTop = el.scrollHeight;
    });
    // Observe once the element exists
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
except Exception as e:
    log.error(f"Failed to launch: {e}")
    traceback.print_exc()
    sys.exit(1)
