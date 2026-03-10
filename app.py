import sys
import os
import time
import logging
import tempfile
import traceback
import glob
import shutil

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
current_model_key = None  # (model_name, hotwords) tuple for cache invalidation
diarize_model = None
align_model_cache = {}  # keyed by language code


def load_whisper(model_name, hotwords=None):
    """Load (or reuse) a whisperX transcription model."""
    global whisper_model, current_model_key
    # Map friendly names to actual model IDs
    actual_name = model_name
    if model_name == "large":
        actual_name = "large-v3"
    elif model_name == "turbo":
        actual_name = "large-v3-turbo"

    cache_key = (actual_name, hotwords or "")
    if cache_key != current_model_key:
        log.info(f"Loading whisperX model '{actual_name}' on {DEVICE} (compute_type={COMPUTE_TYPE})...")
        if hotwords:
            log.info(f"  Hotwords: {hotwords[:100]}...")
        t0 = time.time()
        asr_options = {}
        if hotwords:
            asr_options["hotwords"] = hotwords
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
            DEFAULT_BATCH_SIZE = 32
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

        # Group consecutive words by speaker
        groups = []
        current_speaker = None
        current_words = []
        for w in words:
            speaker = w.get("speaker", seg.get("speaker", "?"))
            if speaker != current_speaker and current_words:
                groups.append((current_speaker, current_words))
                current_words = []
            current_speaker = speaker
            current_words.append(w)
        if current_words:
            groups.append((current_speaker, current_words))

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


# -- Transcription (generator for live UI updates) ----------------------------
def transcribe(file, model_name, language, output_format, enable_diarization, min_speakers, max_speakers, batch_size, hotwords):
    """Yields (status, transcript, subtitle_file) tuples for live progress."""
    global _previous_subtitle
    request_id = f"req-{int(time.time()*1000) % 100000}"
    log.info(f"[{request_id}] == New transcription request ==")
    log.info(f"[{request_id}] File: {file}")

    if file is None:
        log.warning(f"[{request_id}] No file provided")
        yield "No file uploaded.", "", None
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
    log.info(f"[{request_id}] Model: {model_name}")
    log.info(f"[{request_id}] Language: {language}")
    log.info(f"[{request_id}] Output format: {output_format}")
    log.info(f"[{request_id}] Diarization: {enable_diarization}")
    log.info(f"[{request_id}] Batch size: {batch_size}")
    if hotwords_str:
        log.info(f"[{request_id}] Hotwords: {hotwords_str[:100]}")
    log.info(f"[{request_id}] Device: {DEVICE}, compute_type: {COMPUTE_TYPE}")

    # -- Phase 1: Load whisperX model --
    yield f"Loading whisperX model '{model_name}'...", "", None
    log.info(f"[{request_id}] Loading whisperX model...")
    t0_model = time.time()
    m = load_whisper(model_name, hotwords=hotwords_str or None)
    model_time = time.time() - t0_model
    log.info(f"[{request_id}] WhisperX model ready in {model_time:.2f}s")

    lang = None if (not language or language == "Auto-detect") else language

    # -- Phase 2: Load audio --
    yield f"Loading audio{file_size_str}...", "", None

    # Verify the file still exists (Gradio temp files can vanish between yields)
    if not os.path.exists(file):
        log.error(f"[{request_id}] File no longer exists: {file}")
        yield f"Error: uploaded file no longer exists (may have been cleaned up)", "", None
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
            yield f"Error copying file: {e}", "", None
            return

    log.info(f"[{request_id}] Loading audio...")
    t0_audio = time.time()
    try:
        audio = whisperx.load_audio(safe_path)
    except Exception as e:
        log.error(f"[{request_id}] Failed to load audio: {e}")
        traceback.print_exc()
        yield f"Error loading audio: {e}", "", None
        return
    finally:
        # Remove the safe copy now that audio is in memory
        try:
            os.remove(safe_path)
        except Exception:
            pass
    audio_duration = len(audio) / 16000  # whisperx loads at 16kHz
    log.info(f"[{request_id}] Audio loaded in {time.time()-t0_audio:.2f}s ({audio_duration:.0f}s / {audio_duration/60:.1f} min)")

    # -- Phase 3: Transcribe (batched) --
    yield f"Transcribing{file_size_str} (batch_size={batch_size})...", "", None
    log.info(f"[{request_id}] >> Starting batched transcription...")
    t0 = time.time()

    try:
        result = m.transcribe(
            audio,
            language=lang,
            batch_size=batch_size,
        )
    except Exception as e:
        log.error(f"[{request_id}] Transcription FAILED: {e}")
        traceback.print_exc()
        yield f"Error: {e}", "", None
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
    formatted_lines = []
    for seg in result.get("segments", []):
        ts = format_timestamp_display(seg.get("start", 0))
        formatted_lines.append(f"[{ts}] {seg.get('text', '').strip()}")
    yield f"Transcribed {num_segments} segments, aligning...", "\n".join(formatted_lines), None

    # -- Phase 4: Word-level alignment (wav2vec2) --
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
        )
        align_elapsed = time.time() - t0_align
        log.info(f"[{request_id}]   Alignment complete in {align_elapsed:.1f}s")
    except Exception as e:
        log.warning(f"[{request_id}]   Alignment failed (proceeding without): {e}")
        # result still has segment-level timestamps, just not word-level

    # Rebuild display after alignment (timestamps may have been refined)
    formatted_lines = []
    for seg in result.get("segments", []):
        ts = format_timestamp_display(seg.get("start", 0))
        formatted_lines.append(f"[{ts}] {seg.get('text', '').strip()}")
    yield f"Aligned {num_segments} segments", "\n".join(formatted_lines), None

    # -- Phase 5: Speaker diarization (optional) --
    if enable_diarization and DIARIZATION_AVAILABLE:
        yield "Loading diarization pipeline...", "\n".join(formatted_lines), None
        log.info(f"[{request_id}] Running speaker diarization...")
        min_spk = int(min_speakers) if min_speakers and int(min_speakers) > 0 else None
        max_spk = int(max_speakers) if max_speakers and int(max_speakers) > 0 else None
        if min_spk or max_spk:
            log.info(f"[{request_id}]   Speaker constraints: min={min_spk}, max={max_spk}")
        t0_diar = time.time()
        try:
            dpipe = load_diarization()
            if dpipe is not None:
                yield "Running speaker diarization...", "\n".join(formatted_lines), None
                diarize_segments = dpipe(audio, min_speakers=min_spk, max_speakers=max_spk)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                # Split segments at speaker boundaries and long runs
                original_count = len(result.get("segments", []))
                result["segments"] = split_segments_by_speaker(result.get("segments", []))
                new_count = len(result["segments"])
                if new_count != original_count:
                    log.info(f"[{request_id}]   Split {original_count} -> {new_count} segments at speaker/sentence boundaries")

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

    transcript = "\n".join(formatted_lines)
    segments = result.get("segments", [])
    num_segments = len(segments)  # update after potential splitting
    log.info(f"[{request_id}]   Text length: {len(transcript)} chars")

    # -- Phase 6: Generate subtitle file --
    subtitle_file = None
    has_speakers = enable_diarization and any(seg.get("speaker") for seg in segments)

    if output_format == "txt":
        yield "Generating txt file...", transcript, None
        log.info(f"[{request_id}] Generating txt file...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        tmp.write(transcript)
        tmp.close()
        subtitle_file = tmp.name
    elif output_format in ("srt", "vtt"):
        yield f"Generating {output_format} file...", transcript, None
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
    speed = audio_duration / transcribe_elapsed if transcribe_elapsed > 0 else 0
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

    yield done_msg, transcript, subtitle_file


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

# -- Custom CSS ----------------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}
.header-wrap {
    text-align: center;
    padding: 0.75rem 0 0.5rem;
}
.header-wrap h2 {
    margin: 0 0 2px;
    font-weight: 700;
}
.header-wrap .sub {
    opacity: 0.45;
    font-size: 0.8rem;
}
.header-wrap .gpu {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.75rem;
    opacity: 0.55;
    margin-top: 4px;
}
footer { display: none !important; }
#copy-transcript-btn {
    max-width: 140px;
    margin-left: auto;
    margin-top: -0.5rem;
}
#copy-transcript-btn button {
    font-size: 0.8rem;
    padding: 4px 12px;
}
"""

# -- Gradio UI -----------------------------------------------------------------
log.info("Building Gradio UI...")

with gr.Blocks(title="WhisperX Transcription") as demo:

    gr.HTML(f"""
        <div class="header-wrap">
            <h2>WhisperX Transcription</h2>
            <div class="sub">whisperX (faster-whisper + wav2vec2 alignment + pyannote diarization)</div>
            <div class="gpu">{GPU_INFO_STR}</div>
        </div>
    """)

    # -- Settings first, then upload --
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large", "turbo"],
            value="turbo",
            label="Model",
        )
        lang_dropdown = gr.Dropdown(
            choices=LANGUAGES,
            value="Auto-detect",
            label="Language",
        )
        format_dropdown = gr.Dropdown(
            choices=["txt", "srt", "vtt"],
            value="srt",
            label="Format",
        )
    with gr.Row():
        diarize_checkbox = gr.Checkbox(
            label="Speaker diarization (identify who is speaking)",
            value=False,
            interactive=DIARIZATION_AVAILABLE,
        )
        min_speakers_input = gr.Number(
            value=0,
            label="Min speakers (0 = auto)",
            minimum=0,
            maximum=20,
            precision=0,
            interactive=DIARIZATION_AVAILABLE,
        )
        max_speakers_input = gr.Number(
            value=0,
            label="Max speakers (0 = auto)",
            minimum=0,
            maximum=20,
            precision=0,
            interactive=DIARIZATION_AVAILABLE,
        )
        batch_slider = gr.Slider(
            minimum=1,
            maximum=64,
            step=1,
            value=DEFAULT_BATCH_SIZE,
            label="Batch size (higher = faster, more VRAM)",
        )

    hotwords_input = gr.Textbox(
        label="Hotwords (names, jargon, or terms the model might mishear)",
        placeholder="e.g. proper nouns, product names, technical terms",
        lines=1,
        max_lines=1,
    )

    # Upload triggers transcription automatically with current settings
    file_input = gr.File(
        label="Upload Audio/Video (transcription starts automatically)",
        file_types=["audio", "video"],
        height=140,
    )
    transcribe_btn = gr.Button(
        "Transcribe",
        variant="primary",
        interactive=False,
    )

    # -- Output --
    status_text = gr.Textbox(
        label="Status",
        lines=1,
        max_lines=1,
        interactive=False,
        placeholder="Ready",
    )
    output_text = gr.Textbox(
        label="Transcript",
        lines=14,
        max_lines=14,
        placeholder="Transcript will appear here...",
        elem_id="transcript-box",
    )
    copy_btn = gr.Button(
        "Copy Transcript",
        size="sm",
        variant="secondary",
        elem_id="copy-transcript-btn",
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
    output_file = gr.File(label="Subtitles", height=50, interactive=False)

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

    all_inputs = [file_input, model_dropdown, lang_dropdown, format_dropdown, diarize_checkbox, min_speakers_input, max_speakers_input, batch_slider, hotwords_input]
    all_outputs = [status_text, output_text, output_file]

    notification_js = """(status) => {
        if (!status || !status.startsWith("Done --")) return;
        if ("Notification" in window && Notification.permission === "granted") {
            new Notification("Transcription Complete", {
                body: status,
                icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙</text></svg>"
            });
        }
    }"""

    # Auto-transcribe on upload (settings are above, so they're already configured)
    file_input.upload(
        fn=None,
        js="""() => {
            if ("Notification" in window && Notification.permission === "default") {
                Notification.requestPermission();
            }
        }""",
    ).then(
        fn=transcribe,
        inputs=all_inputs,
        outputs=all_outputs,
    ).then(
        fn=None,
        inputs=[status_text],
        js=notification_js,
    )

    # Manual re-transcribe button (for changing settings on an already-uploaded file)
    transcribe_btn.click(
        fn=transcribe,
        inputs=all_inputs,
        outputs=all_outputs,
    ).then(
        fn=None,
        inputs=[status_text],
        js=notification_js,
    )

# -- Launch --------------------------------------------------------------------
log.info("Launching Gradio on 0.0.0.0:7860...")
try:
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
except Exception as e:
    log.error(f"Failed to launch: {e}")
    traceback.print_exc()
    sys.exit(1)
