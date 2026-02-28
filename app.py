import sys
import os
import time
import logging
import tempfile
import traceback
import glob
import shutil

# -- Logging setup -------------------------------------------------------------
# Set DEBUG_MODE=0 in compose.yaml env when you want clean logs
DEBUG_MODE = os.environ.get("DEBUG_MODE", "1") == "1"

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("whisper-ui")
log.setLevel(logging.DEBUG)

# Even in debug mode, these are pure noise (chunk-level upload logs, PIL plugins, etc.)
for noisy in ("PIL", "python_multipart", "python_multipart.multipart",
              "multipart", "asyncio", "watchfiles",
              "httpcore", "httpcore.http11", "httpcore.connection",
              "httpx", "filelock", "faster_whisper",
              "matplotlib", "matplotlib.font_manager",
              "pyannote", "pyannote.audio", "torchcodec",
              "fsspec", "fsspec.local", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Suppress torchcodec UserWarning (we load audio manually, pyannote's warning is noise)
# The message starts with \n so we need a permissive regex
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, message="TensorFloat-32")
warnings.filterwarnings("ignore", category=UserWarning, message="std\\(\\): degrees of freedom")

if not DEBUG_MODE:
    # Production: silence everything except our logs + access logs
    for noisy in ("httpcore", "httpx", "urllib3", "gradio", "starlette"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
else:
    log.info("DEBUG_MODE is ON -- verbose third-party logging enabled")

log.info("=" * 60)
log.info("Whisper Transcription UI starting")
log.info("=" * 60)

# -- Python / env info ---------------------------------------------------------
log.info(f"Python: {sys.version}")
log.info(f"CWD: {os.getcwd()}")
log.info(f"ENV NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unset')}")
log.info(f"ENV CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

# -- GPU detection (lightweight, no torch import) ------------------------------
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
GPU_INFO_STR = "CPU mode (no GPU detected)"

try:
    import ctranslate2
    gpu_count = ctranslate2.get_cuda_device_count()
    log.info(f"ctranslate2 CUDA device count: {gpu_count}")
    if gpu_count > 0:
        DEVICE = "cuda"
        COMPUTE_TYPE = "float16"
        # Try to get detailed GPU info via nvidia-smi
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
                GPU_INFO_STR = f"{gpu_name}  |  {vram_mb // 1024} GB VRAM  |  faster-whisper (CTranslate2)  |  float16"
                log.info(f"GPU 0: {gpu_name} ({vram_mb} MB VRAM)")
            else:
                GPU_INFO_STR = f"CUDA GPU detected ({gpu_count} device(s))  |  faster-whisper (CTranslate2)  |  float16"
        except Exception:
            GPU_INFO_STR = f"CUDA GPU detected ({gpu_count} device(s))  |  faster-whisper (CTranslate2)  |  float16"
except Exception as e:
    log.warning(f"ctranslate2 CUDA detection failed: {e}")
    log.info("Falling back to CPU mode")

log.info(f"Selected device: {DEVICE}, compute_type: {COMPUTE_TYPE}")

# -- Import faster-whisper -----------------------------------------------------
log.info("Importing faster-whisper...")
t0 = time.time()
from faster_whisper import WhisperModel
log.info(f"faster-whisper imported in {time.time()-t0:.2f}s")

# -- Import gradio -------------------------------------------------------------
log.info("Importing gradio...")
t0 = time.time()
import gradio as gr
log.info(f"Gradio {gr.__version__} imported in {time.time()-t0:.2f}s")

# -- Import pyannote (optional) ------------------------------------------------
diarization_pipeline = None
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if HF_TOKEN:
    log.info("Importing pyannote.audio...")
    t0 = time.time()
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        log.info(f"pyannote.audio imported in {time.time()-t0:.2f}s")
    except ImportError:
        log.warning("pyannote.audio not installed -- diarization disabled")
        PyannotePipeline = None
else:
    log.info("HF_TOKEN not set -- speaker diarization disabled")
    PyannotePipeline = None

# -- Model management ---------------------------------------------------------
whisper_model = None
current_model_name = None


def load_whisper(model_name):
    global whisper_model, current_model_name
    if model_name != current_model_name:
        log.info(f"Loading whisper model '{model_name}' on {DEVICE} (compute_type={COMPUTE_TYPE})...")

        t0 = time.time()
        whisper_model = WhisperModel(
            model_name if model_name != "large" else "large-v3",
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        elapsed = time.time() - t0

        log.info(f"  Whisper model loaded in {elapsed:.2f}s")
        current_model_name = model_name
    else:
        log.info(f"Whisper model '{model_name}' already loaded, reusing")
    return whisper_model


def load_diarization():
    global diarization_pipeline
    if diarization_pipeline is not None:
        log.info("Diarization pipeline already loaded, reusing")
        return diarization_pipeline
    if PyannotePipeline is None:
        return None
    log.info("Loading pyannote diarization pipeline...")
    t0 = time.time()
    try:
        diarization_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
        import torch
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
        log.info(f"  Diarization pipeline loaded in {time.time()-t0:.2f}s")
    except Exception as e:
        log.error(f"  Failed to load diarization pipeline: {e}")
        diarization_pipeline = None
    return diarization_pipeline


def assign_speakers(segments, diarization_result):
    """Assign speaker labels to whisper segments using pyannote diarization.

    diarization_result is a DiarizeOutput dataclass; use exclusive_speaker_diarization
    (no overlapping turns) which maps better to transcription segments.
    """
    # DiarizeOutput wraps Annotation objects
    annotation = getattr(diarization_result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarization_result, "speaker_diarization", diarization_result)
    labeled = []
    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        speaker = "?"
        for turn, _, spk in annotation.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                speaker = spk
                break
        labeled.append((seg, speaker))
    return labeled


# -- Temp file cleanup ---------------------------------------------------------
_previous_subtitle = None  # track last subtitle file for cleanup


def cleanup_upload(file_path):
    """Remove the uploaded file and its parent gradio temp dir if empty."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
        parent = os.path.dirname(file_path)
        # Remove the per-upload hash directory if empty
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


# Run cleanup on startup
cleanup_stale_gradio_tmp()


# -- Transcription (generator for live UI updates) ----------------------------
def transcribe(file, model_name, language, output_format, enable_diarization):
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

    log.info(f"[{request_id}] Model: {model_name}")
    log.info(f"[{request_id}] Language: {language}")
    log.info(f"[{request_id}] Output format: {output_format}")
    log.info(f"[{request_id}] Diarization: {enable_diarization}")
    log.info(f"[{request_id}] Device: {DEVICE}, compute_type: {COMPUTE_TYPE}")

    # -- Phase 1: Load whisper model --
    yield f"Loading whisper model '{model_name}'...", "", None
    log.info(f"[{request_id}] Loading whisper model...")
    t0_model = time.time()
    m = load_whisper(model_name)
    model_time = time.time() - t0_model
    log.info(f"[{request_id}] Whisper model ready in {model_time:.2f}s")

    # Transcription options
    lang = None if (not language or language == "Auto-detect") else language

    log.info(f"[{request_id}] language: {lang}")
    log.info(f"[{request_id}] beam_size: 5, condition_on_previous_text: False")

    # -- Phase 2: Transcribe --
    yield f"Transcribing{file_size_str}...", "", None
    log.info(f"[{request_id}] >> Starting transcription...")
    t0 = time.time()

    try:
        segments_gen, info = m.transcribe(
            file,
            language=lang,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Consume the generator to collect all segments
        segments = []
        formatted_lines = []
        for seg in segments_gen:
            segments.append(seg)
            ts = format_timestamp_display(seg.start)
            formatted_lines.append(f"[{ts}] {seg.text.strip()}")
            # Stream to UI every 10 segments
            if len(segments) % 10 == 0:
                elapsed_so_far = time.time() - t0
                # Log every 50
                if len(segments) % 50 == 0:
                    log.info(f"[{request_id}]   ... {len(segments)} segments processed ({elapsed_so_far:.0f}s)")
                yield (
                    f"Transcribing{file_size_str}... {len(segments)} segments ({elapsed_so_far:.0f}s)",
                    "\n".join(formatted_lines),
                    None,
                )

    except Exception as e:
        log.error(f"[{request_id}] Transcription FAILED: {e}")
        traceback.print_exc()
        yield f"Error: {e}", "", None
        return

    elapsed = time.time() - t0
    num_segments = len(segments)
    detected_lang = info.language
    lang_prob = info.language_probability
    duration = info.duration

    log.info(f"[{request_id}] [OK] Transcription complete")
    log.info(f"[{request_id}]   Time: {elapsed:.1f}s")
    log.info(f"[{request_id}]   Audio duration: {duration:.0f}s ({duration/60:.1f} min)")
    log.info(f"[{request_id}]   Speed: {duration/elapsed:.1f}x realtime")
    log.info(f"[{request_id}]   Segments: {num_segments}")
    log.info(f"[{request_id}]   Detected language: {detected_lang} ({lang_prob:.0%})")

    # -- Phase 3: Speaker diarization (optional) --
    speaker_map = None
    if enable_diarization and PyannotePipeline is not None:
        yield f"Loading audio for diarization...", "\n".join(formatted_lines), None
        log.info(f"[{request_id}] Running diarization...")
        t0_diar = time.time()
        try:
            import torchaudio
            dpipe = load_diarization()
            if dpipe is not None:
                # torchaudio 2.10 uses torchcodec under the hood (needs libpython3.12t64)
                # Load file directly -- torchcodec handles mkv, mp4, wav, etc.
                log.info(f"[{request_id}]   Loading audio with torchaudio...")
                waveform, sample_rate = torchaudio.load(file)
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}
                log.info(f"[{request_id}]   Audio loaded: {waveform.shape[1]/sample_rate:.0f}s, {sample_rate}Hz")

                yield f"Running speaker diarization...", "\n".join(formatted_lines), None
                diar_result = dpipe(audio_input)
                labeled = assign_speakers(segments, diar_result)
                # Rebuild formatted lines with speaker labels
                formatted_lines = []
                for seg, speaker in labeled:
                    ts = format_timestamp_display(seg.start)
                    formatted_lines.append(f"[{ts}] [{speaker}] {seg.text.strip()}")
                speaker_map = [spk for _, spk in labeled]
                diar_time = time.time() - t0_diar
                num_speakers = len(set(spk for _, spk in labeled))
                log.info(f"[{request_id}]   Diarization complete in {diar_time:.1f}s ({num_speakers} speakers)")
            else:
                log.warning(f"[{request_id}]   Diarization pipeline not available")
        except Exception as e:
            log.error(f"[{request_id}]   Diarization failed: {e}")
            traceback.print_exc()
    elif enable_diarization:
        log.warning(f"[{request_id}]   Diarization requested but pyannote not available (check HF_TOKEN)")

    transcript = "\n".join(formatted_lines)
    log.info(f"[{request_id}]   Text length: {len(transcript)} chars")

    # -- Phase 4: Generate subtitle file --
    subtitle_file = None
    if output_format == "txt":
        yield "Generating txt file...", transcript, None
        log.info(f"[{request_id}] Generating txt file...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        tmp.write(transcript)
        tmp.close()
        subtitle_file = tmp.name
        log.info(f"[{request_id}] Subtitle file: {subtitle_file}")
    elif output_format in ("srt", "vtt", "all"):
        yield f"Generating {output_format} file...", transcript, None
        log.info(f"[{request_id}] Generating {output_format} subtitle file...")
        ext = "srt" if output_format != "vtt" else "vtt"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", mode="w")
        if ext == "srt":
            for i, seg in enumerate(segments, 1):
                start_ts = format_timestamp_srt(seg.start)
                end_ts = format_timestamp_srt(seg.end)
                speaker_prefix = ""
                if speaker_map and i - 1 < len(speaker_map):
                    speaker_prefix = f"[{speaker_map[i - 1]}] "
                tmp.write(f"{i}\n{start_ts} --> {end_ts}\n{speaker_prefix}{seg.text.strip()}\n\n")
        else:
            tmp.write("WEBVTT\n\n")
            for idx, seg in enumerate(segments):
                start_ts = format_timestamp_vtt(seg.start)
                end_ts = format_timestamp_vtt(seg.end)
                speaker_prefix = ""
                if speaker_map and idx < len(speaker_map):
                    speaker_prefix = f"[{speaker_map[idx]}] "
                tmp.write(f"{start_ts} --> {end_ts}\n{speaker_prefix}{seg.text.strip()}\n\n")
        tmp.close()
        subtitle_file = tmp.name
        log.info(f"[{request_id}] Subtitle file: {subtitle_file}")

    total_time = model_time + elapsed
    speed = duration / elapsed if elapsed > 0 else 0
    done_msg = f"Done -- {num_segments} segments, {elapsed:.0f}s ({speed:.1f}x realtime), {detected_lang} ({lang_prob:.0%})"
    log.info(f"[{request_id}] == Request complete ({total_time:.1f}s total) ==")

    # Cleanup: remove uploaded file and previous subtitle temp file
    cleanup_upload(file)
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
"""

# -- Gradio UI -----------------------------------------------------------------
log.info("Building Gradio UI...")

with gr.Blocks(title="Whisper Transcription") as demo:

    gr.HTML(f"""
        <div class="header-wrap">
            <h2>Whisper Transcription</h2>
            <div class="sub">faster-whisper (CTranslate2)</div>
            <div class="gpu">{GPU_INFO_STR}</div>
        </div>
        <script>
            // Request notification permission on load
            if ("Notification" in window && Notification.permission === "default") {{
                Notification.requestPermission();
            }}
            window._whisperNotify = function(title, body) {{
                if ("Notification" in window && Notification.permission === "granted") {{
                    if (document.hidden) {{
                        new Notification(title, {{ body: body, icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙</text></svg>" }});
                    }}
                }}
            }};
            // Watch the status field for changes
            const observer = new MutationObserver(function() {{
                const statusEl = document.querySelector('input[aria-label="Status"], textarea[aria-label="Status"]');
                if (!statusEl) return;
                const val = statusEl.value || "";
                if (val.startsWith("Done --")) {{
                    window._whisperNotify("Transcription Complete", val);
                }}
            }});
            setTimeout(function() {{
                const target = document.querySelector('.gradio-container');
                if (target) observer.observe(target, {{ childList: true, subtree: true, characterData: true, attributes: true }});
            }}, 2000);
        </script>
    """)

    # -- Top row: file upload + settings + button --
    file_input = gr.File(
        label="Upload Audio/Video",
        file_types=["audio", "video"],
        height=140,
    )
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large"],
            value="large",
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
    diarize_checkbox = gr.Checkbox(
        label="Speaker diarization (identify who is speaking)",
        value=False,
        interactive=PyannotePipeline is not None,
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
        max_lines=50,
        placeholder="Transcript will appear here...",
    )
    output_file = gr.File(label="Subtitles", height=50)

    # Hidden HTML for triggering notifications from Python
    notify_html = gr.HTML(visible=False)

    # Enable/disable button based on file upload state
    def on_file_change(f):
        if f is not None:
            try:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                fname = os.path.basename(f)
                log.info(f"File uploaded: {fname} ({size_mb:.1f} MB)")
                notify_js = f'<script>window._whisperNotify && window._whisperNotify("Upload Complete", "{fname} ({size_mb:.0f} MB)");</script>'
            except Exception:
                log.info(f"File uploaded: {f}")
                notify_js = '<script>window._whisperNotify && window._whisperNotify("Upload Complete", "File ready");</script>'
        else:
            log.info("File cleared")
            notify_js = ""
        return gr.update(interactive=f is not None), notify_js

    file_input.change(
        fn=on_file_change,
        inputs=[file_input],
        outputs=[transcribe_btn, notify_html],
    )

    transcribe_btn.click(
        fn=transcribe,
        inputs=[file_input, model_dropdown, lang_dropdown, format_dropdown, diarize_checkbox],
        outputs=[status_text, output_text, output_file],
    )

# -- Launch --------------------------------------------------------------------
log.info("Launching Gradio on 0.0.0.0:7860...")
try:
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
except Exception as e:
    log.error(f"Failed to launch: {e}")
    traceback.print_exc()
    sys.exit(1)
