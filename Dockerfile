FROM denoland/deno:bin-2.7.14 AS deno

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libpython3.12t64 \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp now requires a JS runtime to decipher YouTube Music / signature-protected
# streams. Deno is yt-dlp's default supported runtime.
COPY --from=deno /deno /usr/local/bin/deno

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt && \
    pip install --no-cache-dir --break-system-packages yt-dlp

# torchcodec is pulled in as a transitive dep but has a C++ ABI mismatch with
# the installed PyTorch (undefined symbol). Neither whisperX nor our code uses
# it -- whisperX uses ffmpeg directly, pyannote falls back to torchaudio.
# Removing it eliminates the noisy startup warning from pyannote.
RUN pip uninstall -y torchcodec 2>/dev/null || true

# whisperX ships a Lightning v1.5.4 checkpoint that gets auto-upgraded at
# runtime on every start. Run the upgrade once at build time.
RUN python3 -m lightning.pytorch.utilities.upgrade_checkpoint \
    /usr/local/lib/python3.12/dist-packages/whisperx/assets/pytorch_model.bin \
    2>/dev/null || true

WORKDIR /app
COPY app.py .

EXPOSE 7860

CMD ["python3", "app.py"]
