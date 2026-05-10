FROM denoland/deno:bin-2.7.14 AS deno

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libpython3.12t64 \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp now requires a JS runtime to decipher YouTube Music / signature-protected
# streams. Deno is yt-dlp's default supported runtime.
COPY --from=deno /deno /usr/local/bin/deno

# Reproducible pip installs:
# - PIP_NO_COMPILE: don't write .pyc bytecode at install time. .pyc embeds
#   build timestamps → different layer hash on every build → 4+ GB re-push
#   to Docker Hub even when inputs didn't change.
# - PYTHONDONTWRITEBYTECODE: no .pyc files written by Python at runtime
#   either (bot/whisper containers don't need them).
# - SOURCE_DATE_EPOCH: pip + setuptools respect this for normalising file
#   mtimes inside installed packages. Pinned to 2024-01-01 (arbitrary fixed
#   epoch). Combined with PIP_NO_COMPILE, the heavy install layer is fully
#   content-addressable and Docker Hub's cache survives across rebuilds.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=1704067200

# ─── Heavy stable layer ──────────────────────────────────────────────────────
# whisperx + gradio pull in torch + cuDNN + transformers (~4 GB on disk).
# This layer is rebuilt only when requirements.txt changes; without
# PIP_NO_COMPILE it changes on every build.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# torchcodec is pulled in as a transitive dep but has a C++ ABI mismatch with
# the installed PyTorch (undefined symbol). Neither whisperX nor our code uses
# it -- whisperX uses ffmpeg directly, pyannote falls back to torchaudio.
# Removing it eliminates the noisy startup warning from pyannote.
RUN pip uninstall -y torchcodec 2>/dev/null || true

# ─── Volatile yt-dlp layer (small, frequent updates) ─────────────────────────
# Kept separate from the heavy install above so a yt-dlp release doesn't
# invalidate the multi-GB whisperx layer. Pinned in requirements.txt with
# ARG override so a `--build-arg YT_DLP_VERSION=2026.5.1` rebuilds only this
# layer.
ARG YT_DLP_VERSION=2026.3.17
RUN pip install --no-cache-dir --break-system-packages "yt-dlp>=${YT_DLP_VERSION}"

# whisperX ships a Lightning v1.5.4 checkpoint that gets auto-upgraded at
# runtime on every start. We previously ran the upgrade utility at build
# time to make this persistent; it's now disabled because the upgrade CLI
# is broken on PyTorch ≥ 2.6:
#   - PyTorch 2.6 flipped torch.load's `weights_only` default to True.
#   - The whisperX checkpoint pickles `omegaconf.listconfig.ListConfig`
#     which isn't in the safe-globals allowlist.
#   - The CLI doesn't expose a way to pass weights_only=False.
# Result: `python -m lightning.pytorch.utilities.upgrade_checkpoint` errors
# out with `WeightsUnpickler error: Unsupported global ListConfig`.
#
# whisperX itself loads the checkpoint with weights_only=False at runtime,
# so the in-memory upgrade still happens — only the persistence step is
# missing. Cost: one extra INFO log line on first model load. Worth it
# vs. shipping a custom wrapper that monkey-patches torch.load.
#
# Re-enable when Lightning either (a) adds an unsafe-load flag to the CLI
# or (b) updates its safe-globals to include omegaconf types.

# ─── Non-root user ────────────────────────────────────────────────────────────
# ubuntu:24.04 ships a default `ubuntu` user at uid 1000. Reuse it instead of
# creating a fresh user — fewer surprises with bind-mounted host paths.
#
# All writable state lives under paths owned by uid 1000 in the image:
#   /app                                 — code (chowned via COPY --chown)
#   /data                                — runtime state (history.json etc.)
#   /home/ubuntu/.cache/huggingface      — HF model downloads
#   /home/ubuntu/.cache/torch            — torch.hub (wav2vec2 alignment)
#
# Compose maps each of these to a NAMED Docker volume. Docker initialises
# new named volumes by copying the image-path's ownership and contents, so
# a fresh `compose up` works on any host without manual chown — no host
# uid mismatch risk because nothing writable is bind-mounted from the host.
ENV HF_HOME=/home/ubuntu/.cache/huggingface \
    XDG_CACHE_HOME=/home/ubuntu/.cache \
    TORCH_HOME=/home/ubuntu/.cache/torch

RUN install -d -o ubuntu -g ubuntu \
        /app /data \
        /home/ubuntu/.cache \
        /home/ubuntu/.cache/huggingface \
        /home/ubuntu/.cache/torch

WORKDIR /app

# Switch to runtime user. Everything below this line runs as ubuntu (uid 1000).
# COPY --chown still works under non-root because BuildKit applies the
# ownership during layer creation, not as the running user.
USER ubuntu

# ─── Pre-warm wav2vec2 alignment models ──────────────────────────────────────
# Run BEFORE the COPY app.py below so that editing source code doesn't bust
# this expensive cache layer. Cache invalidates only when ALIGN_LANGS or the
# pip-installed whisperx version changes.
#
# Lives in TORCH_HOME (torch.hub) which is NOT volume-mounted — so the
# downloaded files stay in the image layer and survive container recreations.
# Other languages download lazily on first use; baking all of them would
# balloon the image. Override ALIGN_LANGS to pre-warm more:
#   docker compose build --build-arg ALIGN_LANGS=en,es,fr,ja whisper
ARG ALIGN_LANGS=en
RUN for lang in $(echo "$ALIGN_LANGS" | tr ',' ' '); do \
        echo "Pre-warming wav2vec2 alignment model for '$lang'..." && \
        python3 -c "import whisperx; whisperx.load_align_model('$lang', device='cpu')" \
        || echo "WARN: pre-warm for '$lang' failed (will download lazily)"; \
    done

# ─── Source code (changes here invalidate only this layer + CMD) ─────────────
COPY --chown=ubuntu:ubuntu app.py .

EXPOSE 7860

CMD ["python3", "app.py"]
