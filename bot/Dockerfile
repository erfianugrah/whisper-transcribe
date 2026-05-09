FROM python:3.13-slim

# Non-root user for the bot. python:3.13-slim is a Debian image with no
# default unprivileged user, so create one.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid $APP_GID bot && \
    useradd --uid $APP_UID --gid $APP_GID --create-home --shell /bin/bash bot

# Reproducible pip installs — see whisper Dockerfile for rationale. Without
# these, the install layer's hash drifts on every build and Docker Hub
# re-pushes the layer instead of cache-hitting it.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COMPILE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=1704067200

WORKDIR /app

# Install deps first (cached unless requirements.txt changes). Pip install
# happens as root so /usr/local/lib remains root-owned (correct).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=bot:bot main.py prompts.py ./

# Transcript cache lives here; mount as a volume to persist across restarts.
# Pre-created and owned by uid 1000 so the named volume in compose inherits
# correct ownership on first creation.
ENV CACHE_DIR=/app/cache
RUN install -d -o bot -g bot /app/cache

USER bot

CMD ["python", "main.py"]
