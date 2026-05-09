FROM python:3.13-slim

WORKDIR /app

# Install deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Transcript cache lives here; mount as a volume to persist across restarts.
ENV CACHE_DIR=/app/cache
RUN mkdir -p /app/cache

CMD ["python", "main.py"]
