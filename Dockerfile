FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libpython3.12t64 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

WORKDIR /app
COPY app.py .

EXPOSE 7860

CMD ["python3", "app.py"]
