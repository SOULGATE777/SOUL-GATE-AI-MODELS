# ──────────────────────────────────────────────────────────────────────────────
# Frame Extractor — Dockerfile
#
# Base image and structure mirror frontal_prod/preprocesamiento/Dockerfile.
# Port: 8020  (verified free across all docker-compose files in the repo)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.9-slim

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1 / libglib2.0-0 required by OpenCV headless libs.
# libgomp1 required by MediaPipe's native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime environment ────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8020

# ── Network ───────────────────────────────────────────────────────────────────
EXPOSE 8020

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/health')"

# ── Entry point ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020", "--workers", "1"]
