FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Slim OpenCV/runtime deps (+ service-specific overlay when needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Copy requirements first for better caching
COPY requirements.txt .

# App deps only — torch/torchvision ship in the base image (do not re-pip CUDA wheels)
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

COPY app/ ./app/

# Models are bind-mounted at runtime (./models:/app/models:ro); do not bake weights into the image
RUN mkdir -p /app/models /app/results

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
