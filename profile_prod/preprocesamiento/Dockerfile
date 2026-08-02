FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the rembg matting model into the image so the first request does not
# trigger a runtime download (and works with no outbound net).
# bake ≡ REMBG_MODEL must stay in sync; override ENV only if you also bake that model.
# isnet-general-use chosen after EC2 hard-subset A/B (u2net failed glued BG);
# birefnet better but ~46s/image — do not bake unless async path exists.
ENV U2NET_HOME=/app/.u2net
ENV REMBG_MODEL=isnet-general-use
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]