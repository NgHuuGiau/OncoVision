# OncoVision Dockerfile
# Build: docker build -t onkovision:latest .
# Run: docker run -d --name onkovision -p 8000:8000 \
#   -v $(PWD)/models:/app/models \
#   -v $(PWD)/output:/app/output \
#   -v $(PWD)/data:/app/data \
#   --gpus all onkovision:latest

# Base image with CUDA 12.6 for NVIDIA GPUs (updated from 12.1.1)
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

# Install system dependencies (deduplicated packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ONCOVISION_HOST=0.0.0.0

# Install Python
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching (web-only: lighter, no PySide6/whisper)
COPY requirements/requirements-web.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy source code (after pip for better layer caching)
COPY . .

# Create necessary directories
RUN mkdir -p \
    models/pretrained/cancers \
    models/trained \
    models/exported \
    output/medical/reports \
    output/medical/processed_images \
    output/medical/overlays \
    output/medical/heatmaps \
    output/medical/exports \
    output/medical/captures \
    output/chat_captures \
    data \
    assets/icons \
    runs

# Create .gitkeep files for empty directories
RUN touch \
    models/pretrained/cancers/.gitkeep \
    models/trained/.gitkeep \
    models/exported/.gitkeep \
    output/medical/reports/.gitkeep \
    output/medical/processed_images/.gitkeep \
    output/medical/overlays/.gitkeep \
    output/medical/heatmaps/.gitkeep \
    output/medical/exports/.gitkeep \
    output/medical/captures/.gitkeep \
    output/chat_captures/.gitkeep \
    runs/.gitkeep

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command: web app (desktop camera UI needs a display, không chạy trong container)
CMD ["python", "web_app.py"]