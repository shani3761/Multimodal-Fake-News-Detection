# ══════════════════════════════════════════════════════════════════════════
# FakeShield AI — Multi-Stage Dockerfile
# ══════════════════════════════════════════════════════════════════════════
#
# Build (CPU):
#   docker build -t fakeshield-ai .
#
# Build (GPU — requires nvidia-docker):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
#                -t fakeshield-ai-gpu .
#
# Run Streamlit:
#   docker run -p 8501:8501 fakeshield-ai
#
# Run FastAPI:
#   docker run -p 8000:8000 fakeshield-ai python main.py --web --mode api
#
# ══════════════════════════════════════════════════════════════════════════

ARG BASE_IMAGE=python:3.11-slim
ARG PYTHON_VERSION=3.11

# ────────────────────────────────────────────────────────────────────────────
# Stage 1 — system-deps builder
# Installs OS libraries required by OpenCV, ffmpeg, WeasyPrint, etc.
# ────────────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS system-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    wget \
    # OpenCV runtime dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image format support
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # ffmpeg (video processing)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    # Audio (moviepy)
    libsndfile1 \
    # WeasyPrint / Pango / Cairo
    libcairo2-dev \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    # Fonts
    fonts-liberation \
    # SQLite
    libsqlite3-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python dependency builder
# ────────────────────────────────────────────────────────────────────────────
FROM system-builder AS python-builder

WORKDIR /build

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Install Python packages into /build/venv
RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Install CPU-only PyTorch first (smaller image; override for GPU)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt


# ────────────────────────────────────────────────────────────────────────────
# Stage 3 — production image
# ────────────────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS production

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Virtual-env first on PATH
    PATH="/app/venv/bin:$PATH" \
    # Streamlit settings
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Hugging Face cache location
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    TORCH_HOME=/app/.cache/torch

# ── Install runtime OS deps (no build tools) ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    ffmpeg \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 fonts-liberation \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy venv from builder ────────────────────────────────────────────────────
COPY --from=python-builder /build/venv /app/venv

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p data reports temp assets models/weights \
    .cache/huggingface .cache/torch

# ── Non-root user (security best-practice) ───────────────────────────────────
RUN groupadd -r fakeshield && useradd -r -g fakeshield -d /app fakeshield \
    && chown -R fakeshield:fakeshield /app

USER fakeshield

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${STREAMLIT_SERVER_PORT:-8501}/_stcore/health \
      || curl -f http://localhost:8000/health \
      || exit 1

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8501
EXPOSE 8000

# ── Default command: Streamlit UI ─────────────────────────────────────────────
CMD ["python", "main.py", "--web", "--mode", "streamlit"]
