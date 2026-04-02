# Stage 1: Build PyTorch wheels
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_PREFER_BINARY=1
ARG PYTHON_VERSION=3.10

# Install build deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Build minimal PyTorch (optional - use wheels instead)
# This stage is for building custom ops if needed

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VERSION=3.10

# Minimal deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# Create venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install PyTorch from wheel (smaller than pip install)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install inference deps
RUN pip install --no-cache-dir \
    pillow numpy \
    safetensors einops \
    transformers accelerate sentencepiece protobuf peft

# Install ComfyUI core only (not full install)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

# Copy app code
WORKDIR /app
COPY handler.py requirements.txt config.yaml ./
COPY src/ ./src/

# Install runtime deps only
RUN pip install --no-cache-dir -r requirements.txt \
    runpod websockets aiohttp

# Environment
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV COMFYUI_PATH=/comfyui
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

CMD ["python", "handler.py"]
