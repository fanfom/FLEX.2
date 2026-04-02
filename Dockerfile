# ============================================
# ComfyUI Serverless Worker for RunPod
# Optimized for Flux with Reference Images
# ============================================

# Base image with CUDA and PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent Python bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set ComfyUI port
ENV COMFYUI_PORT=8199

WORKDIR /app

# ============================================
# System dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    # Fonts for image processing
    fonts-dejavu-core \
    fonts-freefont-ttf \
    fontconfig \
    # Process management
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# ============================================
# Clone ComfyUI
# ============================================
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install xformers for better memory efficiency
RUN pip install --no-cache-dir xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# ============================================
# Install Custom Nodes
# ============================================
WORKDIR /app/ComfyUI/custom_nodes

# Reference Chain Node (для base64 изображений)
RUN git clone --depth 1 https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git

# SaveImage64 Node (возвращает base64)
RUN git clone --depth 1 https://github.com/Kijan/AutoScaledImage.git \
    && mv AutoScaledImage/save_image_64.py /app/ComfyUI/custom_nodes/ \
    || git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git

# ============================================
# Clone worker-comfyui
# ============================================
WORKDIR /app
RUN git clone --depth 1 https://github.com/runpod-workers/worker-comfyui.git /app/worker

# Install worker dependencies
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

# ============================================
# Copy application files
# ============================================
COPY config.json /app/config.json
COPY src/ /app/src/

# ============================================
# Model paths symlinks (если модели на volume)
# ============================================

# Create symlinks to network volume
RUN ln -sf /runpod-volume/models/checkpoints /app/ComfyUI/models/checkpoints \
    && ln -sf /runpod-volume/models/clip /app/ComfyUI/models/clip \
    && ln -sf /runpod-volume/models/vae /app/ComfyUI/models/vae \
    && ln -sf /runpod-volume/models/unet /app/ComfyUI/models/unet
# ============================================
# Health check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${COMFYUI_PORT}/system_stats || exit 1

# Expose port
EXPOSE ${COMFYUI_PORT}

# ============================================
# Start
# ============================================
WORKDIR /app

CMD ["python", "-m", "worker.main"]
