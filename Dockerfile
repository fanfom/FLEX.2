FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN /usr/bin/python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Зависимости
RUN pip install --no-cache-dir \
    pillow==10.3.0 \
    aiohttp==3.9.5

# ComfyUI — только requirements, без pip install .
RUN git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && pip install --no-cache-dir -r requirements.txt

# RunPod SDK
RUN pip install --no-cache-dir runpod==1.1.0

RUN mkdir -p /comfyui/custom_nodes && cd /comfyui/custom_nodes \
    && git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git
# Код приложения
WORKDIR /app
COPY handler.py config.yaml ./

ENV COMFYUI_PATH=/comfyui
RUN echo "runpod:\n    base_path: /runpod-volume/models\n    checkpoints: checkpoints/\n    clip: clip/\n    vae: vae/\n    unet: unet/\n    loras: loras/" > /comfyui/extra_model_paths.yaml
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True

CMD ["python", "handler.py"]
