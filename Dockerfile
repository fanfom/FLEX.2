FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
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

RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    pillow==10.3.0 \
    aiohttp==3.9.5

RUN git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

RUN pip install --no-cache-dir runpod==1.1.0

WORKDIR /app
COPY handler.py config.yaml ./

ENV COMFYUI_PATH=/comfyui
ENV MODELS_BASE=/runpod-volume
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True

CMD ["python", "handler.py"]
