FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Устанавливаем всё необходимое за один RUN
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 по умолчанию
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Виртуальное окружение
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch (cu124)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Базовые зависимости
RUN pip install --no-cache-dir pillow==10.3.0 aiohttp==3.9.5 runpod==1.1.0

# ComfyUI
RUN git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && pip install --no-cache-dir -r requirements.txt

# Кастомные ноды
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git 

# Конфиг путей для моделей
RUN echo "runpod:\n    base_path: /runpod-volume/models\n    checkpoints: checkpoints/\n    clip: clip/\n    vae: vae/\n    unet: unet/\n    loras: loras/" > /comfyui/extra_model_paths.yaml

WORKDIR /app
COPY handler.py config.yaml ./

CMD ["python", "handler.py"]
