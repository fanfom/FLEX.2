FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Системные зависимости (минимально)
RUN apt-get update && apt-get install -y \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 по умолчанию
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Виртуальное окружение
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch (cu124 совместим с вашей CUDA 12.6)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Базовые зависимости
RUN pip install --no-cache-dir pillow==10.3.0 aiohttp==3.9.5 runpod==1.1.0

# ComfyUI (чистая установка)
RUN git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui && pip install --no-cache-dir -r requirements.txt

# ===== КАСТОМНЫЕ НОДЫ =====
WORKDIR /comfyui/custom_nodes

# 1. ReferenceChain (вам нужен для img2img?)
RUN git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git

# 2. Base64 нода — но этот репозиторий может иметь другое название класса
#    Лучше использовать проверенный: ComfyUI-KJNodes (там есть LoadImageBase64)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git

# Альтернатива: если хотите именно images_base64, то раскомментируйте:
# RUN git clone https://github.com/GrailGreg/images_base64.git
# Но KJNodes более известен и поддерживается.

# 3. (Опционально) Для удобства — менеджер нод, но не обязательно

# Конфиг моделей (пути)
RUN echo "runpod:\n    base_path: /runpod-volume/models\n    checkpoints: checkpoints/\n    clip: clip/\n    vae: vae/\n    unet: unet/\n    loras: loras/" > /comfyui/extra_model_paths.yaml

# Хендлер
WORKDIR /app
COPY handler.py config.yaml ./

# Запуск
CMD ["python", "handler.py"]
