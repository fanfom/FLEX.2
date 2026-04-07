# Берём за основу официальный, уже оптимизированный образ от RunPod.
# Он уже содержит всё для совместимости с SillyTavern и быстрого старта.
FROM runpod/worker-comfyui:latest

# Устанавливаем недостающие системные зависимости для работы кастомных нод.
# Это чисто, потому что образ не раздувается готовыми моделями.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER comfyui

# Добавляем КАСТОМНЫЕ НОДЫ, которых нет в официальном образе.
# ВАЖНО: Не копируем никакие модели, они будут на Network Volume.
WORKDIR /comfyui/custom_nodes

# 1. Нода для загрузки Base64 изображений от SillyTavern.
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git

# 2. Нода для img2img.
RUN git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git

# 3. Нода для вывода результата в Base64.
RUN git clone https://github.com/ramyma/A8R8_ComfyUI_nodes.git

# Создаём символические ссылки на модели из Network Volume (место хранения моделей в RunPod)
RUN mkdir -p /comfyui/models/clip /comfyui/models/vae
RUN ln -s /runpod-volume/models/clip/qwen_3_8b_fp8mixed.safetensors /comfyui/models/clip/ || true
RUN ln -s /runpod-volume/models/vae/flux2-vae.safetensors /comfyui/models/vae/ || true
# Для чекпоинтов — добавляем папку checkpoints, если её нет, и создаём ссылку
RUN mkdir -p /comfyui/models/checkpoints
RUN ln -s /runpod-volume/models/checkpoints/snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors /comfyui/models/checkpoints/ || true

# Оставляем CMD из родительского образа — он уже умеет правильно запускать ComfyUI и обрабатывать запросы.
