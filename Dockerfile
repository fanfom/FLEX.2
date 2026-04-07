# Берём за основу официальный образ RunPod (уже оптимизирован для ComfyUI и совместимости с SillyTavern)
FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

# Добавляем кастомные ноды, необходимые для img2img и работы с base64
WORKDIR /comfyui/custom_nodes


# 1. Нода для Reference Chain (img2img с контролем)
RUN git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git
# 2. Нода для выдачи результата в Base64 обратно в SillyTavern
RUN git clone https://github.com/GrailGreg/images_base64.git

# Создаём символические ссылки на модели из Network Volume (предполагается, что volume примонтирован в /runpod-volume)
RUN mkdir -p /comfyui/models/checkpoints /comfyui/models/clip /comfyui/models/vae \
    && ln -sf /runpod-volume/models/checkpoints/snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors /comfyui/models/checkpoints/ \
    && ln -sf /runpod-volume/models/clip/qwen_3_8b_fp8mixed.safetensors /comfyui/models/clip/ \
    && ln -sf /runpod-volume/models/vae/flux2-vae.safetensors /comfyui/models/vae/ \
    || true  # не падаем, если файлов ещё нет (они появятся на volume позже)

# Оставляем CMD из родительского образа — он уже умеет запускать ComfyUI и обрабатывать запросы SillyTavern
# Никакой дополнительной команды не требуется
