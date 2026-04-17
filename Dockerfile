# Берём за основу официальный образ RunPod (уже оптимизирован для ComfyUI и совместимости с SillyTavern)
FROM runpod/worker-comfyui:5.8.5-base-cuda12.8.1

# Устанавливаем переменные для ComfyUI (если ещё не заданы)
ENV COMFYUI_PATH="/comfyui" \
    COMFYUI_MODEL_PATH="/comfyui/models"

# Создаём все нужные папки для моделей
RUN mkdir -p ${COMFYUI_MODEL_PATH}/checkpoints \
             ${COMFYUI_MODEL_PATH}/vae \
             ${COMFYUI_MODEL_PATH}/clip \
             ${COMFYUI_MODEL_PATH}/text_encoders

# ========= ЗАПЕКАЕМ МОДЕЛИ ПРЯМО В ОБРАЗ =========
# 1. Основная модель (FP8 дистиллированная)
ADD https://huggingface.co/badosss/flux_nsfw_2/resolve/main/snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors \
    ${COMFYUI_MODEL_PATH}/checkpoints/snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors

# 2. VAE для Flux.2
ADD https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors \
    ${COMFYUI_MODEL_PATH}/vae/flux2-vae.safetensors

# 3. Текстовые энкодеры (стандартные для Flux)
ADD https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors \
    ${COMFYUI_MODEL_PATH}/clip/clip_l.safetensors

ADD https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors \
    ${COMFYUI_MODEL_PATH}/text_encoders/t5xxl_fp8_e4m3fn.safetensors

# Добавляем кастомные ноды, необходимые для img2img и работы с base64
WORKDIR /comfyui/custom_nodes


# 1. Нода для Reference Chain (img2img с контролем)
RUN git clone https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git
# 2. Нода для выдачи результата в Base64 обратно в SillyTavern
RUN git clone https://github.com/GrailGreg/images_base64.git

