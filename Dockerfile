FROM runpod/worker-comfyui:latest-base

WORKDIR /app/ComfyUI/custom_nodes

# Custom nodes
RUN git clone --depth 1 https://github.com/remingtonspaz/ComfyUI-ReferenceChain.git

# Models symlinks (Network Volume обычно в /runpod-volume)
RUN mkdir -p /runpod-volume/models/checkpoints \
             /runpod-volume/models/clip \
             /runpod-volume/models/vae \
             /runpod-volume/models/unet
