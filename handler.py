import os, io, base64, torch
from PIL import Image
import runpod
from diffusers import Flux2KleinPipeline # Используй тот класс, который реально в библиотеке

MODEL_ID = os.environ.get("MODEL_ID", "/runpod-volume/models/FLUX.2-klein-9B")
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Глобальная переменная для пайплайна
pipe = None

def load_pipe_once():
    global pipe
    if pipe is not None:
        return

    print(f"--- Loading model to A40 VRAM: {MODEL_ID} ---")
    
    # Загружаем сразу в нужной точности
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        local_files_only=True
    )
    
    # ПЕРЕНОСИМ ВСЁ В GPU (На A40 места полно!)
    pipe.to(DEVICE)
    
    # Включаем ускорение внимания (если xformers установлены в контейнере)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

def handler(event):
    load_pipe_once()

    inp = event["input"]
    # Для Klein/Schnell жестко ставим guidance_scale 0 или 1, если юзер не понимает
    # Иначе время генерации удвоится впустую
    gs = float(inp.get("guidance_scale", 1.0))
    steps = int(inp.get("num_inference_steps", 4))

    # Генерация
    with torch.inference_mode(): # Отключаем градиенты для скорости
        image = pipe(
            prompt=inp["prompt"],
            height=int(inp.get("height", 1024)),
            width=int(inp.get("width", 1024)),
            guidance_scale=gs,
            num_inference_steps=steps,
            generator=torch.Generator(device=DEVICE).manual_seed(int(inp.get("seed", 0)))
        ).images[0]

    # Конвертация в b64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image_b64": img_str}

runpod.serverless.start({"handler": handler})
