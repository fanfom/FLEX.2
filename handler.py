import os, io, base64
import torch
from PIL import Image
import runpod
from diffusers import Flux2KleinPipeline

MODEL_ID = os.environ.get("MODEL_ID", "/runpod-volume/models/FLUX.2-klein-9B")
DTYPE = torch.bfloat16
DEVICE = "cuda"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", 
    torch_dtype=torch.bfloat16 
)

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_pipe_once():
    global pipe
    if pipe is not None:
        return

    print("Loading model from:", MODEL_ID)

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        local_files_only=True
    )

    pipe.enable_model_cpu_offload()

def handler(event):
    load_pipe_once()

    inp = event["input"]
    prompt = inp["prompt"]

    height = int(inp.get("height", 1024))
    width = int(inp.get("width", 1024))
    guidance_scale = float(inp.get("guidance_scale", 1.0))
    num_inference_steps = int(inp.get("num_inference_steps", 4))
    seed = int(inp.get("seed", 0))

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return {
        "image_b64": pil_to_b64_png(image),
        "meta": {
            "model": MODEL_ID,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
    }

runpod.serverless.start({"handler": handler})
