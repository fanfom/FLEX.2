#!/usr/bin/env python3
"""RunPod Serverless Handler"""
import os
import sys
import json
import base64
import asyncio
from typing import Dict, Any, List, Optional

# Add ComfyUI to path
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
sys.path.insert(0, COMFYUI_PATH)

import torch
from PIL import Image

from src.image_utils import prepare_images, pil_to_b64
from src.workflow_builder import build_workflow
from src.comfy_executor import ComfyExecutor

# Global executor instance
executor: Optional[ComfyExecutor] = None

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

# Model paths
MODELS_BASE = CONFIG.get("models", {}).get("base_path", "/runpod-volume")
CHECKPOINT = CONFIG.get("models", {}).get("checkpoint", "models/checkpoints/model.safetensors")
CLIP = CONFIG.get("models", {}).get("clip", "models/clip/clip.safetensors")
VAE = CONFIG.get("models", {}).get("vae", "models/vae/vae.safetensors")

# Inference defaults
DEFAULT_STEPS = CONFIG.get("inference", {}).get("default_steps", 20)
DEFAULT_CFG = CONFIG.get("inference", {}).get("default_cfg", 3.5)


async def init_executor():
    """Initialize ComfyUI executor once per cold start"""
    global executor
    
    print(f"[INIT] Models base: {MODELS_BASE}")
    print(f"[INIT] Checkpoint: {CHECKPOINT}")
    print(f"[INIT] Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    executor = ComfyExecutor(MODELS_BASE)
    await executor.initialize()
    
    print("[INIT] Executor ready")


def generate_seed() -> int:
    """Generate random seed"""
    import time
    return int(time.time() * 1000) % 2147483647


async def process_images(
    images: List[str],
    prompt: str,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
) -> List[str]:
    """Process images and return base64 results"""
    
    if not executor:
        await init_executor()
    
    # Prepare input images
    input_pil_images = prepare_images(images) if images else None
    
    # Get dimensions from first image if available
    if input_pil_images:
        width, height = input_pil_images[0].size
    
    # Build workflow
    workflow = build_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint_path=CHECKPOINT,
        clip_path=CLIP,
        vae_path=VAE,
        ref_images=images,  # Will be handled by executor
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed or generate_seed(),
    )
    
    # Execute
    results = await executor.generate(workflow, input_pil_images)
    
    # Convert to base64
    return [pil_to_b64(img) for img in results]


async def handler(event) -> Dict[str, Any]:
    """
    Main handler for RunPod serverless
    
    Expected input:
    {
        "prompt": "A beautiful sunset over mountains",
        "images": ["base64_image1", "base64_image2"],  // optional, up to 4
        "negative_prompt": "",  // optional
        "steps": 20,  // optional
        "cfg": 3.5,  // optional
        "width": 1024,  // optional
        "height": 1024,  // optional
        "seed": 12345  // optional
    }
    
    Returns:
    {
        "images": ["base64_image1"],
        "seed": 12345,
        "stats": {...}
    }
    """
    try:
        body = event.get("input", {})
        
        prompt = body.get("prompt", "")
        images = body.get("images", [])
        negative_prompt = body.get("negative_prompt", "")
        steps = body.get("steps", DEFAULT_STEPS)
        cfg = body.get("cfg", DEFAULT_CFG)
        width = body.get("width", 1024)
        height = body.get("height", 1024)
        seed = body.get("seed")
        
        # Validate
        if not prompt:
            return {"error": "No prompt provided"}
        
        if len(images) > 4:
            return {"error": "Maximum 4 images allowed"}
        
        print(f"[HANDLER] Processing: prompt='{prompt[:50]}...', images={len(images)}")
        
        # Process
        results = await process_images(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            seed=seed,
        )
        
        return {
            "images": results,
            "seed": seed or generate_seed(),
            "count": len(results)
        }
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e)}


# RunPod serverless interface
def main():
    import runpod
    
    # Initialize executor
    asyncio.run(init_executor())
    
    runpod.serverless.start({
        "handler": handler,
        "return_ray": False,
        "gpu_ids": ["0"],
    })


if __name__ == "__main__":
    main()
