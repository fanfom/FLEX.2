"""Build ComfyUI workflow JSON for Flux"""
from typing import List, Optional, Dict, Any

def build_workflow(
    prompt: str,
    negative_prompt: str = "",
    checkpoint_path: str = "",
    clip_path: str = "",
    vae_path: str = "",
    ref_images: Optional[List[str]] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 3.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build minimal Flux workflow"""
    
    workflow = {
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint_path
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 1]
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 1]
            }
        },
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": cfg
            }
        },
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0]
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["3", 2]
            }
        },
        "10": {
            "class_type": "PreviewImage",
            "inputs": {
                "images": ["9", 0]
            }
        }
    }
    
    # Add reference image nodes if provided
    if ref_images:
        workflow = add_reference_nodes(workflow, ref_images, clip_path, vae_path)
    
    return workflow

def add_reference_nodes(
    workflow: Dict[str, Any],
    ref_images: List[str],
    clip_path: str,
    vae_path: str
) -> Dict[str, Any]:
    """Add Flux Kontext reference nodes for image-to-image"""
    
    # Load reference images
    image_node_id = 20
    for i, img_path in enumerate(ref_images[:4]):
        workflow[str(image_node_id)] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": img_path
            }
        }
        image_node_id += 1
    
    # Clip loader for Qwen
    workflow["21"] = {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": clip_path,
            "type": "flux2",
            "device": "default"
        }
    }
    
    # VAE loader
    workflow["22"] = {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": vae_path
        }
    }
    
    # Scale reference image to match target
    workflow["23"] = {
        "class_type": "FluxKontextImageScale",
        "inputs": {
            "method": "nearest-exact",
            "scale_by": 1.0
        }
    }
    
    # Text encode with reference
    workflow["24"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {
            "prompt": workflow["4"]["inputs"]["text"]
        }
    }
    
    # Connect the flow
    # Update main nodes to use reference conditioning
    return workflow
