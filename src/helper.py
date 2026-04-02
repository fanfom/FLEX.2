"""
Utility functions for ComfyUI worker
"""
import os
import json
import base64
import uuid
from pathlib import Path
from typing import Optional


def get_model_path(model_type: str, filename: str) -> str:
    """Get full path to model"""
    base = os.getenv("MODEL_PATH", "/workspace/models")
    return str(Path(base) / model_type / filename)


def save_uploaded_images(images: list, temp_dir: str = "/tmp/uploads") -> list:
    """
    Save base64 images to temp files and return paths
    
    Args:
        images: List of dicts with 'name' and 'image' keys
        temp_dir: Directory to save images
    
    Returns:
        List of file paths
    """
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    
    for i, img_data in enumerate(images):
        img_bytes = img_data["image"]
        
        # Remove data URL prefix
        if "," in img_bytes:
            img_bytes = img_bytes.split(",")[1]
        
        # Decode and save
        data = base64.b64decode(img_bytes)
        filename = f"{uuid.uuid4().hex}_{i}.png"
        filepath = Path(temp_dir) / filename
        
        with open(filepath, "wb") as f:
            f.write(data)
        
        paths.append(str(filepath))
    
    return paths


def build_workflow_from_template(
    template: dict,
    prompt: str,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    images: list = None
) -> dict:
    """Build workflow from template with dynamic values"""
    import random
    
    workflow = json.loads(json.dumps(template))  # Deep copy
    
    # Update prompts
    workflow["5"]["inputs"]["text"] = prompt
    workflow["10"]["inputs"]["text"] = negative_prompt
    
    # Update sampler
    workflow["25"]["inputs"]["steps"] = steps
    workflow["25"]["inputs"]["seed"] = seed if seed else random.randint(0, 2**32 - 1)
    
    # Update dimensions
    workflow["26"]["inputs"]["width"] = width
    workflow["26"]["inputs"]["height"] = height
    
    # Update reference images
    if images:
        workflow["30"]["inputs"]["images_base64"] = images
