"""BASE64 image utilities"""
import base64
import io
from PIL import Image
from typing import List, Tuple

def b64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))

def pil_to_b64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def prepare_images(images: List[str], max_size: int = 4) -> List[Image.Image]:
    """Prepare images from base64, resize if needed"""
    pil_images = []
    
    for i, b64 in enumerate(images[:max_size]):
        img = b64_to_pil(b64)
        # Convert RGBA to RGB if needed
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)
    
    return pil_images

def get_image_dimensions(images: List[str]) -> Tuple[int, int]:
    """Get dimensions of first image for latent generation"""
    if not images:
        return (1024, 1024)
    img = b64_to_pil(images[0])
    return img.size
