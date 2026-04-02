#!/usr/bin/env python3
"""
RunPod Serverless Handler for ComfyUI
Tavern отправляет workflow JSON — мы его исполняем и возвращаем base64 картинки
"""

import os
import sys
import json
import uuid
import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Конфиг
MODELS_BASE = os.environ.get("MODELS_BASE", "/runpod-volume")
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
TEMP_DIR = "/tmp/comfy_workflows"

# sys.path для импорта ComfyUI модулей
sys.path.insert(0, COMFYUI_PATH)


# ============================================================
# Base64 <-> Image утилиты
# ============================================================

def b64_to_file(b64_string: str, output_path: Path) -> str:
    """Сохранить base64 как файл, вернуть путь"""
    import base64
    
    # Убираем data:image/...;base64, если есть
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    
    image_data = base64.b64decode(b64_string)
    
    # Определяем расширение
    ext = "png"
    if b64_string[:4] in ("/9j/", "iVB0"):
        ext = "png"
    elif b64_string[:4] in ("/8j/", "AAAB"):
        ext = "jpeg"
    
    full_path = output_path.with_suffix(f".{ext}")
    full_path.write_bytes(image_data)
    return str(full_path)


def file_to_b64(file_path: Path) -> str:
    """Прочитать файл и вернуть base64"""
    import base64
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def pil_to_b64(image, format: str = "PNG") -> str:
    """PIL Image -> base64"""
    import base64
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# ComfyUI API клиент
# ============================================================

class ComfyUIClient:
    """Простой клиент для ComfyUI API"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self.base_url = f"http://{host}:{port}"
    
    async def queue_prompt(self, workflow: Dict) -> str:
        """Поставить workflow в очередь"""
        import aiohttp
        
        prompt_request = {
            "prompt": workflow,
            "client_id": self.client_id,
            "extra_data": {}
        }
        
        async with aiohttp.ClientSession() as session:
