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
            # Queue
            async with session.post(
                f"{self.base_url}/prompt",
                json=prompt_request
            ) as resp:
                result = await resp.json()
                prompt_id = result["prompt_id"]
            
            # Wait for completion
            status = "queued"
            while status == "queued" or status == "executing":
                await asyncio.sleep(1)
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}"
                ) as resp:
                    history = await resp.json()
                    if prompt_id in history:
                        status = "done"
                    else:
                        status = "executing"
            
            return history[prompt_id]
    
    def get_images(self, outputs: Dict) -> List[Dict]:
        """Извлечь изображения из результата"""
        images = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append({
                        "node_id": node_id,
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output")
                    })
        
        return images


# ============================================================
# Подготовка workflow
# ============================================================

def prepare_workflow(workflow: Dict, input_dir: Path) -> Dict:
    """
    Подготовить workflow к исполнению:
    - Подменить пути к моделям на /runpod-volume/models/...
    - Обработать base64 изображения
    """
    import re
    
    prepared = {}
    
    for node_id, node in workflow.items():
        node_type = node.get("class_type", "")
        inputs = node.get("inputs", {}).copy()
        
        # LoadImage nodes — заменить base64 на путь к файлу
        if node_type == "LoadImage" and "image" in inputs:
            image_b64 = inputs["image"]
            if isinstance(image_b64, str) and len(image_b64) > 100:
                # Это base64, конвертируем в файл
                saved_path = b64_to_file(image_b64, input_dir / f"input_{node_id}")
                inputs["image"] = saved_path
                print(f"[PREP] LoadImage {node_id}: {saved_path}")
        
        # Пути к моделям — подменить на абсолютные
        if node_type in ("CheckpointLoaderSimple", "VAELoader", "CLIPLoader"):
            for key in inputs:
                if key.endswith("_name") or key.endswith("_path"):
                    filename = inputs[key]
                    if not filename.startswith("/"):
                        # Относительный путь — добавляем /runpod-volume/models/
                        full_path = os.path.join(MODELS_BASE, "models", filename)
                        inputs[key] = full_path
                        print(f"[PREP] {node_type} {node_id}: {filename} -> {full_path}")
        
        # CLIPTextEncode, KSampler и др. — оставляем как есть
        prepared[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
    
    return prepared


# ============================================================
# Подготовка output изображений
# ============================================================

def extract_output_images(
    result: Dict,
    client: ComfyUIClient,
    output_dir: Path
) -> List[str]:
    """Извлечь результаты и конвертировать в base64"""
    images_b64 = []
    
    # Ищем ноды с изображениями на выходе
    outputs = result.get("outputs", {})
    
    for node_id, node_data in outputs.items():
        # PreviewImage, SaveImage, etc.
        if "images" in node_data:
            for img in node_data["images"]:
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                
                # Путь к файлу
                if subfolder:
                    img_path = Path(COMFYUI_PATH) / "output" / subfolder / filename
                else:
                    img_path = Path(COMFYUI_PATH) / "output" / filename
                
                # Читаем и конвертируем в base64
                if img_path.exists():
                    b64 = file_to_b64(img_path)
                    images_b64.append(b64)
                    print(f"[OUTPUT] {filename} -> {len(b64)} bytes base64")
    
    return images_b64


# ============================================================
# Главный обработчик
# ============================================================

async def handler(event) -> Dict[str, Any]:
    """
    Точка входа для RunPod serverless
    
    Input:
    {
        "workflow": { ... },  // Workflow JSON от Tavern
        "images": ["base64...", ...]  // Опционально входные изображения
    }
    
    Output:
    {
        "images": ["base64...", ...],  // Результат
        "success": true
    }
    """
    try:
        body = event.get("input", {})
        workflow = body.get("workflow")
        
        if not workflow:
            return {"error": "No workflow provided", "success": False}
        
        print(f"[HANDLER] Starting workflow execution")
        print(f"[HANDLER] Nodes: {len(workflow)}")
        
        # Создаём временную директорию
        run_id = str(uuid.uuid4())[:8]
        run_dir = Path(TEMP_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        input_dir = run_dir / "input"
        input_dir.mkdir(exist_ok=True)
        
        # Подготовка workflow
        prepared_workflow = prepare_workflow(workflow, input_dir)
        
        # Запуск ComfyUI сервера если не запущен
        comfy_server = await ensure_comfyui_running()
        
        # Выполнение
        client = ComfyUIClient()
        result = await client.queue_prompt(prepared_workflow)
        
        # Извлечение результатов
        images = extract_output_images(result, client, run_dir / "output")
        
        # Очистка
        shutil.rmtree(run_dir, ignore_errors=True)
        
        print(f"[HANDLER] Done: {len(images)} images")
        
        return {
            "images": images,
            "success": True,
            "count": len(images)
        }
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e), "success": False}


async def ensure_comfyui_running():
    """Проверить что ComfyUI сервер запущен, если нет — запустить"""
    import aiohttp
    import subprocess
    import time
    
    url = "http://127.0.0.1:8188/system_stats"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    print("[COMFY] Already running")
                    return True
    except:
        pass
    
    # Запускаем ComfyUI в фоне
    print("[COMFY] Starting ComfyUI server...")
    
    cmd = [
        sys.executable,
        os.path.join(COMFYUI_PATH, "main.py"),
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--enable-cors",
        "--disable-auto-launch",
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments=True"}
    )
    
    # Ждём запуска
    for _ in range(30):
        await asyncio.sleep(1)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        print("[COMFY] Server ready")
                        return True
        except:
            continue
    
    raise RuntimeError("ComfyUI failed to start")


# ============================================================
# RunPod entry point
# ============================================================

def main():
    import runpod
    
    # Создаём temp директорию
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    runpod.serverless.start({
        "handler": handler,
        "return_ray": False,
    })


if __name__ == "__main__":
    main()
