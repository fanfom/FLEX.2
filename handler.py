#!/usr/bin/env python3
"""
RunPod Serverless Handler for ComfyUI

Tavern отправляет workflow JSON → мы выполняем → возвращаем base64 изображения
"""

import os
import sys
import json
import uuid
import asyncio
import shutil
import subprocess
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

# ============================================================
# Конфигурация
# ============================================================

MODELS_BASE = os.environ.get("MODELS_BASE", "/runpod-volume")
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
TEMP_DIR = Path("/tmp/comfy_runs")

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# Добавляем ComfyUI в path для импортов
sys.path.insert(0, COMFYUI_PATH)


# ============================================================
# Base64 утилиты
# ============================================================

def decode_base64_image(b64_string: str) -> bytes:
    """Декодировать base64 в bytes. Поддерживает data: URL."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return base64.b64decode(b64_string)


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Закодировать bytes в base64 строку"""
    return base64.b64encode(image_bytes).decode("utf-8")


def save_base64_image(b64_string: str, output_dir: Path, prefix: str = "input") -> str:
    """Сохранить base64 как файл, вернуть путь"""
    image_bytes = decode_base64_image(b64_string)
    
    # Определяем расширение по сигнатуре
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = "png"
    elif image_bytes[:2] == b"\xff\xd8":
        ext = "jpg"
    else:
        ext = "png"
    
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = output_dir / filename
    filepath.write_bytes(image_bytes)
    
    return str(filepath)


def load_image_as_base64(filepath: Path) -> str:
    """Прочитать файл и вернуть base64"""
    return encode_image_to_base64(filepath.read_bytes())


# ============================================================
# ComfyUI клиент
# ============================================================

class ComfyUIClient:
    """Асинхронный клиент для ComfyUI API"""
    
    def __init__(self):
        self.client_id = str(uuid.uuid4())
    
    async def _post(self, endpoint: str, data: Any = None, json_data: Dict = None) -> Dict:
        """POST запрос к ComfyUI"""
        import aiohttp
        
        url = f"{COMFY_URL}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if json_data:
                async with session.post(url, json=json_data) as resp:
                    return await resp.json()
            async with session.post(url, data=data) as resp:
                return await resp.json()
    
    async def _get(self, endpoint: str) -> Any:
        """GET запрос к ComfyUI"""
        import aiohttp
        
        url = f"{COMFY_URL}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.content_type == "application/json":
                    return await resp.json()
                return await resp.read()
    
# 1. В queue_prompt добавь проверку, чтобы не было KeyError
    async def queue_prompt(self, workflow: Dict) -> str:
        prompt_data = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        url = f"{COMFY_URL}/prompt"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=prompt_data) as resp:
                result = await resp.json()
                if "prompt_id" not in result:
                    # Если ComfyUI ругается, мы увидим ЧТО именно не так
                    raise RuntimeError(f"ComfyUI Error: {result}")
                return result["prompt_id"]

    
    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict:
        """Ждать завершения выполнения"""
        import aiohttp
        
        for _ in range(timeout):
            await asyncio.sleep(1)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
                    history = await resp.json()
                    
                    if prompt_id in history:
                        return history[prompt_id]
        
        raise TimeoutError(f"Workflow timeout after {timeout}s")
    
    async def execute_workflow(self, workflow: Dict) -> Dict:
        """Выполнить workflow и дождаться результата"""
        prompt_id = await self.queue_prompt(workflow)
        return await self.wait_for_completion(prompt_id)


# ============================================================
# Подготовка workflow
# ============================================================


# 2. В prepare_workflow УДАЛИ или закомментируй блок замены путей для моделей
def prepare_workflow(workflow: Dict, run_dir: Path) -> Dict:
    prepared = {}
    for node_id, node in workflow.items():
        node_class = node.get("class_type", "")
        inputs = node.get("inputs", {}).copy()
        
        # Оставляем только обработку картинок (LoadImage)
        if node_class == "LoadImage" and "image" in inputs:
            image_value = inputs["image"]
            if isinstance(image_value, str) and len(image_value) > 50:
                saved_path = save_base64_image(image_value, run_dir / "input", f"node_{node_id}")
                inputs["image"] = saved_path
        
        # БЛОК С CheckpointLoader / VAELoader / CLIPLoader НУЖНО УДАЛИТЬ. 
        # Пусть имена остаются просто именами (напр. "flux1.safetensors")
        
        prepared[node_id] = {
            "class_type": node_class,
            "inputs": inputs
        }
    return prepared



def extract_images_from_result(result: Dict) -> List[str]:
    """Извлечь изображения из результата выполнения"""
    images = []
    
    outputs = result.get("outputs", {})
    
    for node_id, node_data in outputs.items():
        # Ищем изображения в выходных данных
        if isinstance(node_data, dict):
            # PreviewImage, SaveImage, etc.
            if "images" in node_data:
                for img in node_data["images"]:
                    filepath = Path(COMFYUI_PATH) / "output"
                    if img.get("subfolder"):
                        filepath = filepath / img["subfolder"]
                    filepath = filepath / img["filename"]
                    
                    if filepath.exists():
                        images.append(load_image_as_base64(filepath))
                        print(f"[OUTPUT] {img['filename']} ({len(images)} images total)")
            
            # alternativa: latent samples
            elif "gifs" in node_data:
                for gif in node_data["gifs"]:
                    filepath = Path(COMFYUI_PATH) / "output"
                    if gif.get("subfolder"):
                        filepath = filepath / gif["subfolder"]
                    filepath = filepath / gif["filename"]
                    
                    if filepath.exists():
                        images.append(load_image_as_base64(filepath))
    
    return images


# ============================================================
# ComfyUI сервер
# ============================================================

_comfy_process: Optional[subprocess.Popen] = None


async def start_comfyui_server() -> bool:
    """Запустить ComfyUI сервер в фоне"""
    global _comfy_process
    
    import aiohttp
    
    # Проверяем, не запущен ли уже
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFY_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    print("[COMFY] Server already running")
                    return True
    except:
        pass
    
    # Запускаем
    print("[COMFY] Starting ComfyUI server...")
    
    comfy_main = os.path.join(COMFYUI_PATH, "main.py")
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments=True"
    
    _comfy_process = subprocess.Popen(
        [
            sys.executable,
            comfy_main,
            "--listen", COMFY_HOST,
            "--port", str(COMFY_PORT),
            "--enable-cors",
            "--disable-auto-launch",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Ждём запуска (до 60 секунд)
    for i in range(60):
        await asyncio.sleep(1)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{COMFY_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        print(f"[COMFY] Server ready after {i+1}s")
                        return True
        except:
            continue
    
    raise RuntimeError("ComfyUI failed to start")


async def stop_comfyui_server():
    """Остановить ComfyUI сервер"""
    global _comfy_process
    
    if _comfy_process:
        _comfy_process.terminate()
        _comfy_process.wait(timeout=10)
        _comfy_process = None


# ============================================================
# Главный обработчик
# ============================================================

async def handler(event) -> Dict[str, Any]:
    """
    RunPod serverless handler
    
    Input:
        {
            "workflow": { ... }   // Workflow JSON от Tavern
        }
    
    Output:
        {
            "images": ["base64...", ...],
            "success": true
        }
    """
    run_id = None
    
    try:
        body = event.get("input", {})
        workflow = body.get("workflow")
        
        if not workflow:
            return {"error": "No workflow provided", "success": False}
        
        print(f"[HANDLER] Workflow received: {len(workflow)} nodes")
        
        # Создаём временную директорию для этого запуска
        run_id = uuid.uuid4().hex[:8]
        run_dir = TEMP_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "input").mkdir(exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)
        
        # Убеждаемся что ComfyUI запущен
        await start_comfyui_server()
        
        # Подготавливаем workflow
        prepared = prepare_workflow(workflow, run_dir)
        
        # Выполняем
        client = ComfyUIClient()
        print("[HANDLER] Queuing prompt...")
        result = await client.execute_workflow(prepared)
        
        # Извлекаем результат
        images = extract_images_from_result(result)
        
        print(f"[HANDLER] Done: {len(images)} images generated")
        
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
        
    finally:
        # Очищаем временные файлы
        if run_id:
            run_dir = TEMP_DIR / run_id
            shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# RunPod entry point
# ============================================================

def main():
    import runpod
    
    # Создаём temp директорию
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    runpod.serverless.start({
        "handler": handler,
        "return_ray": False,
    })


if __name__ == "__main__":
    main()
