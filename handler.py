import os
import uuid
import asyncio
import aiohttp
import subprocess
import sys
import runpod
from typing import Dict, Any, List

COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
COMFY_URL = "http://127.0.0.1:8188"
_server_started = False

# 1. Запуск сервера (без него будет 'Cannot connect')
async def start_comfyui_server():
    global _server_started
    if _server_started: return
    
    print("[COMFY] Starting server...")
    subprocess.Popen([
        sys.executable, f"{COMFYUI_PATH}/main.py",
        "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch"
    ])
    
    # Ждем пока порт откроется
    for i in range(60):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{COMFY_URL}/system_stats", timeout=1) as resp:
                    if resp.status == 200:
                        print(f"[COMFY] Ready after {i}s")
                        _server_started = True
                        return
        except:
            await asyncio.sleep(1)
    raise RuntimeError("ComfyUI failed to start")

# 2. Клиент для работы с очередью
class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())

    async def execute_workflow(self, workflow: Dict) -> Dict:
        async with aiohttp.ClientSession() as session:
            # Отправка флоу
            payload = {"prompt": workflow, "client_id": self.client_id}
            async with session.post(f"{COMFY_URL}/prompt", json=payload) as resp:
                result = await resp.json()
                if "prompt_id" not in result:
                    raise RuntimeError(f"ComfyUI Error: {result}")
                prompt_id = result["prompt_id"]

            # Ожидание результата
            while True:
                async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
                    history = await resp.json()
                    if prompt_id in history:
                        return history[prompt_id]
                await asyncio.sleep(1)

# 3. Экстрактор (для ноды SaveImage64)
def extract_images(result: Dict) -> List[str]:
    images = []
    outputs = result.get("outputs", {})
    for node_id, node_data in outputs.items():
        if "images" in node_data:
            for item in node_data["images"]:
                # SaveImage64 отдает сразу готовую строку
                if isinstance(item, str): images.append(item)
                elif isinstance(item, dict) and "base64" in item: images.append(item["base64"])
    return images

# Главный хендлер RunPod
async def handler(event: Dict) -> Dict:
    try:
        await start_comfyui_server()
        
        # Поддерживаем оба варианта ключа
        workflow = event["input"].get("workflow") or event["input"].get("prompt")
        if not workflow: return {"error": "No workflow", "success": False}

        client = ComfyUIClient()
        result = await client.execute_workflow(workflow)
        images = extract_images(result)
        
        return {"images": images, "success": True, "count": len(images)}
    except Exception as e:
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
