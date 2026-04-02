import os
import uuid
import asyncio
import aiohttp
import runpod
from typing import Dict, Any, List

COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
COMFY_URL = "http://127.0.0.1:8188"

# 1. Оставляем только базовый клиент для связи с ComfyUI
class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())

    async def execute_workflow(self, workflow: Dict) -> Dict:
        # Отправляем промпт
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": workflow, "client_id": self.client_id}
            async with session.post(f"{COMFY_URL}/prompt", json=payload) as resp:
                result = await resp.json()
                if "prompt_id" not in result:
                    raise RuntimeError(f"ComfyUI Error: {result}")
                prompt_id = result["prompt_id"]

            # Ждем завершения (опрашиваем историю)
            while True:
                async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
                    history = await resp.json()
                    if prompt_id in history:
                        return history[prompt_id]
                await asyncio.sleep(1)

# 2. Ультра-простой экстрактор картинок (берет готовый Base64 из ноды)
def extract_images(result: Dict) -> List[str]:
    images = []
    outputs = result.get("outputs", {})
    for node_id, node_data in outputs.items():
        # Нода SaveImage64 кладет готовые строки прямо сюда
        if "images" in node_data:
            for item in node_data["images"]:
                # Если это строка (Base64), просто берем её
                if isinstance(item, str):
                    images.append(item)
                # Если это словарь (зависит от версии ноды), ищем ключ с данными
                elif isinstance(item, dict) and "base64" in item:
                    images.append(item["base64"])
    return images

async def handler(event: Dict) -> Dict:
    try:
        # Берем флоу из запроса
        workflow = event["input"].get("workflow") or event["input"].get("prompt")
        if not workflow:
            return {"error": "No workflow provided", "success": False}

        client = ComfyUIClient()
        
        # Запускаем генерацию
        result = await client.execute_workflow(workflow)
        
        # Достаем картинки (они уже в Base64)
        images = extract_images(result)
        
        return {
            "images": images,
            "success": True,
            "count": len(images)
        }
    except Exception as e:
        return {"error": str(e), "success": False}

# Запуск
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
