import os
import uuid
import asyncio
import aiohttp
import subprocess
import sys
import runpod
from typing import Dict, Any, List
import base64
import tempfile

# --- Глобальный запуск ComfyUI (один раз при старте воркера) ---
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
COMFY_URL = "http://127.0.0.1:8188"
_comfy_process = None

def start_comfyui_sync():
    """Синхронный запуск сервера (вызывается при импорте)"""
    global _comfy_process
    if _comfy_process is not None:
        return
    print("[COMFY] Starting server...")
    _comfy_process = subprocess.Popen([
        sys.executable, f"{COMFYUI_PATH}/main.py",
        "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch"
    ])
    # Ждём готовности
    import time, requests
    for _ in range(60):
        try:
            requests.get(f"{COMFY_URL}/system_stats", timeout=1)
            print("[COMFY] Ready")
            return
        except:
            time.sleep(1)
    raise RuntimeError("ComfyUI failed to start")

# Запускаем сервер при загрузке модуля
start_comfyui_sync()

# --- Вспомогательные функции ---
def save_base64_to_temp_file(base64_string):
    """Сохраняет base64 во временный файл и возвращает путь к нему"""
    # Удаляем префикс, если есть
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_data = base64.b64decode(base64_string)
    # Создаём уникальное имя, чтобы избежать гонок
    temp_dir = f"{COMFYUI_PATH}/input"
    os.makedirs(temp_dir, exist_ok=True)
    unique_name = f"input_{uuid.uuid4().hex}.png"
    file_path = os.path.join(temp_dir, unique_name)
    with open(file_path, "wb") as f:
        f.write(image_data)
    return file_path

def cleanup_temp_file(file_path):
    """Удаляет временный файл после использования"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"[WARN] Failed to delete {file_path}: {e}")

# --- Клиент ComfyUI ---
class ComfyUIClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())

    async def execute_workflow(self, workflow: Dict, base64_image: str) -> Dict:
        # 1. Сохраняем base64 во временный файл
        temp_file = save_base64_to_temp_file(base64_image)
        try:
            # 2. Модифицируем workflow: подставляем имя файла в ноду LoadImage (ID 35)
            #    Если в вашем workflow нода LoadImage имеет другой ID, укажите правильный
            if '35' in workflow:
                workflow['35']['inputs']['image'] = os.path.basename(temp_file)
            else:
                raise RuntimeError("Workflow must contain node #35 (LoadImage)")

            # 3. Отправляем запрос в ComfyUI
            async with aiohttp.ClientSession() as session:
                payload = {"prompt": workflow, "client_id": self.client_id}
                async with session.post(f"{COMFY_URL}/prompt", json=payload) as resp:
                    result = await resp.json()
                    if "prompt_id" not in result:
                        raise RuntimeError(f"ComfyUI Error: {result}")
                    prompt_id = result["prompt_id"]

                # 4. Ждём результат
                while True:
                    async with session.get(f"{COMFY_URL}/history/{prompt_id}") as resp:
                        history = await resp.json()
                        if prompt_id in history:
                            return history[prompt_id]
                    await asyncio.sleep(1)
        finally:
            # 5. Удаляем временный файл (даже если произошла ошибка)
            cleanup_temp_file(temp_file)

# --- Извлечение base64 из результата (нода SaveImage64) ---
def extract_images(result: Dict) -> List[str]:
    images = []
    outputs = result.get("outputs", {})
    for node_id, node_data in outputs.items():
        # Если нода SaveImage64 возвращает список [ [файлы], [base64_строки] ]
        if isinstance(node_data, list) and len(node_data) > 1:
            base64_list = node_data[1]
            if isinstance(base64_list, list):
                for item in base64_list:
                    if isinstance(item, str) and len(item) > 100:
                        images.append(item)
    return images

# --- Главный хендлер RunPod ---
async def handler(event: Dict) -> Dict:
    try:
        # Получаем workflow и base64 из входных данных
        workflow = event["input"].get("workflow") or event["input"].get("prompt")
        base64_image = event["input"].get("base64_image")  # ожидаем поле base64_image
        
        if not workflow:
            return {"error": "Missing workflow", "success": False}
        if not base64_image:
            return {"error": "Missing base64_image", "success": False}
        
        client = ComfyUIClient()
        result = await client.execute_workflow(workflow, base64_image)
        images = extract_images(result)
        
        return {"images": images, "success": True, "count": len(images)}
    except Exception as e:
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
