"""Tracely demo API on Modal: POST a photo and a task, then poll GET /result?id=... for the path.

  modal deploy demo/modal_demo.py          # prints the endpoint URL; put it in docs/index.html

The model (Qwen3-VL-8B + the Tracely adapter from the Hub) loads on the first request and stays warm
for a few minutes afterwards. Requests are counted per day in a modal.Dict and refused above DAILY_LIMIT,
so a public page cannot run up the bill.
"""

import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import modal

ADAPTER = "ahd03/tracely"
BASE = "Qwen/Qwen3-VL-8B-Instruct"
MAX_SIDE = 896
DAILY_LIMIT = 100
ALLOWED_ORIGINS = ["https://ahedmousa1-crypto.github.io", "http://localhost:8000", "http://127.0.0.1:8000"]

app = modal.App("tracely-demo")
volume = modal.Volume.from_name("vln-finetune")       # the base model weights are cached here (HF_HOME)
usage = modal.Dict.from_name("tracely-usage", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1", "accelerate==1.10.1",
                 "peft==0.17.1", "pillow", "fastapi[standard]", "huggingface_hub>=0.35")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_file(Path(__file__).parent.parent / "finetune" / "prompts.py", "/root/prompts.py")
)


def _resize(img):
    img = img.convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


def _parse(text):
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    pts = [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]
    return [[min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)] for x, y in pts]


@app.cls(image=image, gpu=["L4", "A10G", "L40S"], volumes={"/vol": volume}, scaledown_window=300, timeout=900)
class Tracely:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(BASE)
        model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(model, ADAPTER).eval()
        self.processor.tokenizer.padding_side = "left"

    @modal.method()
    def predict(self, image_bytes: bytes, task: str):
        import torch
        from PIL import Image

        sys.path.insert(0, "/root")
        from prompts import messages

        img = _resize(Image.open(io.BytesIO(image_bytes)))
        text = self.processor.apply_chat_template(messages(img, task), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"points": _parse(reply), "raw": reply}


@app.function(image=image, timeout=600)
@modal.asgi_app()
def trace():
    """POST / with {"image": "<base64 jpeg/png>", "task": "Go to the bench on the left"}.
    Returns {"points": [[x, y], ...]} with x, y in 0-1 (origin top-left), or {"error": "..."}."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
    api.post("/")(_trace)
    api.get("/result")(_result)
    return api


def _result(id: str):
    """Poll a started request: 202 while the GPU is still working, else the result."""
    from fastapi.responses import JSONResponse

    try:
        call = modal.FunctionCall.from_id(id)
        result = call.get(timeout=0)
    except TimeoutError:
        return JSONResponse({"status": "working"}, status_code=202)
    except Exception as exc:  # expired id, failed call
        return JSONResponse({"error": f"The request failed: {type(exc).__name__}"}, status_code=500)
    result["remaining_today"] = DAILY_LIMIT - usage.get(time.strftime("%Y-%m-%d"), 0)
    return JSONResponse(result)


def _trace(request: dict):
    from fastapi.responses import JSONResponse

    day = time.strftime("%Y-%m-%d")
    used = usage.get(day, 0)
    if used >= DAILY_LIMIT:
        return JSONResponse({"error": "The demo has reached its daily limit. Please try again tomorrow."}, status_code=429)
    task = str(request.get("task", "")).strip()[:300]
    data = request.get("image", "")
    if not task or not data:
        return JSONResponse({"error": "Send an image and a task."}, status_code=400)
    try:
        image_bytes = base64.b64decode(data.split(",", 1)[-1])
        if len(image_bytes) > 8_000_000:
            return JSONResponse({"error": "Image too large (max 8 MB)."}, status_code=400)
    except Exception:
        return JSONResponse({"error": "Could not read the image."}, status_code=400)
    usage[day] = used + 1
    call = Tracely().predict.spawn(image_bytes, task)   # the page polls /result?id=... (a cold GPU takes minutes)
    return JSONResponse({"id": call.object_id, "remaining_today": DAILY_LIMIT - used - 1}, status_code=202)
