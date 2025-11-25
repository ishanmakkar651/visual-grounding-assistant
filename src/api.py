
import os
import io
import uuid
import shutil
import torch
import json
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
UPLOAD_DIR = "src/storage"
BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "visual-grounding-adapter"

app = FastAPI(title="VisionEngine Ultimate")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- LOAD MODEL ---
print("⏳ Booting AI Engine...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
    )
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("✅ Adapter Loaded")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, min_pixels=256*28*28, max_pixels=1280*28*28)
except Exception as e:
    print(f"❌ Error: {e}")

# --- HELPER: COORDINATES ---
def normalize_bbox(x1, y1, x2, y2, width, height):
    nx1 = int((x1 / width) * 1000)
    ny1 = int((y1 / height) * 1000)
    nx2 = int((x2 / width) * 1000)
    ny2 = int((y2 / height) * 1000)
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(1000, nx2), min(1000, ny2)
    return f"<|box_start|>({ny1},{nx1}),({ny2},{nx2})<|box_end|>"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = []
    if file.filename.lower().endswith('.pdf'):
        try:
            images = convert_from_bytes(open(file_path, 'rb').read())
            for i, img in enumerate(images):
                img_name = f"page_{i}.jpg"
                img.save(os.path.join(session_dir, img_name), "JPEG")
                pages.append(f"/storage/{session_id}/{img_name}")
        except: return JSONResponse(status_code=400, content={"error": "PDF Error"})
    else:
        pages.append(f"/storage/{session_id}/{file.filename}")

    return {"session_id": session_id, "pages": pages, "filename": file.filename}

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    image_url: str = Form(...),
    message: str = Form(...),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None)
):
    try:
        clean_path = image_url.replace("/storage/", "")
        local_img_path = os.path.join(UPLOAD_DIR, clean_path)
        img = Image.open(local_img_path).convert("RGB")
        w, h = img.size

        bbox_str = ""
        system_context = "You are an expert technical engineer."

        if x1 is not None and x2 is not None:
            rx1, rx2 = min(x1, x2), max(x1, x2)
            ry1, ry2 = min(y1, y2), max(y1, y2)
            # Pad small clicks
            if (rx2 - rx1) < 10 and (ry2 - ry1) < 10:
                pad = 50
                rx1, ry1 = max(0, rx1 - pad), max(0, ry1 - pad)
                rx2, ry2 = min(w, rx2 + pad), min(h, ry2 + pad)

            bbox_str = normalize_bbox(rx1, ry1, rx2, ry2, w, h)
            system_context += f" Focus ONLY on the region {bbox_str}."

        # Detect CSV Request
        if "csv" in message.lower() or "table" in message.lower():
            system_context += " Extract the data visible in the region as a CSV string. Do not include markdown code blocks."

        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": f"{system_context} {message}"}]}]

        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return {"role": "assistant", "content": answer, "bbox": bbox_str}
    except Exception as e:
        return {"error": str(e)}

app.mount("/storage", StaticFiles(directory=UPLOAD_DIR), name="storage")
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
def home(): return FileResponse('src/static/index.html')
