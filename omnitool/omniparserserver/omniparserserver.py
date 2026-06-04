'''
python -m omniparserserver --som_model_path ../../weights/ui_detr/model.pth --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cpu --BOX_TRESHOLD 0.3
'''

import sys
import os
import io
import re
import time
import base64
from typing import Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import argparse
import uvicorn
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/ui_detr/model.pth', help='Path to the UI-DETR model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    parser.add_argument('--enable_grounding', action='store_true', help='Enable UI-TARS grounding endpoint')
    parser.add_argument('--grounding_model', type=str, default='ByteDance-Seed/UI-TARS-1.5-7B', help='HuggingFace model ID for grounding (UI-TARS or Qwen2.5-VL-Instruct variant)')
    parser.add_argument('--api_key', type=str, default=os.environ.get('OMNIPARSER_API_KEY', ''), help='API key for authentication (or set OMNIPARSER_API_KEY env var)')
    args, _ = parser.parse_known_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()


@app.on_event("startup")
async def warmup_grounding():
    """Eagerly load the grounding model so the first /ground/ request doesn't
    pay the ~60s model-load tax on a cold pod (and time out client-side)."""
    if args.enable_grounding:
        get_grounding_model()


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if args.api_key and request.url.path != "/probe/":
        key = request.headers.get("X-API-Key", "")
        if key != args.api_key:
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
    return await call_next(request)


# OmniParser (lazy-loaded, may not have weights available)
_omniparser = None


def get_omniparser():
    global _omniparser
    if _omniparser is None:
        _omniparser = Omniparser(config)
    return _omniparser


# UI-TARS grounding model (lazy-loaded)
_grounding_model = None
_grounding_processor = None


def get_grounding_model():
    global _grounding_model, _grounding_processor
    if _grounding_model is None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        dtype = torch.bfloat16 if device != 'cpu' else torch.float32
        print(f'Loading grounding model {args.grounding_model} on {device}...')

        _grounding_model = AutoModelForImageTextToText.from_pretrained(
            args.grounding_model,
            torch_dtype=dtype,
            attn_implementation='sdpa',
            device_map=device,
        ).eval()
        _grounding_model.config.use_cache = True
        if hasattr(_grounding_model.config, 'text_config'):
            _grounding_model.config.text_config.use_cache = True
        _grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
        print('Grounding model loaded!')
    return _grounding_model, _grounding_processor


# Official UI-TARS GROUNDING prompt template
# https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py
GROUNDING_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{instruction}"""

# Parse UI-TARS output. Returns dict with `point` and optional `bbox`, or None.
# Native GROUNDING format from the official template: <point>x y</point>
_POINT_TAG = re.compile(r"<point>\s*(\d+)\s+(\d+)\s*</point>", re.IGNORECASE)
# COMPUTER_USE / agent-style with start_box / end_box bracketed coords
_BBOX_ACTION = re.compile(r"start_box\s*=\s*['\"][\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")
_POINT_ACTION = re.compile(r"start_box\s*=\s*['\"][\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")
# Alternative SVG-ish <points x1="..." y1="..."> the model occasionally emits
_POINTS_TAG = re.compile(
    r'<points\b[^>]*?\bx1\s*=\s*"(\d+)"[^>]*?\by1\s*=\s*"(\d+)"'
    r'(?:[^>]*?\bx2\s*=\s*"(\d+)")?(?:[^>]*?\by2\s*=\s*"(\d+)")?',
    re.IGNORECASE,
)
# Bare fallbacks (no system prompt → model improvises)
_BBOX_BARE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_POINT_BARE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_uitars_output(text):
    m = _POINT_TAG.search(text)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return {"point": (float(x), float(y)), "bbox": None}
    m = _BBOX_ACTION.search(text)
    if m:
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        return {"point": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "bbox": (x1, y1, x2, y2)}
    m = _POINT_ACTION.search(text)
    if m:
        x, y = (int(v) for v in m.groups())
        return {"point": (float(x), float(y)), "bbox": None}
    m = _POINTS_TAG.search(text)
    if m:
        x1, y1 = int(m.group(1)), int(m.group(2))
        if m.group(3) and m.group(4):
            x2, y2 = int(m.group(3)), int(m.group(4))
            return {"point": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "bbox": (x1, y1, x2, y2)}
        return {"point": (float(x1), float(y1)), "bbox": None}
    m = _BBOX_BARE.search(text)
    if m:
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        return {"point": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "bbox": (x1, y1, x2, y2)}
    m = _POINT_BARE.search(text)
    if m:
        x, y = (int(v) for v in m.groups())
        return {"point": (float(x), float(y)), "bbox": None}
    return None


class ParseRequest(BaseModel):
    base64_image: str
    paragraph_enabled: bool = True  # Default to True to maintain backward compatibility

class GroundRequest(BaseModel):
    base64_image: str
    instruction: str
    topk: Optional[int] = 1


@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    omniparser = get_omniparser()
    dino_labled_img, parsed_content_list, image_size = omniparser.parse(parse_request.base64_image, paragraph_enabled=parse_request.paragraph_enabled)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency, 'image_size': image_size}


@app.post("/ground/")
async def ground(request: GroundRequest):
    """Find a UI element by description using UI-TARS visual grounding.

    Returns click coordinates in both pixel and normalized [0,1] space.
    For topk>1, the model is sampled with mild temperature; for topk=1
    decoding is deterministic (greedy).
    """
    if not args.enable_grounding:
        return {"error": "Grounding not enabled. Start server with --enable_grounding"}

    start = time.time()
    model, processor = get_grounding_model()
    topk = max(1, request.topk or 1)

    image_bytes = base64.b64decode(request.base64_image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": GROUNDING_PROMPT.format(instruction=request.instruction)},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 256,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if topk == 1:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["num_return_sequences"] = topk

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    prefix_len = inputs["input_ids"].shape[1]
    points_px = []
    raw_outputs = []
    for seq in out:
        text = processor.decode(seq[prefix_len:], skip_special_tokens=True)
        raw_outputs.append(text)
        parsed = parse_uitars_output(text)
        if parsed is None:
            continue
        px, py = parsed["point"]
        # Dedup near-identical candidates (within 8 px)
        if any(abs(p["x"] - px) < 8 and abs(p["y"] - py) < 8 for p in points_px):
            continue
        points_px.append({
            "x": px,
            "y": py,
            "x_norm": px / w if w else 0.0,
            "y_norm": py / h if h else 0.0,
            "bbox": list(parsed["bbox"]) if parsed["bbox"] is not None else None,
        })

    latency = time.time() - start
    result = {
        "points": points_px,
        "topk_values": [1.0] * len(points_px),  # UI-TARS does not expose calibrated confidence
        "image_size": [w, h],
        "latency": latency,
        "verified": False,
        "raw_outputs": raw_outputs,
    }
    print(f'ground: "{request.instruction}" -> {len(points_px)} points, {latency:.2f}s')
    return result


@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
