'''
python -m omniparserserver --som_model_path ../../weights/ui_detr/model.pth --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cpu --BOX_TRESHOLD 0.3
'''

import sys
import os
import io
import time
import base64
from typing import Optional

import torch
from fastapi import FastAPI
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
    parser.add_argument('--enable_grounding', action='store_true', help='Enable GUI-Actor grounding endpoint')
    parser.add_argument('--grounding_model', type=str, default='microsoft/GUI-Actor-3B-Qwen2.5-VL', help='HuggingFace model ID for grounding')
    args, _ = parser.parse_known_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()

# OmniParser (lazy-loaded, may not have weights available)
_omniparser = None


def get_omniparser():
    global _omniparser
    if _omniparser is None:
        _omniparser = Omniparser(config)
    return _omniparser


# GUI-Actor grounding model (lazy-loaded)
_grounding_model = None
_grounding_processor = None
_grounding_tokenizer = None


def get_grounding_model():
    global _grounding_model, _grounding_processor, _grounding_tokenizer
    if _grounding_model is None:
        from transformers import AutoProcessor
        from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        dtype = torch.bfloat16 if device != 'cpu' else torch.float32
        print(f'Loading grounding model {args.grounding_model} on {device}...')

        _grounding_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            args.grounding_model,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        _grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
        _grounding_tokenizer = _grounding_processor.tokenizer
        print('Grounding model loaded!')
    return _grounding_model, _grounding_processor, _grounding_tokenizer


class ParseRequest(BaseModel):
    base64_image: str
    paragraph_enabled: bool = True  # Default to True to maintain backward compatibility

class GroundRequest(BaseModel):
    base64_image: str
    instruction: str
    topk: Optional[int] = 3


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
    """Find a specific UI element by description using GUI-Actor visual grounding.

    Returns normalized click coordinates [0,1] and image pixel dimensions.
    """
    if not args.enable_grounding:
        return {"error": "Grounding not enabled. Start server with --enable_grounding"}

    from gui_actor.inference import inference
    from gui_actor.constants import grounding_system_message

    start = time.time()
    model, processor, tokenizer = get_grounding_model()

    image_bytes = base64.b64decode(request.base64_image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": grounding_system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.instruction},
            ],
        },
    ]

    pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=request.topk)
    latency = time.time() - start

    topk_points = pred.get("topk_points", [])
    # Convert normalized coords to pixel coords
    points_px = [{"x": pt[0] * w, "y": pt[1] * h, "x_norm": pt[0], "y_norm": pt[1]} for pt in topk_points]

    result = {
        "points": points_px,
        "topk_values": pred.get("topk_values", []),
        "image_size": [w, h],
        "latency": latency,
    }
    print(f'ground: "{request.instruction}" -> {len(points_px)} points, {latency:.2f}s')
    return result


@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
