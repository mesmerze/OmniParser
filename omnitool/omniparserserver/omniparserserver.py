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
    parser.add_argument('--verifier_model', type=str, default='microsoft/GUI-Actor-Verifier-2B', help='HuggingFace model ID for grounding verifier')
    parser.add_argument('--enable_verifier', action='store_true', help='Enable verifier for grounding (re-ranks candidates)')
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

        attn_impl = "flash_attention_2" if device == 'cuda' else None
        _grounding_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            args.grounding_model,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
        _grounding_processor = AutoProcessor.from_pretrained(args.grounding_model)
        _grounding_tokenizer = _grounding_processor.tokenizer
        print('Grounding model loaded!')
    return _grounding_model, _grounding_processor, _grounding_tokenizer


# GUI-Actor Verifier model (lazy-loaded)
_verifier_model = None
_verifier_processor = None


def get_verifier_model():
    global _verifier_model, _verifier_processor
    if _verifier_model is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        dtype = torch.bfloat16 if device != 'cpu' else torch.float32
        print(f'Loading verifier model {args.verifier_model} on {device}...')

        attn_impl = "flash_attention_2" if device == 'cuda' else None
        _verifier_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.verifier_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).eval()
        _verifier_processor = AutoProcessor.from_pretrained(args.verifier_model)
        _verifier_processor.tokenizer.pad_token = _verifier_processor.tokenizer.eos_token
        print('Verifier model loaded!')
    return _verifier_model, _verifier_processor


def verify_point(model, processor, instruction, image, point_px, crop_size=500):
    """Score a candidate point using the verifier. Returns P(True) / (P(True) + P(False))."""
    from PIL import ImageDraw
    import numpy as np
    from qwen_vl_utils import process_vision_info

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    size_factor = h / 1000 * 1.2
    radius = int(np.ceil(7 * size_factor))
    line_width = int(np.ceil(3 * size_factor))
    x, y = int(point_px[0]), int(point_px[1])
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline='red', width=line_width)

    # Crop around point
    left = max(0, x - crop_size)
    right = min(w - 1, x + crop_size)
    top = max(0, y - crop_size)
    bottom = min(h - 1, y + crop_size)
    img = img.crop((left, top, right, bottom))

    prompt = f"Please observe the screenshot and exame whether the hollow red circle accurately placed on the intended position in the image: '{instruction}'. Answer True or False."
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]

    true_id = processor.tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = processor.tokenizer.encode("False", add_special_tokens=False)[0]
    probs = torch.softmax(logits, dim=-1)
    true_prob = probs[0, true_id].item()
    false_prob = probs[0, false_id].item()
    return true_prob / (true_prob + false_prob)


class ParseRequest(BaseModel):
    base64_image: str
    paragraph_enabled: bool = True  # Default to True to maintain backward compatibility

class GroundRequest(BaseModel):
    base64_image: str
    instruction: str
    topk: Optional[int] = 3
    use_verifier: Optional[bool] = None  # None = use server default (--enable_verifier)


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
    When verifier is enabled, re-ranks candidates by verification score.
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

    should_verify = request.use_verifier if request.use_verifier is not None else args.enable_verifier

    pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=request.topk)

    topk_points = pred.get("topk_points", [])

    # When verifier is active, re-derive candidates from raw attention scores
    # for more diverse coverage (matches GUI-Actor repo's approach)
    verified = False
    if should_verify and pred.get("attn_scores") is not None:
        from gui_actor.inference import get_prediction_region_point
        import numpy as np

        n_width = pred.get("n_width")
        n_height = pred.get("n_height")
        attn_scores = torch.tensor(pred["attn_scores"])

        # Get all region centers from attention map (more diverse than topk alone)
        _, all_centers, all_scores, _ = get_prediction_region_point(
            attn_scores, n_width, n_height, top_n=20, return_all_regions=True, rect_center=False
        )
        candidate_points = [(pt[0] * w, pt[1] * h) for pt in all_centers]

        verifier, verifier_proc = get_verifier_model()
        scores = []
        threshold = 0.8
        for pt in candidate_points:
            score = verify_point(verifier, verifier_proc, request.instruction, image, [pt[0], pt[1]])
            scores.append(score)
            if score >= threshold:
                break  # Confident match, stop early

        # Re-rank by verifier score
        ranked = sorted(zip(candidate_points, all_centers[:len(scores)], scores), key=lambda x: x[2], reverse=True)
        points_px = [{"x": r[0][0], "y": r[0][1], "x_norm": r[1][0], "y_norm": r[1][1]} for r in ranked[:request.topk]]
        verifier_scores = [r[2] for r in ranked[:request.topk]]
        verified = True
        print(f'  verifier scores: {[f"{s:.3f}" for s in verifier_scores]}')
    else:
        points_px = [{"x": pt[0] * w, "y": pt[1] * h, "x_norm": pt[0], "y_norm": pt[1]} for pt in topk_points[:request.topk]]
        verifier_scores = None

    latency = time.time() - start

    result = {
        "points": points_px,
        "topk_values": pred.get("topk_values", [])[:request.topk] if not verified else verifier_scores,
        "image_size": [w, h],
        "latency": latency,
        "verified": verified,
    }
    print(f'ground: "{request.instruction}" -> {len(points_px)} points, verified={verified}, {latency:.2f}s')
    return result


@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
