'''
python -m omniparserserver --som_model_path ../../weights/ui_detr/model.pth --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cpu --BOX_TRESHOLD 0.3
'''

import sys
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/ui_detr/model.pth', help='Path to the UI-DETR model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.1, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str
    paragraph_enabled: bool = True  # Default to True to maintain backward compatibility

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list, image_size = omniparser.parse(parse_request.base64_image, paragraph_enabled=parse_request.paragraph_enabled)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency, 'image_size': image_size}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
