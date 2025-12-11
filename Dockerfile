FROM nvidia/cuda:11.4.3-base-ubuntu20.04 as nvidia

FROM python:3.12

COPY --from=nvidia /usr/local /usr/local

WORKDIR /OmniParser

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install libgl1 -y && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Install dependencies
    pip install -r requirements.txt --no-cache-dir && \
    # Cleanup
    rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence weights/ui_detr && \
    # Download the models from Hugging Face
    huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "icon_caption/*" && \
    mv weights/icon_caption weights/icon_caption_florence && \
    # Download UI-DETR model
    mkdir -p weights/ui_detr && \
    huggingface-cli download racineai/UI-DETR-1 model.pth --local-dir weights/ui_detr
    
COPY . .

EXPOSE 8000

WORKDIR omnitool/omniparserserver/

CMD ["python", "-m", "omniparserserver", "--som_model_path", "../../weights/ui_detr/model.pth", "--caption_model_name", "florence2", "--caption_model_path", "../../weights/icon_caption_florence", "--device", "cuda", "--BOX_TRESHOLD", "0.3"]

ENV PATH=${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
