FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as nvidia

FROM python:3.12

COPY --from=nvidia /usr/local /usr/local

WORKDIR /OmniParser

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install libgl1 git -y && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Install dependencies
    pip install -r requirements.txt --no-cache-dir && \
    # Cleanup
    rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence weights/ui_detr && \
    # Download the models from Hugging Face
    huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "icon_caption/*" && \
    mv weights/icon_caption weights/icon_caption_florence && \
    # Download UI-DETR model
    mkdir -p weights/ui_detr && \
    huggingface-cli download racineai/UI-DETR-1 model.pth --local-dir weights/ui_detr && \
    # Install GUI-Actor dependencies, flash attention, and download models
    pip install qwen-vl-utils==0.0.8 accelerate --no-cache-dir && \
    pip install flash-attn --no-build-isolation --no-cache-dir && \
    pip install "gui-actor @ git+https://github.com/microsoft/GUI-Actor.git" --no-deps --no-cache-dir && \
    huggingface-cli download microsoft/GUI-Actor-3B-Qwen2.5-VL && \
    huggingface-cli download microsoft/GUI-Actor-Verifier-2B

COPY . .

EXPOSE 8000

WORKDIR omnitool/omniparserserver/

CMD ["python", "-m", "omniparserserver", "--som_model_path", "../../weights/ui_detr/model.pth", "--caption_model_name", "florence2", "--caption_model_path", "../../weights/icon_caption_florence", "--device", "cuda", "--BOX_TRESHOLD", "0.3", "--enable_grounding", "--enable_verifier"]

ENV PATH=${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
