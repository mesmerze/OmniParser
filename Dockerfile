FROM nvidia/cuda:11.4.3-base-ubuntu20.04 as nvidia

FROM python:3.12

COPY --from=nvidia /usr/local /usr/local

WORKDIR /OmniParser

COPY . .

RUN apt-get update && apt-get install libgl1 -y

RUN pip install -r requirements.txt

RUN rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence

RUN for folder in icon_caption icon_detect; do \
    huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; \
    done && \
    mv weights/icon_caption weights/icon_caption_florence

EXPOSE 8000

WORKDIR omnitool/omniparserserver/

CMD ["python", "-m", "omniparserserver", "--som_model_path", "../../weights/icon_detect/model.pt", "--caption_model_name", "florence2", "--caption_model_path", "../../weights/icon_caption_florence", "--device", "cuda", "--BOX_TRESHOLD", "0.05"]

ENV PATH=${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
