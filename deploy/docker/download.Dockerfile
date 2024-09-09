# download model for local inference
FROM achatbot:base AS download_bin
ARG A_HF_ENDPOINT="https://huggingface.co"
ARG A_CHECKPOINT_URL=${A_HF_ENDPOINT}/karpathy/tinyllamas/resolve/main/stories42M.bin
ARG ASR_SENSEVOICE_CKPT=FunAudioLLM/SenseVoiceSmall

COPY download_model.sh ./

RUN mkdir -p ~/.achatbot/models \
    && curl -L ${A_CHECKPOINT_URL} -o ~/.achatbot/models/stories42M.bin \
    && bash download_model.sh ~/.achatbot/models