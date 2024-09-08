# https://vsupalov.com/docker-arg-env-variable-guide/
# docker build arg to all FROM

ARG A_ALPINE_IMG_TAG=latest
FROM alpine:${A_ALPINE_IMG_TAG} as base

# docker img meta
LABEL achatbot.image.authors="weege007@gmail.com"

# docker build arg after each FROM
ARG A_WORKER_PARAM="--port 9997 --nthreads 1"
ARG A_INFERENCE_PARAM="--prompt 'Hello' --steps 1 --nthreads 1 --workers 127.0.0.1:9997"
ARG A_GIT_REPO_URL=https://github.com/ai-bot-pro/achatbot.git
# container env
ENV E_WORKER_PARAM ${A_WORKER_PARAM}
ENV E_INFERENCE_PARAM ${A_INFERENCE_PARAM}

# build prepare layer, use arg/env


# Custom cache invalidation
ARG CACHEBUST=1

FROM base as download_bin
ARG A_HF_ENDPOINT="https://huggingface.co"
ARG A_CHECKPOINT_URL=${A_HF_ENDPOINT}/karpathy/tinyllamas/resolve/main/stories42M.bin
ARG A_TOKEN_URL=https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
# convert checkpoint/tokenizer, use arg
