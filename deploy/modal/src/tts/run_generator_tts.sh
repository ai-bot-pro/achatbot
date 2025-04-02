#!/bin/bash

# copyright 2025 by weedge (weege007@gmail.com)
# bash src/tts/run_generator_tts.sh

# https://modal.com/docs/guide
pip install -q modal

usage() {
  echo "Usage: $0 [-h] [-s STAGE] [-t TTS_TEXT] [-g GENERATOR_ENGINE] [-d IMAGE_GPU] [-a TTS_TAG]"
  echo "  -h                 Show this help message and exit."
  echo "  -s STAGE           Set the stage (default: all)."
  echo "                     Valid options: download, run, run_all, all"
  echo "  -t TTS_TEXT        Set the TTS text to be generated."
  echo "  -g GENERATOR_ENGINE Set the generator engine (default: transformers)."
  echo "                     Valid options: transformers, llamacpp, vllm, sglang, trtllm, trtllm_runner"
  echo "  -d IMAGE_GPU       Set the GPU image (default: L40S)."
  echo "                     Valid options: T4 A10G A100 L4 L40S H100 https://fullstackdeeplearning.com/cloud-gpus/"
  echo "  -a TTS_TAG         Set the TTS tag (default: tts_generator_spark)."
}

run() {
    local GENERATOR_ENGINE=$1
    echo "run $TTS_TAG $GENERATOR_ENGINE $IMAGE_GPU"
    # return
    export GENERATOR_ENGINE=$GENERATOR_ENGINE
    case $GENERATOR_ENGINE in
      transformers)
        modal run src/tts/run_generator_tts.py
        IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        ;;
      llamacpp)
        # qunat f16 unsupported

        for QUANT in Q8_0 Q6_K Q5_K_S Q5_K_M Q4_K_M Q4_K_S IQ4_XS Q3_K_L Q3_K_M Q3_K_S Q2_K; do
            # llamacpp with cpu, quant Q8_0
            QUANT=$QUANT modal run src/tts/run_generator_tts.py
            # llamacpp with gpu cuda
            QUANT=$QUANT IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        done
        ;;
      vllm)
        # vllm with gpu cuda | bf16 | Using Flash Attention backend | Using FlashInfer for top-p & top-k sampling
        IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        ;;
      sglang)
        # tensorrt-llm with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
        IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        ;;
      trtllm)
        echo "Tips: run trtllm_runner generator engine, if use diff gpu arch, need rebuild engine"
        # tensorrt-llm with gpu cuda | bf16
        IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        ;;
      trtllm_runner)
        echo "Tips: run trtllm_runner generator engine, if use diff gpu arch, need rebuild engine"
        # tensorrt-llm runner with gpu cuda | bf16
        IMAGE_GPU=$IMAGE_GPU modal run src/tts/run_generator_tts.py
        ;;
      *)
        echo "Invalid GENERATOR_ENGINE: $GENERATOR_ENGINE" 1>&2
        usage
        exit 1
        ;;
    esac
}

download() {
    modal run src/download_models.py --repo-ids "SparkAudio/Spark-TTS-0.5B"
    modal run src/download_models.py --repo-ids "mradermacher/SparkTTS-LLM-GGUF"
}




set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit

TTS_TEXT="hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
GENERATOR_ENGINE="transformers"
IMAGE_GPU="L40S"
TTS_TAG="tts_generator_spark"
STAGE="all"

# 处理命令行参数
while getopts ":t:g:d:a:s:h" opt; do
  case ${opt} in
    t )
      TTS_TEXT=$OPTARG
      ;;
    g )
      GENERATOR_ENGINE=$OPTARG
      ;;
    d )
      IMAGE_GPU=$OPTARG
      ;;
    a )
      TTS_TAG=$OPTARG
      ;;
    s )
      STAGE=$OPTARG
      ;;
    h )
      usage
      exit 0
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      usage
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

if [ -z "$TTS_TEXT" ]; then
  echo "TTS_TEXT is required."
  usage
  exit 1
fi

if [ -z "$GENERATOR_ENGINE" ]; then
  echo "GENERATOR_ENGINE is required."
  usage
  exit 1
fi

ALLOWED_GPUS=("A100" "A10G" "L4" "L40S" "H100")
if [[ ! " ${ALLOWED_GPUS[@]} " =~ " ${IMAGE_GPU} " ]]; then
  echo "if use flash attention, need gpu arch >= 8.0 e.g.:  A100 A10G L4 L40S H100"
fi

export EXTRA_INDEX_URL="https://pypi.org/simple/"
export ACHATBOT_VERSION="0.0.9.post3"
export TTS_TAG=$TTS_TAG
export TTS_TEXT=$TTS_TEXT


case $STAGE in
    run)
        run $GENERATOR_ENGINE
        ;;
    download)
        download
        ;;
    run_all)
        for GENERATOR_ENGINE in "transformers" "llamacpp" "vllm" "sglang" "trtllm" "trtllm_runner"; do
            run $GENERATOR_ENGINE
        done
        ;;
    all)
        download
        for GENERATOR_ENGINE in "transformers" "llamacpp" "vllm" "sglang" "trtllm" "trtllm_runner"; do
            run $GENERATOR_ENGINE
        done
        ;;
    *)
        echo "Invalid stage: $STAGE" 1>&2
        usage
        exit 1
        ;;
esac