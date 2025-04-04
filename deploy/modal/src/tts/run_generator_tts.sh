#!/bin/bash

# copyright 2025 by weedge (weege007@gmail.com)
# bash src/tts/run_generator_tts.sh

# https://modal.com/docs/guide
if command -v modal &> /dev/null
then
  echo "modal command found. start to run ..."
else
  echo "pip install modal ..."
  pip install -q modal
  modal setup
fi
modal --version

set -e

#---- default values ----

TTS_TEXT="hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。|PyTorch 将值组织成Tensor ， Tensor是具有丰富数据操作操作的通用 n 维数组。|Module 定义从输入值到输出值的转换，其在正向传递期间的行为由其forward成员函数指定。Module 可以包含Tensor作为参数。|例如，线性模块包含权重参数和偏差参数，其正向函数通过将输入与权重相乘并添加偏差来生成输出。|应用程序通过在自定义正向函数中将本机Module （*例如*线性、卷积等）和Function （例如relu、pool 等）拼接在一起来组成自己的Module 。|典型的训练迭代包含使用输入和标签生成损失的前向传递、用于计算参数梯度的后向传递以及使用梯度更新参数的优化器步骤。|更具体地说，在正向传递期间，PyTorch 会构建一个自动求导图来记录执行的操作。|然后，在反向传播中，它使用自动梯度图进行反向传播以生成梯度。最后，优化器应用梯度来更新参数。训练过程重复这三个步骤，直到模型收敛。"
GENERATOR_ENGINE="transformers"
IMAGE_GPU="L40S"
TTS_TAG="tts_generator_spark"
STAGE="all"
ALLOWED_QUANTS=("Q8_0" "Q6_K" "Q5_K_S" "Q5_K_M" "Q4_K_M" "Q4_K_S" "IQ4_XS" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")
QUANTS=${ALLOWED_QUANTS[*]}


#----- function -------

usage() {
  echo "Usage: $0 [-h] [-s STAGE] [-t TTS_TEXT] [-g GENERATOR_ENGINE] [-d IMAGE_GPU] [-a TTS_TAG] [-q QUANTS]"
  echo "  -h                 Show this help message and exit."
  echo "  -s STAGE           Set the stage (default: all)."
  echo "                     Valid options: download, run, run_all, all"
  echo "  -t TTS_TEXT        Set the TTS text to be generated."
  echo "  -g GENERATOR_ENGINE Set the generator engine (default: transformers)."
  echo "                     Valid options: transformers, llamacpp, vllm, sglang, trtllm, trtllm_runner"
  echo "  -d IMAGE_GPU       Set the GPU image (default: L40S)."
  echo "                     Valid options: T4 A10G A100 L4 L40S H100 https://fullstackdeeplearning.com/cloud-gpus/"
  echo "  -a TTS_TAG         Set the TTS tag (default: tts_generator_spark)."
  echo "  -q QUANTS          Set llamapcpp gguf quants (default: ${ALLOWED_QUANTS[*]})."
  echo "e.g.: "
  echo "bash run_generator_tts.sh -s all"
  echo "bash run_generator_tts.sh -s download "
  echo "bash run_generator_tts.sh -s run_all"
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。' -g transformers -d T4 -a tts_generator_spark "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。' -g transformers -d T4 -a tts_generator_spark "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。' -g llamacpp -d T4 -a tts_generator_spark -q 'Q8_0' "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。' -g llamacpp -d T4 -a tts_generator_spark -q 'Q8_0 Q4_K_M Q2_K' "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。' -g vllm -d L4 -a tts_generator_spark "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。' -g sglang -d L4 -a tts_generator_spark "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。' -g trtllm -d L4 -a tts_generator_spark "
  echo "bash run_generator_tts.sh -s run -t 'hello,你好，我是机器人。|万物之始,大道至简,衍化至繁。|君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。' -g trtllm -d L40s -a tts_generator_spark "
}

run() {
    local GENERATOR_ENGINE=$1
    echo "run $TTS_TAG $GENERATOR_ENGINE $IMAGE_GPU"
    if [ -e "src/tts/run_generator_tts.py" ]; then
      echo "src/tts/run_generator_tts.py exists"
      cd src/tts
    else
      cd $SCRIPT_DIR
      wget -q https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/deploy/modal/src/tts/run_generator_tts.py -O run_generator_tts.py
    fi
    # return
    export GENERATOR_ENGINE=$GENERATOR_ENGINE
    case $GENERATOR_ENGINE in
      transformers)
        modal run run_generator_tts.py
        IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
        ;;
      llamacpp)
        # quant f32/f16 unsupported
        read -a array <<< "$QUANTS"
        for element in "${array[@]}"; do
            if [[ " ${ALLOWED_QUANTS[@]} " =~ " ${element} " ]]; then
              echo "run llamacpp quant $element with cpu and gpu"
              # llamacpp with cpu, quant Q8_0
              QUANT=$element modal run run_generator_tts.py
              # llamacpp with gpu cuda
              QUANT=$element IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
            else
              echo "$element not in ${ALLOWED_QUANTS[*]}"
            fi
        done
        ;;
      vllm)
        # vllm with gpu cuda | bf16 | Using Flash Attention backend | Using FlashInfer for top-p & top-k sampling
        IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
        ;;
      sglang)
        # tensorrt-llm with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
        IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
        ;;
      trtllm)
        echo "Tips: run trtllm_runner generator engine, if use diff gpu arch, need rebuild engine"
        # tensorrt-llm with gpu cuda | bf16
        IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
        ;;
      trtllm_runner)
        echo "Tips: run trtllm_runner generator engine, if use diff gpu arch, need rebuild engine"
        # tensorrt-llm runner with gpu cuda | bf16
        IMAGE_GPU=$IMAGE_GPU modal run run_generator_tts.py
        ;;
      *)
        echo "Invalid GENERATOR_ENGINE: $GENERATOR_ENGINE" 1>&2
        usage
        exit 1
        ;;
    esac
}

download() {
    if [ -e "src/download_models.py" ]; then
      echo "src/download_models.py exists"
      cd src
    else
      cd $SCRIPT_DIR
      wget -q https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/deploy/modal/src/download_models.py -O download_models.py
    fi
    modal run download_models.py --repo-ids "SparkAudio/Spark-TTS-0.5B"
    modal run download_models.py --repo-ids "mradermacher/SparkTTS-LLM-GGUF"
}


#----- let's go ------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit

# 处理命令行参数
while getopts ":t:g:d:a:s:q:h" opt; do
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
    q )
      QUANTS=$OPTARG
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
export ACHATBOT_VERSION="0.0.9.post4"
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