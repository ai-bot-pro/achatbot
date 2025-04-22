#!/bin/bash

# copyright 2025 by weedge (weege007@gmail.com)
# bash src/llm/transformers/run_omni_cases.sh

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

IMAGE_GPU="L40S"
STAGE="all"
CASE="all"
MODEL_TYPE="qwen2_5omni"


#----- function -------

usage() {
  echo "Usage: $0 [-h] [-s STAGE] [-d IMAGE_GPU] [-m MODEL_TYPE] [-c CASE]"
  echo "  -h                 Show this help message and exit."
  echo "  -s STAGE           Set the stage (default: all)."
  echo "                     Valid options: download, run, run_all, all"
  echo "  -m MODEL_TYPE      model type (default: qwen2_5omni)."
  echo "  -c CASE            run case (default: all)."
  echo "                     Valid options e.g.: all"
  echo "                       universal_audio_understanding"
  echo "                       voice_chatting"
  echo "                       video_information_extracting, screen_recording_interaction"
  echo "                       omni_chatting_for_math, omni_chatting_for_music, multi_round_omni_chatting,asr_stream"
  echo "  -d IMAGE_GPU       Set the GPU image (default: L40S)."
  echo "                     Valid options: A10G A100 A100-80GB L4 L40S H100 https://fullstackdeeplearning.com/cloud-gpus/"
  echo "e.g.: "
  echo "bash run_omni_cases.sh -s all"
  echo "bash run_omni_cases.sh -s download "
  echo "bash run_omni_cases.sh -s run_all"
  echo "bash run_omni_cases.sh -s run -c all"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c universal_audio_understanding"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c voice_chatting"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c video_information_extracting"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c screen_recording_interaction"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_math"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_music"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c multi_round_omni_chatting -d A100-80G"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c image_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c image_chunk_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c asr_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c asr_chunk_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c thinker_chunk_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_stream -d L4"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c screen_recording_interaction_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c screen_recording_interaction_chunk_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c video_information_extracting_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c video_information_extracting_chunk_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_math_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_music_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_math_chunk_stream -d L40s"
  echo "bash run_omni_cases.sh -s run -m qwen2_5omni -c omni_chatting_for_music_chunk_stream -d L40s"
}

run() {
    #local CASE=$1
    echo "run $MODEL_TYPE $CASE $IMAGE_GPU $TAG_OR_COMMIT"
    if [ -e "src/llm/transformers/$MODEL_TYPE.py" ]; then
      echo "src/llm/transformers/$MODEL_TYPE.py exists"
      cd src/llm/transformers/
    else
      cd $SCRIPT_DIR
      wget -q https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/feat/vision_voice/deploy/modal/src/llm/transformers/$MODEL_TYPE.py -O $MODEL_TYPE.py
    fi
    all_cases=(
      "universal_audio_understanding"
      "voice_chatting"
      "video_information_extracting"
      "video_information_extracting_stream"
      "video_information_extracting_chunk_stream"
      "screen_recording_interaction"
      "screen_recording_interaction_stream"
      "screen_recording_interaction_chunk_stream"
      "omni_chatting_for_math"
      "omni_chatting_for_math_stream"
      "omni_chatting_for_math_chunk_stream"
      "omni_chatting_for_music"
      "omni_chatting_for_music_stream"
      "omni_chatting_for_music_chunk_stream"
      "multi_round_omni_chatting"
      "thinker_chunk_stream"
      "image_stream"
      "image_chunk_stream"
      "asr_stream"
      "asr_chunk_stream"
      "omni_chatting_stream"
      "omni_chatting_segment_stream"
    )
    #return
    case $CASE in
      all)
        for CASE in "${all_cases[@]}"; do
          [[ $CASE == "multi_round_omni_chatting" ]] && IMAGE_GPU="A100-80GB"
          echo "IMAGE_GPU=$IMAGE_GPU modal run $MODEL_TYPE.py --task $CASE"
          IMAGE_GPU=$IMAGE_GPU modal run $MODEL_TYPE.py --task $CASE
        done
        ;;
      *)
        if [[ " ${all_cases[@]} " =~ " ${CASE} " ]]; then
          [[ $CASE == "multi_round_omni_chatting" ]] && IMAGE_GPU="A100-80GB"
          echo "IMAGE_GPU=$IMAGE_GPU modal run $MODEL_TYPE.py --task $CASE"
          IMAGE_GPU=$IMAGE_GPU modal run $MODEL_TYPE.py --task $CASE
        else
          echo "$CASE not in ${all_cases[*]}"
          usage
          exit 1
        fi
        ;;
    esac
}

download_models() {
  if [ -e "src/download_models.py" ]; then
    echo "src/download_models.py exists"
    cd src
  else
    cd $SCRIPT_DIR
    wget -q https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/deploy/modal/src/download_models.py -O download_models.py
  fi

  modal run download_models.py --repo-ids "Qwen/Qwen2.5-Omni-7B"
  cd -
}

download_assets() {
  if [ -e "src/download_assets.py" ]; then
    echo "src/download_assets.py exists"
    cd src
  else
    cd $SCRIPT_DIR
    wget -q https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/deploy/modal/src/download_assets.py -O download_assets.py
  fi

  modal run download_assets.py --asset-urls "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/test/img_files/03-Confusing-Pictures.jpg,https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw1.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw2.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw3.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/music.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/math.mp4,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/BAC009S0764W0121.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/10000611681338527501.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/7105431834829365765.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav,https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/shopping.mp4"
  cd -
}


#----- let's go ------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.." || exit

export TAG_OR_COMMIT=$TAG_OR_COMMIT

# 处理命令行参数
while getopts ":d:s:m:c:h" opt; do
  case ${opt} in
    d )
      IMAGE_GPU=$OPTARG
      ;;
    s )
      STAGE=$OPTARG
      ;;
    c )
      CASE=$OPTARG
      ;;
    m )
      MODEL_TYPE=$OPTARG
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


ALLOWED_MODEL_TYPE=("qwen2_5omni" "minicpmo")
if [[ ! " ${ALLOWED_MODEL_TYPE[@]} " =~ " ${MODEL_TYPE} " ]]; then
  echo "Invalid model type: $MODEL_TYPE" 1>&2
  usage
  exit 1
fi

ALLOWED_GPUS=("A100" "A100-80GB" "A10G" "L4" "L40S" "H100")
if [[ ! " ${ALLOWED_GPUS[@]} " =~ " ${IMAGE_GPU} " ]]; then
  echo "if use flash attention, need gpu arch >= 8.0 e.g.:  A100 A100-80G A10G L4 L40S H100"
fi

#export EXTRA_INDEX_URL="https://pypi.org/simple/"
#export ACHATBOT_VERSION="0.0.9.post8"


case $STAGE in
    run)
        run
        ;;
    download)
        download_models
        download_assets
        ;;
    run_all)
        CASE=all
        run
        ;;
    all)
        download_models
        download_assets
        CASE=all
        run
        ;;
    *)
        echo "Invalid stage: $STAGE" 1>&2
        usage
        exit 1
        ;;
esac