#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit

export EXTRA_INDEX_URL="https://pypi.org/simple/"
export TTS_TAG="tts_generator_spark"
export ACHATBOT_VERSION="0.0.9.post3"

# transformers with cpu
modal run src/tts/run_generator_tts.py
# transformers with gpu cuda
IMAGE_GPU=T4 modal run src/tts/run_generator_tts.py

# f16 don't support
# llamacpp with cpu, quant Q8_0
GENERATOR_ENGINE=llamacpp QUANT=Q8_0 modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q4_K_M
GENERATOR_ENGINE=llamacpp QUANT=Q4_K_M modal run src/tts/run_generator_tts.py
# llamacpp with cpu, quant Q2_K
GENERATOR_ENGINE=llamacpp QUANT=Q2_K modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q8_0 flash attention
GENERATOR_ENGINE=llamacpp QUANT=Q8_0 IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q4_K_M flash attention
GENERATOR_ENGINE=llamacpp QUANT=Q4_K_M IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
# llamacpp with gpu cuda, quant Q2_K
GENERATOR_ENGINE=llamacpp QUANT=Q2_K IMAGE_GPU=T4 modal run src/tts/run_generator_tts.py

# vllm with gpu cuda | bf16 | Using Flash Attention backend | Using FlashInfer for top-p & top-k sampling
GENERATOR_ENGINE=vllm IMAGE_GPU=L4 modal run src/tts/run_generator_tts.py
GENERATOR_ENGINE=vllm IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py

# tensorrt-llm with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
GENERATOR_ENGINE=trtllm IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py
# tensorrt-llm runner with gpu cuda | bf16 | Using FlashInfer Attention backend (flashinfer.jit)
GENERATOR_ENGINE=trtllm_runner IMAGE_GPU=L40S modal run src/tts/run_generator_tts.py