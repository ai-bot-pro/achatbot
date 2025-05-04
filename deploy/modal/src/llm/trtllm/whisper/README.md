## download whisper assets and models

- download assets
```shell
modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"        
modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"
```
- download model ckpt: https://github.com/openai/whisper/blob/v20240930/whisper/__init__.py#L17

```shell
modal run src/download_models.py::download_ckpts --ckpt-urls "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
```

## build whisper tensorrt engine model
```shell
# C++ runtime

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "" \
    --compile-other-args ""

## use INT8 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int8" \
    --compile-other-args ""

## use INT4 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int4" \
    --compile-other-args ""

# Python runtime

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-script-url "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/refs/tags/v0.19.0rc0/examples/whisper/convert_checkpoint.py" \
    --convert-other-args "" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

## use INT8 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int8" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"

## use INT4 weight-only Quant
modal run src/llm/trtllm/whisper/compile_model.py \
    --app-name "whisper" \
    --model-name "large-v3" \
    --trt-dtype "float16" \
    --convert-other-args "--use_weight_only --weight_only_precision int4" \
    --compile-other-args "--paged_kv_cache disable --remove_input_padding disable"
```

## run whisper test
```shell
# run_single_wav_test, NOTE: don't WER
## C++ runtime
modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int8" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int4" \
    --other-args "--log_level info"

## Python runtime
modal run src/llm/trtllm/whisper/run.py \
    --other-args "--use_py_session" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

# run_dataset_bench, NOTE: have WER
## C++ runtime
modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int8" \
    --other-args "--log_level info"

modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16_int4" \
    --other-args "--log_level info"

## Python runtime
modal run src/llm/trtllm/whisper/run.py \
    --task "run_dataset_bench" \
    --app-name "whisper" \
    --engine-dir "trt_engines_float16" \
    --other-args "--log_level info --use_py_session"
```