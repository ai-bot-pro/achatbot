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

## fill params for triton server
```shell
# whisper large-v3 | whisper_bls + tensorrt_llm decoupled_mode:False
modal run src/llm/trtllm/whisper/ready_model.py \
    --tag-or-hash "feat/asr" \
    --engine-dir "trt_engines_float16" \
    --whisper-params "triton_max_batch_size:8,max_queue_delay_microseconds:5000,n_mels:128" \
    --whisper-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false" \
    --whisper-tensorrt-llm-params "max_tokens_in_paged_kv_cache:24000,max_attention_window_size:2560,batch_scheduler_policy:guaranteed_no_evict,batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,triton_max_batch_size:8,max_queue_delay_microseconds:0,max_beam_width:1,enable_kv_cache_reuse:False,normalize_log_probs:True,enable_chunked_context:False,decoding_mode:top_k_top_p,max_queue_size:0,enable_context_fmha_fp32_acc:False,cross_kv_cache_fraction:0.5,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

# whisper large-v3 | whisper_bls + tensorrt_llm decoupled_mode:True
modal run src/llm/trtllm/whisper/ready_model.py \
    --tag-or-hash "feat/asr" \
    --stream 1 \
    --engine-dir "trt_engines_float16" \
    --whisper-params "triton_max_batch_size:8,max_queue_delay_microseconds:5000,n_mels:128" \
    --whisper-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false" \
    --whisper-tensorrt-llm-params "max_tokens_in_paged_kv_cache:24000,max_attention_window_size:2560,batch_scheduler_policy:guaranteed_no_evict,batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,triton_max_batch_size:8,max_queue_delay_microseconds:0,max_beam_width:1,enable_kv_cache_reuse:False,normalize_log_probs:True,enable_chunked_context:False,decoding_mode:top_k_top_p,max_queue_size:0,enable_context_fmha_fp32_acc:False,cross_kv_cache_fraction:0.5,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"
```

## run triton server
```shell
# run tritonserver with whisper_bls + whisper_tensorrt_llm(decoder)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal serve src/llm/trtllm/whisper/tritonserver.py 

# curl server is ready
curl -vv -X GET "https://weege--tritonserver-serve-dev.modal.run/v2/health/ready" -H  "accept: application/json"

# run grpc tritonserver by tcp tunnel and http server
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal run src/llm/trtllm/whisper/tritonserver.py 
```

# reference:
- ⭐ run whisper with tensorrt-llm: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md
- ⭐ run whisper service with triton tensorrtllm backend: https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/whisper.md
- https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md (fill necessary params)

## wav2vec 2.0
- https://ai.meta.com/research/impact/wav2vec/
- 2019.4 [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
- ⭐️⭐️ 2020.6 [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) | [paper code](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

## whisper
- https://openai.com/index/whisper/
- ⭐️⭐️ 2022.12 [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) | [paper code](https://github.com/openai/whisper)
- ⭐️ https://huggingface.co/learn/audio-course/en/chapter3/seq2seq
- 

> [!NOTE]
> compile_model trtllm-build with TensorRT-LLM version must equal to tritonserver with TensorRT-LLM version
> now last tritonserver container 25.03 use TensortRT-LLM v0.18.0, so compile_model step trtllm-build need use TensortRT-LLM v0.18.0 !!!