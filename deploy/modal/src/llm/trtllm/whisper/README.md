<figure>
<img src="https://github.com/user-attachments/assets/c5dc556f-c61b-4e1b-a78d-e63af039ce59" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Figure 1:</b> Whisper model. The architecture
follows the standard Transformer-based encoder-decoder model. A
log-Mel spectrogram is input to the encoder. The last encoder
hidden states are input to the decoder via cross-attention mechanisms. The
decoder autoregressively predicts text tokens, jointly conditional on the
encoder hidden states and previously predicted tokens. Figure source:
<a href="https://openai.com/blog/whisper/">OpenAI Whisper Blog</a>.</figcaption>
</figure>


<figure>
<img src="https://github.com/user-attachments/assets/33e0d42e-f823-4a04-b86d-369d5e7bff66" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Figure 2:</b> The multitask training tokenize format for transcription with timestamps and translation </figcaption>
</figure>


---

podcast: https://podcast-997.pages.dev/podcast/c99c96522e1b4b6ea398126e026f1eaf

---

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
    --whisper-params "triton_max_batch_size:8,max_queue_delay_microseconds:5000,n_mels:128,zero_pad:false,cross_kv_cache_fraction:0.5,kv_cache_free_gpu_mem_fraction:0.5" \
    --whisper-infer-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0" \
    --whisper-tensorrt-llm-cpprunner-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false,cross_kv_cache_fraction:0.5,kv_cache_free_gpu_mem_fraction:0.5" \
    --whisper-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false" \
    --whisper-tensorrt-llm-params "triton_backend:tensorrtllm,max_tokens_in_paged_kv_cache:24000,max_attention_window_size:2560,batch_scheduler_policy:guaranteed_no_evict,batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,triton_max_batch_size:8,max_queue_delay_microseconds:0,max_beam_width:1,enable_kv_cache_reuse:False,normalize_log_probs:True,enable_chunked_context:False,decoding_mode:top_k_top_p,max_queue_size:0,enable_context_fmha_fp32_acc:False,cross_kv_cache_fraction:0.5,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

# whisper large-v3 | whisper_bls + tensorrt_llm decoupled_mode:True
modal run src/llm/trtllm/whisper/ready_model.py \
    --tag-or-hash "feat/asr" \
    --stream 1 \
    --engine-dir "trt_engines_float16" \
    --whisper-params "triton_max_batch_size:8,max_queue_delay_microseconds:5000,n_mels:128,zero_pad:false,cross_kv_cache_fraction:0.5,kv_cache_free_gpu_mem_fraction:0.5" \
    --whisper-infer-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0" \
    --whisper-tensorrt-llm-cpprunner-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false,cross_kv_cache_fraction:0.5,kv_cache_free_gpu_mem_fraction:0.5" \
    --whisper-bls-params "triton_max_batch_size:8,max_queue_delay_microseconds:0,n_mels:128,zero_pad:false" \
    --whisper-tensorrt-llm-params "triton_backend:tensorrtllm,max_tokens_in_paged_kv_cache:24000,max_attention_window_size:2560,batch_scheduler_policy:guaranteed_no_evict,batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,triton_max_batch_size:8,max_queue_delay_microseconds:0,max_beam_width:1,enable_kv_cache_reuse:False,normalize_log_probs:True,enable_chunked_context:False,decoding_mode:top_k_top_p,max_queue_size:0,enable_context_fmha_fp32_acc:False,cross_kv_cache_fraction:0.5,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"
```

## run triton server
```shell
# run tritonserver with whisper(python BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper modal serve src/llm/trtllm/whisper/tritonserver.py

# run tritonserver with whisper_infer_bls + whisper_tensorrt_llm_cpprunner(encoder-decoder python BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_infer_bls,whisper_tensorrt_llm_cpprunner modal serve src/llm/trtllm/whisper/tritonserver.py 

# run tritonserver with whisper_bls + whisper_tensorrt_llm(encoder-decoder tensorrtllm BE)
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal serve src/llm/trtllm/whisper/tritonserver.py 

# curl server is ready
curl -vv -X GET "https://weege009--tritonserver-serve-dev.modal.run/v2/health/ready" -H  "accept: application/json"

# run grpc tritonserver by tcp tunnel and http server
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper modal run src/llm/trtllm/whisper/tritonserver.py
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_bls,whisper_tensorrt_llm modal run src/llm/trtllm/whisper/tritonserver.py 
APP_NAME=whisper TENSORRT_LLM_MODEL_NAME=whisper_infer_bls,whisper_tensorrt_llm_cpprunner modal run src/llm/trtllm/whisper/tritonserver.py 
```

## test whisper service
```shell
# gRPC
## health check for whisper
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --model-name whisper \
    --model-names whisper \
    --server-url "r15.modal.host:34101"

## health check for whisper_bls,whisper_tensorrt_llm
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --server-url "r15.modal.host:34101"

## health check for whisper_infer_bls, whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --model-name whisper_infer_bls \
    --model-names whisper_infer_bls,whisper_tensorrt_llm_cpprunner \
    --server-url "r28.modal.host:38535"


## single wav test for whisper
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --model-name whisper \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --model-name whisper \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## single wav test for whisper_bls,whisper_tensorrt_llm
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r24.modal.host:44175"

## single wav test for whisper_infer_bls,whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --model-name whisper_infer_bls \
    --server-url "r28.modal.host:38535"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --model-name whisper_infer_bls \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper_infer_bls \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper_infer_bls \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## bench for whisper
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --model-name whisper \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## bench (concurency_cn:1->2->4->8->16 | batch_size(requests_cn):1->2->4->8)
## bench throughput and latency, grpc just test, modal support http, grpc now use tunnel
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "r28.modal.host:33695"

modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "r18.modal.host:41787"

modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r18.modal.host:41787"

## bench for whisper_infer_bls,whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --model-name whisper_infer_bls \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

# http
## health check
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --action health --verbose \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

## single wav test
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

## bench (concurency_cn:1->2->4->8->16 | batch_size(requests_cn):1->2->4->8)
## bench throughput and latency, grpc just test, modal support http, grpc now use tunnel
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

# other the same as grpc :-)

# WER eval
see run.py to change
```

# reference:
- ⭐ run whisper with tensorrt-llm: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/README.md
- ⭐ run whisper service with triton tensorrtllm backend: https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/whisper.md
- https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md (fill necessary params)
- deploy with triton cpp runner: https://github.com/k2-fsa/sherpa/tree/master/triton/whisper

## wav2vec (CTC Transformer Encoder-only architecture)
- ⭐️ CTC: https://huggingface.co/learn/audio-course/en/chapter3/ctc
- wav2vec: https://ai.meta.com/research/impact/wav2vec/ 
- 2019.4 [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
- ⭐️⭐️ 2020.6 [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) | [paper code](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
- wav2vec-U: https://ai.meta.com/blog/wav2vec-unsupervised-speech-recognition-without-supervision/

## whisper (Seq2Seq Transformer Encoder-Decoder architecture)
- https://openai.com/index/whisper/
- ⭐️⭐️ 2022.9 [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) | [paper code](https://github.com/openai/whisper)
- tiktoken text tokenizer from GPT-2: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
- ⭐️ https://huggingface.co/learn/audio-course/en/chapter3/seq2seq (ASR TTS with seq2seq transformer model, but TTS is more complex than ASR, need vocoder to generate audio waveform,This vocoder is not part of the seq2seq architecture and is trained separately, e.g.: hifigan, bigvgan)
- whisper fine-tuning: https://github.com/openai/whisper/discussions/759 | https://huggingface.co/blog/fine-tune-whisper
- fine-tune whisper with k2(icefall): https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR/whisper

---

> [!NOTE]
> compile_model trtllm-build with TensorRT-LLM version must equal to tritonserver with TensorRT-LLM version
> now last tritonserver container 25.03 use TensortRT-LLM v0.18.0, so compile_model step trtllm-build need use TensortRT-LLM v0.18.0 !!!