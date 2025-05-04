```
# https://nvidia.github.io/TensorRT-LLM/index.html (nice doc)
"""
NeMo -------------
                  |
HuggingFace ------
                  |   convert                             build                    load
Modelopt ---------  ----------> TensorRT-LLM Checkpoint --------> TensorRT Engine ------> TensorRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
"""
# https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
```
1. download  model hf SparkAudio/Spark-TTS-0.5B ckpt (Bicodec(audio tokenizer) + Wav2Vec2FeatureExtractor(features extractor)  + Qwen2.5 0.5B LLM)
```shell
modal run src/download_models.py --repo-ids "SparkAudio/Spark-TTS-0.5B"
```
2. convert hf SparkAudio/Spark-TTS-0.5B LLM(Qwen2.5 0.5B) to TensorRT-LLM Checkpoint with defined quantization and model parallel mapping, then static build TensorRT Engine with build compile params
```shell
# covert to dtype bfloat16
modal run src/llm/trtllm/tts_spark/compile_model.py \
    --app-name "tts-spark" \
    --hf-repo-dir "SparkAudio/Spark-TTS-0.5B/LLM" \
    --trt-dtype "bfloat16" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"

# use convert from convert-script-url,
modal run src/llm/trtllm/tts_spark/compile_model.py \
    --app-name "tts-spark" \
    --hf-repo-dir "SparkAudio/Spark-TTS-0.5B/LLM" \
    --trt-dtype "bfloat16" \
    --convert-script-url "https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/runtime/triton_trtllm/scripts/convert_checkpoint.py" \
    --convert-other-args "" \
    --compile-other-args "--max_batch_size 16 --max_num_tokens 32768"
```
see trtllm-build build TensorRT Engine config(include convert to TensorRT-LLM Checkpoint config)
```json
{
    "version": "0.17.0.post1",
    "pretrained_config": {
        "mlp_bias": false,
        "attn_bias": true,
        "rotary_base": 1000000,
        "rotary_scaling": null,
        "disable_weight_only_quant_plugin": false,
        "num_labels": 1,
        "use_logn_attn": false,
        "moe": {
            "num_experts": 0,
            "shared_expert_intermediate_size": 0,
            "top_k": 0,
            "normalization_mode": 0,
            "sparse_mixer_epsilon": 0.01,
            "tp_mode": 0,
            "device_limited_n_group": 0,
            "device_limited_topk_group": 0,
            "device_limited_routed_scaling_factor": 1
        },
        "architecture": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
        "vocab_size": 166000,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "hidden_act": "silu",
        "logits_dtype": "float32",
        "norm_epsilon": 0.000001,
        "runtime_defaults": null,
        "position_embedding_type": "rope_gpt_neox",
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "mapping": {
            "world_size": 1,
            "gpus_per_node": 8,
            "cp_size": 1,
            "tp_size": 1,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "auto_parallel": false
        },
        "quantization": {
            "quant_algo": null,
            "kv_cache_quant_algo": null,
            "group_size": 128,
            "smoothquant_val": 0.5,
            "clamp_val": null,
            "use_meta_recipe": false,
            "has_zero_point": false,
            "pre_quant_scale": false,
            "exclude_modules": null
        },
        "use_parallel_embedding": false,
        "embedding_sharding_dim": 0,
        "head_size": 64,
        "qk_layernorm": false,
        "rotary_embedding_dim": 64,
        "seq_length": 8192,
        "qwen_type": "qwen2",
        "moe_intermediate_size": 0,
        "moe_shared_expert_intermediate_size": 0,
        "tie_word_embeddings": true
    },
    "build_config": {
        "max_input_len": 1024,
        "max_seq_len": 32768,
        "opt_batch_size": 8,
        "max_batch_size": 16,
        "max_beam_width": 1,
        "max_num_tokens": 32768,
        "opt_num_tokens": 16,
        "max_prompt_embedding_table_size": 0,
        "kv_cache_type": "PAGED",
        "gather_context_logits": false,
        "gather_generation_logits": false,
        "strongly_typed": true,
        "force_num_profiles": null,
        "profiling_verbosity": "layer_names_only",
        "enable_debug_output": false,
        "max_draft_len": 0,
        "speculative_decoding_mode": 1,
        "use_refit": false,
        "input_timing_cache": null,
        "output_timing_cache": "model.cache",
        "lora_config": {
            "lora_dir": [],
            "lora_ckpt_source": "hf",
            "max_lora_rank": 64,
            "lora_target_modules": [],
            "trtllm_modules_to_hf_modules": {}
        },
        "auto_parallel_config": {
            "world_size": 1,
            "gpus_per_node": 8,
            "cluster_key": "L4",
            "cluster_info": null,
            "sharding_cost_model": "alpha_beta",
            "comm_cost_model": "alpha_beta",
            "enable_pipeline_parallelism": false,
            "enable_shard_unbalanced_shape": false,
            "enable_shard_dynamic_shape": false,
            "enable_reduce_scatter": true,
            "builder_flags": null,
            "debug_mode": false,
            "infer_shape": true,
            "validation_mode": false,
            "same_buffer_io": {
                "past_key_value_(\\d+)": "present_key_value_\\1"
            },
            "same_spec_io": {},
            "sharded_io_allowlist": [
                "past_key_value_\\d+",
                "present_key_value_\\d*"
            ],
            "fill_weights": false,
            "parallel_config_cache": null,
            "profile_cache": null,
            "dump_path": null,
            "debug_outputs": []
        },
        "weight_sparsity": false,
        "weight_streaming": false,
        "plugin_config": {
            "dtype": "bfloat16",
            "bert_attention_plugin": "auto",
            "gpt_attention_plugin": "auto",
            "gemm_plugin": "bfloat16",
            "explicitly_disable_gemm_plugin": false,
            "gemm_swiglu_plugin": null,
            "fp8_rowwise_gemm_plugin": null,
            "qserve_gemm_plugin": null,
            "identity_plugin": null,
            "nccl_plugin": null,
            "lora_plugin": null,
            "weight_only_groupwise_quant_matmul_plugin": null,
            "weight_only_quant_matmul_plugin": null,
            "smooth_quant_plugins": true,
            "smooth_quant_gemm_plugin": null,
            "layernorm_quantization_plugin": null,
            "rmsnorm_quantization_plugin": null,
            "quantize_per_token_plugin": false,
            "quantize_tensor_plugin": false,
            "moe_plugin": "auto",
            "mamba_conv1d_plugin": "auto",
            "low_latency_gemm_plugin": null,
            "low_latency_gemm_swiglu_plugin": null,
            "context_fmha": true,
            "bert_context_fmha_fp32_acc": false,
            "paged_kv_cache": true,
            "remove_input_padding": true,
            "reduce_fusion": false,
            "user_buffer": false,
            "tokens_per_block": 64,
            "use_paged_context_fmha": false,
            "use_fp8_context_fmha": false,
            "multiple_profiles": false,
            "paged_state": false,
            "streamingllm": false,
            "manage_weights": false,
            "use_fused_mlp": true,
            "pp_reduce_scatter": false
        },
        "use_strip_plan": false,
        "max_encoder_input_len": 1024,
        "monitor_memory": false,
        "use_mrope": false
    }
}
```
3. ready model repository
   - dev [python bacckend](https://github.com/triton-inference-server/python_backend) (spark_tts, audio_tokenizer, vocoder) like lambda function, u can see [example/add_sub](https://github.com/triton-inference-server/python_backend/tree/main/examples/add_sub) to do
   - tensorrtllm backend don't to dev, use qwen as backbone,use Triton backend [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) + [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) to [Prepare the Model Repository](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#prepare-the-model-repository), use [**inflight_batcher_llm(Using the C++ TensorRT-LLM backend with the executor API, which includes the latest features including inflight batching) tensorrt_llm(This model is a wrapper of your TensorRT-LLM model and is used for inferencing.)**](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt)
   - fill params (dynamic_batching, parameters, input/output etc..): 
     - tensorrtllm backend [Modify the Model Configuration](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#modify-the-model-configuration) to fill api params which used [`fill_template.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/tools/fill_template.py)
     - python backends like tensorrtllm backend which used [`fill_template.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/tools/fill_template.py) to do

there are use developed python backend (spark_tts, audio_tokenizer, vocoder) in https://github.com/SparkAudio/Spark-TTS/tree/main/runtime/triton_trtllm/model_repo and just fill grpc pb stub params like this:
```shell
# fill template with pbtext file for api params
modal run src/llm/trtllm/tts_spark/ready_model.py \
    --tag-or-hash "main" \
    --trt-dtype "bfloat16" \
    --spark-tts-params "bls_instance_num:4,triton_max_batch_size:16,max_queue_delay_microseconds:0" \
    --audio-tokenizer-params "triton_max_batch_size:16,max_queue_delay_microseconds:0" \
    --tensorrt-llm-params "triton_backend:tensorrtllm,triton_max_batch_size:16,decoupled_mode:False,max_beam_width:1,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32" \
    --vocoder-params "triton_max_batch_size:16,max_queue_delay_microseconds:0" 
```

4. run tritonserver
> [!NOTE]
> nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 use python3.12.3, 
> modal python3.12 use python3.12.1, need building-custom-python-backend-stub:
> https://github.com/triton-inference-server/python_backend?#building-custom-python-backend-stub
> more:
> - https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md
> - https://github.com/triton-inference-server/python_backend/blob/main/README.md
```shell
# run tritonserver
APP_NAME=tts-spark modal serve src/llm/trtllm/tts_spark/tritonserver.py 

# curl health to startup tritonserver (as cold-starting)
curl -vv -X GET "https://weedge--tritonserver-serve-dev.modal.run/v2/health/live" -H  "accept: application/json"
```

5. http client test (modal don't support grpc,so don't to test, u can test in local docker)
```shell
modal run src/llm/trtllm/tts_spark/client_http.py \
    --action health_meta_statics \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_health_meta_statics \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action tts \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_tts \
    --output-audio cli_sdk_tts_output.wav \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_tts \
    --output-audio cli_sdk_tts_output.wav \
    --target-text "你好，请将一个儿童故事，文字不少于200字，故事充满童趣和奇妙幻想。" \
    --output-audio "child_story.wav" \
    --server-url "weedge--tritonserver-serve-dev.modal.run"
```

6. bench
- bench data: use yuekai/seed_tts dataset wenetspeech4tts (26 rows)
- infer bench (tritornserver deploy on the L4 GPU , use first Inference to warmup):
- see bench result: https://github.com/ai-bot-pro/achatbot/pull/129#issue-2917069698
```shell
# single test
CLI_MODE=http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "/prompt_audio.wav" \
    --huggingface-dataset ""

# hf dataset with (wenetspeech4tts 26 rows) with concurrency 1 task to bench
CLI_MODE=http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 1 

# single test used by http client sdk
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "/prompt_audio.wav" \
    --huggingface-dataset ""

# hf dataset with (wenetspeech4tts 26 rows) with concurrency 1 task to bench used by http client sdk
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 1 

# hf dataset with (wenetspeech4tts 26 rows) concurrency min(num_tasks,26) task to bench used by http client sdk
# num-tasks: 2->4->8->16->26
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 2
```

todo:
- other GPU comput arch (Ampere+) bench (benchmark performance on different GPU architectures)
- change params to do [Perf Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md) (adjust parameters for optimal performance analysis)

## reference
- TensorRT-LLM:
  - ⭐️ https://nvidia.github.io/TensorRT-LLM/overview.html
  - https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
  - https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
- triton-inference-server(tritonserver):
  - ⭐️ https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
  - ⭐️⭐️ https://github.com/triton-inference-server/server/blob/main/docs/README.md
- pb_stub:
  - [InferenceRequest](https://github.com/triton-inference-server/python_backend/blob/r25.02/src/pb_stub.cc#L1736)
  - [InferenceResponse](https://github.com/triton-inference-server/python_backend/blob/r25.02/src/pb_stub.cc#L1874)



> 雁过留声