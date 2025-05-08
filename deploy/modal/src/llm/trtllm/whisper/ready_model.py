import os
import modal

app_name = "whisper"
app = modal.App(f"{app_name}-tritonserver")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.env({})

TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
TRITONSERVER_DIR = "/root/tritonserver"
tritonserver_vol = modal.Volume.from_name("tritonserver", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

FILL_TEMPLATE_URL = "https://raw.githubusercontent.com/triton-inference-server/tensorrtllm_backend/refs/heads/r24.10/tools/fill_template.py"


# see: https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    retries=0,
    image=image,
    volumes={
        ASSETS_DIR: assets_dir,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def ready_model(
    whisper_params: str,
    whisper_infer_bls_params: str = "",
    whisper_tensorrt_llm_cpprunner_params: str = "",
    whisper_bls_params: str = "",
    whisper_tensorrt_llm_params: str = "",
    tag_or_hash: str = "",
    stream: bool = False,
    fill_template_url: str = FILL_TEMPLATE_URL,
    engine_dir: str = "trt_engines_float16",
) -> str:
    import subprocess

    if tag_or_hash == "":
        tag_or_hash = "main"

    cmd = f"git clone https://github.com/ai-bot-pro/achatbot.git -b {tag_or_hash}"
    subprocess.run(cmd, cwd="/", shell=True, check=True)

    model_repo = os.path.join(TRITONSERVER_DIR, app_name)
    subprocess.run(f"rm -rf {model_repo}", shell=True, check=False)
    subprocess.run(f"mkdir -p {model_repo}", shell=True, check=True)

    cmd = f"wget {fill_template_url} -O /root/fill_template.py"
    subprocess.run(cmd, cwd="/", shell=True, check=True)

    # whisper (python BE)
    cmd = f"cp -r /achatbot/deploy/modal/src/llm/trtllm/model_repo/whisper {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, cwd="/", check=True)
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    cmd = (
        f"python /root/fill_template.py "
        + f"-i {model_repo}/whisper/config.pbtxt "
        + f"{whisper_params},engine_dir:{local_trt_build_dir},tokenizer_dir:{ASSETS_DIR},mel_filters_dir:{ASSETS_DIR}".strip(
            ","
        )
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    # whisper infer bls (python BE)
    cmd = f"cp -r /achatbot/deploy/modal/src/llm/trtllm/model_repo/whisper_infer_bls {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, cwd="/", check=True)
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    cmd = (
        f"python /root/fill_template.py "
        + f"-i {model_repo}/whisper_infer_bls/config.pbtxt "
        + f"{whisper_infer_bls_params},engine_dir:{local_trt_build_dir},tokenizer_dir:{ASSETS_DIR}".strip(
            ","
        )
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    # whisper_tensorrt_llm cpp runner (python BE)
    cmd = f"cp -r /achatbot/deploy/modal/src/llm/trtllm/model_repo/whisper_tensorrt_llm_cpprunner {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, cwd="/", check=True)
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    cmd = (
        f"python /root/fill_template.py "
        + f"-i {model_repo}/whisper_tensorrt_llm_cpprunner/config.pbtxt "
        + f"{whisper_tensorrt_llm_cpprunner_params},engine_dir:{local_trt_build_dir},mel_filters_dir:{ASSETS_DIR}".strip(
            ","
        )
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    # whishper bls (python BE)
    cmd = f"cp -r /achatbot/deploy/modal/src/llm/trtllm/model_repo/whisper_bls {model_repo}"
    print(cmd)
    subprocess.run(cmd, shell=True, cwd="/", check=True)
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    decoupled_mode = "True" if stream else "False"
    cmd = (
        f"python /root/fill_template.py "
        + f"-i {model_repo}/whisper_bls/config.pbtxt "
        + f"{whisper_bls_params},decoupled_mode:{decoupled_mode},engine_dir:{local_trt_build_dir},tokenizer_dir:{ASSETS_DIR},mel_filters_dir:{ASSETS_DIR}".strip(
            ","
        )
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)

    # whisper_tensorrt_llm (python or tensorrtllm BE)
    cmd = (
        f"cp -r /achatbot/deploy/modal/src/llm/trtllm/model_repo/whisper_tensorrt_llm {model_repo}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True, cwd="/", check=True)
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, engine_dir)
    decoupled_mode = "True" if stream else "False"

    cmd = (
        f"python /root/fill_template.py "
        + f"-i {model_repo}/whisper_tensorrt_llm/config.pbtxt "
        + f"{whisper_tensorrt_llm_params},engine_dir:{local_trt_build_dir}/decoder,encoder_engine_dir:{local_trt_build_dir}/encoder,decoupled_mode:{decoupled_mode}".strip(
            ","
        )
    )
    print(cmd)
    subprocess.run(cmd.strip().split(" "), cwd="/", check=True)


"""
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
"""


@app.local_entrypoint()
def main(
    whisper_params: str = "",
    whisper_infer_bls_params: str = "",
    whisper_tensorrt_llm_cpprunner_params: str = "",
    whisper_bls_params: str = "",
    whisper_tensorrt_llm_params: str = "",
    tag_or_hash: str = "main",
    stream: str = "",
    fill_template_url: str = FILL_TEMPLATE_URL,
    engine_dir: str = "trt_engines_float16",
):
    ready_model.remote(
        whisper_params,
        whisper_infer_bls_params,
        whisper_tensorrt_llm_cpprunner_params,
        whisper_bls_params,
        whisper_tensorrt_llm_params,
        tag_or_hash,
        bool(stream),
        fill_template_url,
        engine_dir,
    )
