import os
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-tritonserver")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git")
image = image.env(
    {
        "HF_REPO_ID": os.getenv("HF_REPO_ID", "SparkAudio/Spark-TTS-0.5B"),
    }
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)
TRITONSERVER_DIR = "/root/tritonserver"
tritonserver_vol = modal.Volume.from_name("tritonserver", create_if_missing=True)


# see: https://github.com/triton-inference-server/python_backend/blob/main/README.md
@app.function(
    retries=0,
    image=image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
        TRITONSERVER_DIR: tritonserver_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def ready_model(
    spark_tts_params: str,
    audio_tokenizer_params: str,
    tensorrt_llm_params: str,
    vocoder_params: str,
    trt_dtype: str,
    tag_or_hash: str,
) -> str:
    import subprocess

    if tag_or_hash == "":
        tag_or_hash = "main"

    cmd = f"git clone https://github.com/SparkAudio/Spark-TTS -b {tag_or_hash}".split(" ")
    subprocess.run(cmd, cwd="/")

    model_repo = os.path.join(TRITONSERVER_DIR, app_name)
    cmd = f"rm -rf {model_repo}".split(" ")
    subprocess.run(cmd, cwd="/")

    cmd = f"cp -r /Spark-TTS/runtime/triton_trtllm/model_repo {model_repo}".split(" ")
    subprocess.run(cmd, cwd="/")

    hf_model_local_dir = os.path.join(
        HF_MODEL_DIR, os.getenv("HF_REPO_ID", "SparkAudio/Spark-TTS-0.5B")
    )

    # spark tts (Combine 「 audio_tokenizer -> trt llm enginer -> vocoder 」 services)
    hf_model_local_llm_dir = os.path.join(hf_model_local_dir, "LLM")
    cmd = f"python3 scripts/fill_template.py -i {model_repo}/spark_tts/config.pbtxt" + (
        f" {spark_tts_params},llm_tokenizer_dir:{hf_model_local_llm_dir}".strip(",")
    )
    print(cmd)
    subprocess.run(cmd.split(" "), cwd="/Spark-TTS/runtime/triton_trtllm/")

    # audio_tokenizer
    cmd = f"python3 scripts/fill_template.py -i {model_repo}/audio_tokenizer/config.pbtxt" + (
        f" {audio_tokenizer_params},model_dir:{hf_model_local_dir}".strip(",")
    )
    print(cmd)
    subprocess.run(cmd.split(" "), cwd="/Spark-TTS/runtime/triton_trtllm/")

    # trt llm enginer
    local_trt_build_dir = os.path.join(TRT_MODEL_DIR, app_name, f"trt_engines_{trt_dtype}")
    cmd = f"python3 scripts/fill_template.py -i {model_repo}/tensorrt_llm/config.pbtxt" + (
        f" {tensorrt_llm_params},engine_dir:{local_trt_build_dir }".strip(",")
    )
    print(cmd)
    subprocess.run(cmd.split(" "), cwd="/Spark-TTS/runtime/triton_trtllm/")

    # vocoder
    cmd = f"python3 scripts/fill_template.py -i {model_repo}/vocoder/config.pbtxt" + (
        f" {vocoder_params},model_dir:{hf_model_local_dir}".strip(",")
    )
    print(cmd)
    subprocess.run(cmd.split(" "), cwd="/Spark-TTS/runtime/triton_trtllm/")


"""
# fill template with pbtext file for api params
modal run src/llm/trtllm/bench/tts_spark/ready_model.py \
    --tag-or-hash "main" \
    --trt-dtype "bfloat16" \
    --spark-tts-params "bls_instance_num:4,triton_max_batch_size:16,max_queue_delay_microseconds:0" \
    --audio-tokenizer-params "triton_max_batch_size:16,max_queue_delay_microseconds:0" \
    --tensorrt-llm-params "triton_backend:tensorrtllm,triton_max_batch_size:16,decoupled_mode:False,max_beam_width:1,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32" \
    --vocoder-params "triton_max_batch_size:16,max_queue_delay_microseconds:0" 
"""


@app.local_entrypoint()
def main(
    spark_tts_params: str,
    audio_tokenizer_params: str,
    tensorrt_llm_params: str,
    vocoder_params: str,
    trt_dtype: str = "bfloat16",
    tag_or_hash: str = "main",
):
    ready_model.remote(
        spark_tts_params,
        audio_tokenizer_params,
        tensorrt_llm_params,
        vocoder_params,
        trt_dtype,
        tag_or_hash,
    )
