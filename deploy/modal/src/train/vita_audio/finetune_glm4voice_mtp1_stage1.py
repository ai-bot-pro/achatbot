import datetime
import os
import subprocess

import modal


app = modal.App("vita_audio")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100-80GB")

vita_audio_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/VITA-Audio.git",
        "cd /VITA-Audio && git submodule update --init --recursive",
        "cd /VITA-Audio && pip install -q -r requirements_ds_gpu.txt",
    )
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands(
        "cd /VITA-Audio && git pull origin feat/achatbot",
        "cd /VITA-Audio && git checkout a0f85100edebe805186805aeaaccff392cba3c58",
    )
    .env(
        {
            "DDP_BACKEND": "nccl",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NPROC_PER_NODE": IMAGE_GPU.split(":")[-1] if len(IMAGE_GPU.split(":")) > 1 else "1",
        }
    )
    .env(
        {
            # RuntimeError: The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "LLM_MODEL": os.getenv("LLM_MODEL", "finetune_glm4voice_stage1"),
            "AUDIO_ENCODE_MODEL": os.getenv("AUDIO_ENCODE_MODEL", "THUDM/glm-4-voice-tokenizer"),
            "DATA_CONFIG_PATH": os.getenv(
                "DATA_CONFIG_PATH", "/VITA-Audio/configs/sts_finetune_stage1_debug.yaml"
            ),
            "SEQ_LENGTH": os.getenv("SEQ_LENGTH", "8192"),
            # BOOST: 1 10 4 10
            # Balance: 1 4 3 8 4 10
            "TEXT_AUDIO_INTERVAL_RATIO": os.getenv("TEXT_AUDIO_INTERVAL_RATIO", "1 10 4 10"),
            "MAX_STEPS": os.getenv("MAX_STEPS", "4000"),
        }
    )
)

TRAIN_NAME = "finetune_glm4voice_mtp1_stage1"

HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_dir = modal.Volume.from_name("train_output", create_if_missing=True)
TRITON_CACHE_DIR = "/root/.triton"
triton_dir = modal.Volume.from_name("triton_cache", create_if_missing=True)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=1,
    image=vita_audio_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TRITON_CACHE_DIR: triton_dir,
        TRAIN_OUTPUT_DIR: train_out_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(other_args: str = ""):
    import torch

    # --------------- init ----------------------
    AUDIO_TOKENIZER_TYPE = "glm4voice"
    AUDIO_TOKENIZER_MODEL_PATH = os.path.join(
        HF_MODEL_DIR, os.getenv("AUDIO_ENCODE_MODEL", "THUDM/glm-4-voice-tokenizer")
    )

    LLM_MODEL_PATH = os.path.join(
        TRAIN_OUTPUT_DIR, os.getenv("LLM_MODEL", "finetune_glm4voice_stage1")
    )
    LLM_MODEL_CONFIG = "/VITA-Audio/vita_audio/models/qwen2_mtp_v4_48_3/config_7B_mtp1.json"

    DATA_CONFIG_PATH = os.getenv(
        "DATA_CONFIG_PATH", "/VITA-Audio/configs/sts_finetune_stage1_test.yaml"
    )

    cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{TRAIN_NAME}")
    PRE_OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{TRAIN_NAME}_{cur_date}")
    if os.path.exists(OUTPUT_DIR):
        cmd = f"mv {OUTPUT_DIR} {PRE_OUTPUT_DIR}"
        print(cmd)
        subprocess.run(cmd, shell=True)
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    SEQ_LENGTH = os.getenv("SEQ_LENGTH", "32768")

    DEEPSPEED_CONFIG_PATH = "/VITA-Audio/scripts/deepspeed/ds_config_zero2.json"
    DDP_BACKEND = os.getenv("DDP_BACKEND", "nccl")

    RUN_PATH = "/VITA-Audio/tools/finetune_sts_v4_48_3.py"
    TEXT_AUDIO_INTERVAL_RATIO = os.getenv("TEXT_AUDIO_INTERVAL_RATIO", "1 10 4 10")
    MAX_STEPS = os.getenv("MAX_STEPS", "4000")

    os.makedirs(os.path.join(TRITON_CACHE_DIR, "autotune"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --------------- run ----------------------
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)
    subprocess.run("which torchrun", shell=True)
    # subprocess.run(f"ls -l {RUN_PATH}", shell=True)
    subprocess.run(f"chmod +x {RUN_PATH}", shell=True)

    DISTRIBUTED_ARGS = f"""--standalone --nnodes=1 \
    --nproc_per_node {os.getenv("NPROC_PER_NODE")} \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 34567 \
    """

    # https://docs.pytorch.org/docs/stable/elastic/run.html
    cmd = f"""
PYTHONPATH=$PYTHONPATH:/VITA-Audio:/VITA-Audio/third_party/GLM-4-Voice:/VITA-Audio/third_party/GLM-4-Voice/cosyvoice:/VITA-Audio/third_party/GLM-4-Voice/third_party/Matcha-TTS \
    torchrun {DISTRIBUTED_ARGS} \
    --log_dir {LOGS_DIR} \
    --tee 3 \
    --redirects 3 \
    {RUN_PATH} \
    --log_level info \
    --do_train \
    --overwrite_output_dir \
    --config_name {LLM_MODEL_CONFIG} \
    --tokenizer_name {LLM_MODEL_PATH} \
    --model_name_or_path {LLM_MODEL_PATH} \
    --audio_tokenizer_path {AUDIO_TOKENIZER_MODEL_PATH} \
    --audio_tokenizer_type {AUDIO_TOKENIZER_TYPE} \
    --dataset_name {DATA_CONFIG_PATH} \
    --dataloader_num_workers 8 \
    --dataset_joint true \
    --data_seed 42 \
    --reset_attention_mask \
    --reset_position_ids \
    --create_attention_mask false \
    --create_attention_mask_2d false \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --output_dir {OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_steps {MAX_STEPS} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 2 \
    --learning_rate 1.00e-3 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length {SEQ_LENGTH} \
    --gradient_checkpointing True \
    --deepspeed {DEEPSPEED_CONFIG_PATH} \
    --trust_remote_code False \
    --ddp_timeout 7200 \
    --ddp_backend {DDP_BACKEND} \
    --attn_implementation flash_attention_2 \
    --seed 42 \
    --language-model-freeze \
    --text-audio-interval-ratio {TEXT_AUDIO_INTERVAL_RATIO} \
    {other_args} \
"""
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/VITA-Audio", env=os.environ)

    train_out_dir.commit()


"""
# just debug | 2xA100-80GB for 7B with zero 1/2
IMAGE_GPU=A100-80GB:2 MAX_STEPS=1 modal run src/train/vita_audio/finetune_glm4voice_mtp1_stage1.py

# run train with test data | 2xA100-80GB for 7B with ds zero 1/2
IMAGE_GPU=A100-80GB:2 MAX_STEPS=2 \
    DATA_CONFIG_PATH=/VITA-Audio/configs/sts_finetune_stage1_test.yaml \
    modal run src/train/vita_audio/finetune_glm4voice_mtp1_stage1.py
"""
