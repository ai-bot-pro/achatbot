import datetime
import os
import subprocess

import modal


app = modal.App("demo_train_gsm8k_grpo")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100-80GB")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone https://github.com/weedge/verl.git",
        "cd /verl && pip install -q .",
    )
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # vllm-0.8.3 does not sugrport flashinfer>=0.2.3
    # see https://github.com/vllm-project/vllm/pull/15777
    .pip_install(
        "vllm==0.8.3",
        "flashinfer-python==0.2.2.post1",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .pip_install("tensorboard", "wandb")
    .env(
        {
            "DDP_BACKEND": "nccl",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NPROC_PER_NODE": IMAGE_GPU.split(":")[-1] if len(IMAGE_GPU.split(":")) > 1 else "1",
        }
    )
    .env(
        {
            # RuntimeError: The kernel on this machine does not sugrport the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PYTHONUNBUFFERED": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
            "DATA_PATH": os.getenv("DATA_PATH", "openai/gsm8k"),
        }
    )
    .run_commands(
        "cd /verl && git pull origin feat/achatbot",
        "cd /verl && git checkout dc5740e52f6a48b1eda53569e426e8b6e6fbd89e && pip install -q .",
    )
)

TRAIN_NAME = "demo_grpo_train"

HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
DATASETS_DIR = "/datasets"
datasets_vol = modal.Volume.from_name("datasets", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_vol = modal.Volume.from_name("train_output", create_if_missing=True)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    secrets=[modal.Secret.from_name("achatbot")],
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        DATASETS_DIR: datasets_vol,
        TRAIN_OUTPUT_DIR: train_out_vol,
    },
    timeout=86400,  # [10,86400]
    max_containers=1,
)
def run(
    other_args: str = "",
    total_epochs: int = 2,
    retrain: bool = False,
    strategy: str = "fsdp",
    model_type: str = "fp32",
    tp: int = 1,
    train_batch_size: int = 256,
    ppo_mini_batch_size_per_gpu: int = 60,
    ppo_micro_batch_size_per_gpu: int = 20,
    log_prob_micro_batch_size_per_gpu: int = 8,
):
    import torch

    # --------------- init ----------------------
    NPROC_PER_NODE = os.getenv("NPROC_PER_NODE", "1")
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    LLM_MODEL_PATH = os.path.join(HF_MODEL_DIR, llm_model)

    data_path = os.getenv("DATA_PATH", "openai/gsm8k")
    data_name = data_path.split("/")[-1]
    DATA_PATH = os.path.join(DATASETS_DIR, data_path)

    OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{TRAIN_NAME}")
    cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    PRE_OUTPUT_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{TRAIN_NAME}_{cur_date}")
    if retrain is True and os.path.exists(OUTPUT_DIR):
        cmd = f"mv {OUTPUT_DIR} {PRE_OUTPUT_DIR}"
        print(cmd)
        subprocess.run(cmd, shell=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    TENSORBOARD_LOG = os.path.join(OUTPUT_DIR, "tensorboard_log")
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    # --------------- run ----------------------
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)
    subprocess.run("which torchrun", shell=True)

    # https://verl.readthedocs.io/en/latest/examples/config.html
    # https://github.com/volcengine/verl/blob/main/verl/trainer/config/grpo_trainer.yaml
    cmd = f"""TENSORBOARD_DIR={TENSORBOARD_LOG} PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
data.train_files={DATA_PATH}/train.parquet \
data.val_files={DATA_PATH}/test.parquet \
algorithm.adv_estimator=grpo \
data.train_batch_size={train_batch_size} \
data.max_prompt_length=512 \
data.max_response_length=1024 \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path={LLM_MODEL_PATH} \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=False \
actor_rollout_ref.actor.fsdp_config.model_dtype={model_type} \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size={int(NPROC_PER_NODE) * ppo_mini_batch_size_per_gpu} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu} \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.strategy={strategy} \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch_size_per_gpu} \
actor_rollout_ref.rollout.tensor_model_parallel_size={tp} \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={log_prob_micro_batch_size_per_gpu} \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.ref.strategy={strategy} \
algorithm.use_kl_in_reward=False \
trainer.val_before_train=True \
trainer.logger=['console','tensorboard','wandb'] \
trainer.project_name='verl_grpo_example_{data_name}' \
trainer.experiment_name='{llm_model}_function_rm' \
trainer.default_local_dir={OUTPUT_DIR} \
trainer.n_gpus_per_node={NPROC_PER_NODE} \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.total_epochs={total_epochs} \
{other_args} 2>&1 | tee {LOGS_DIR}/verl_{TRAIN_NAME}.log
"""

    print(cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/verl", env=os.environ)

    train_out_vol.commit()


"""
# actor model_type use fp32, Qwen/Qwen2.5-0.5B-Instruct rl zero with gsm10k
IMAGE_GPU=L40s LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp --retrain
IMAGE_GPU=L40s:2 LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp --retrain

# use fsdp2, actor model_type use fp32, Qwen/Qwen2.5-0.5B-Instruct rl zero with gsm10k
IMAGE_GPU=L40s LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --retrain
IMAGE_GPU=L40s:2 LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --retrain
IMAGE_GPU=L40s LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --total-epochs 15
IMAGE_GPU=L40s:2 LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --total-epochs 15 --retrain --tp 2

# actor model_type use fp32, Qwen/Qwen2.5-3B-Instruct rl zero with gsm10k
modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --retrain
modal run src/train/verl/examples/demo_train_grpo.py --strategy fsdp2 --total-epochs 15
IMAGE_GPU=L40s:2 modal run src/train/verl/examples/demo_train_grpo.py  --strategy fsdp2 --total-epochs 15 --retrain --tp 2
"""
