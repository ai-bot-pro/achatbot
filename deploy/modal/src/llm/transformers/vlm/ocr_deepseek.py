import os
import sys
import asyncio
import subprocess
from pathlib import Path
from threading import Thread


import modal


app = modal.App("deepseek-ocr")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/deepseek-ai/DeepSeek-OCR.git",
        "cd /DeepSeek-OCR && pip install -r requirements.txt",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "wheel",
        "matplotlib",
        "accelerate>=0.26.0",  # for device_map="auto" model loading with safetensors slipt
    )
    # .apt_install("clang", "cmake", "ninja-build")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1 for torch 2.6.0
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR"),
        }
    )
)

# img = img.pip_install(
#    f"achatbot==0.0.25.dev122",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)

TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)


with img.imports():
    from queue import Queue

    import torch
    from transformers.generation.streamers import TextStreamer
    from transformers import AutoModel, AutoTokenizer

    MODEL_ID = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    DEEPSEEK_ASSETS_DIR = os.path.join(ASSETS_DIR, "DeepSeek")

    # torch.set_float32_matmul_precision("high")


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    os.makedirs(DEEPSEEK_ASSETS_DIR, exist_ok=True)
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def dump_model(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("tokenizer.eos_token_id", tokenizer.eos_token_id)
    print("Tokenizer:", tokenizer)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "auto",  # need accelerate>=0.26.0
    )
    model = model.eval()

    print_model_params(model, extra_info="DeepSeek-OCR", f=sys.stdout)


def infer(**kwargs):
    class NoEOSTextStreamer(TextStreamer):
        def on_finalized_text(self, text: str, stream_end: bool = False):
            stream_end and print("stream_end is True", flush=True)
            eos_text = self.tokenizer.decode(
                [self.tokenizer.eos_token_id], skip_special_tokens=False
            )
            text = text.replace(eos_text, "\n")
            print(text, flush=True, end="")

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # prompt = "<image>\nFree OCR. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_files = [
        "/DeepSeek-OCR/assets/fig1.png",
        # use ORC detected Show pictures, detect again :)
        # "/DeepSeek-OCR/assets/show1.jpg",
        # "/DeepSeek-OCR/assets/show2.jpg",
        # "/DeepSeek-OCR/assets/show3.jpg",
        # "/DeepSeek-OCR/assets/show4.jpg",
    ]

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False
    # Gundam: base_size = 1024, image_size = 640, crop_mode = True # default

    for image_file in image_files:
        print("infer image_file:", image_file)
        # https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py#L703
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=DEEPSEEK_ASSETS_DIR,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True,
            eval_mode=False,
            streamer=NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
        )
        print("infer result:", res)


"""
modal run src/download_models.py --repo-ids "deepseek-ai/DeepSeek-OCR" --revision "refs/pr/23"

IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek.py --task dump_model
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek..py --task infer
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "infer": infer,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
