import os
import sys
import time
import asyncio
import subprocess
from pathlib import Path
from threading import Thread


import modal

BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")
PROFILE_DIR = "/root/vllm_profile"

app = modal.App("vllm-deepseek-ocr")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
vllm_image = (
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
    .pip_install("vllm==v0.8.5", extra_index_url="https://download.pytorch.org/whl/cu126")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR"),
            "VLLM_USE_V1": "0",
            "VLLM_TORCH_PROFILER_DIR": PROFILE_DIR,
            "TP": TP,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX",
        }
    )
)

if BACKEND == "flashinfer":
    vllm_image = vllm_image.pip_install(
        f"flashinfer-python==0.2.2.post1",  # FlashInfer 0.2.3+ does not support per-request generators
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6",
    )

if APP_NAME == "achatbot":
    vllm_image = vllm_image.pip_install(
        f"achatbot==0.0.28",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
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

VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)


with vllm_image.imports():
    sys.path.insert(0, "/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")

    import torch
    from tqdm import tqdm
    from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    MODEL_ID = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    DEEPSEEK_ASSETS_DIR = os.path.join(ASSETS_DIR, "DeepSeek-OCR-vllm")

    import config

    config.OUTPUT_PATH = DEEPSEEK_ASSETS_DIR

    from process.image_process import DeepseekOCRProcessor
    from run_dpsk_ocr_image import load_image, re_match, process_image_with_refs, stream_generate

    # torch.set_float32_matmul_precision("high")


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=vllm_image,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    os.makedirs(DEEPSEEK_ASSETS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEEPSEEK_ASSETS_DIR, "images"), exist_ok=True)
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


async def stream_infer(**kwargs):
    CROP_MODE = True

    image = load_image("/DeepSeek-OCR/assets/fig1.png").convert("RGB")

    PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    if "<image>" in PROMPT:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )
    else:
        image_features = ""

    prompt = PROMPT

    result_out = await stream_generate(image_features, prompt)

    save_results = 1

    if save_results and "<image>" in prompt:
        print("=" * 15 + "save results:" + "=" * 15)

        image_draw = image.copy()

        outputs = result_out

        with open(f"{DEEPSEEK_ASSETS_DIR}/result_ori.mmd", "w", encoding="utf-8") as afile:
            afile.write(outputs)

        matches_ref, matches_images, matches_other = re_match(outputs)
        # print(matches_ref)
        # save images with boxes
        result = process_image_with_refs(image_draw, matches_ref)

        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, f"![](images/" + str(idx) + ".jpg)\n")

        for idx, a_match_other in enumerate(tqdm(matches_other, desc="other")):
            outputs = (
                outputs.replace(a_match_other, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
            )

        # if 'structural formula' in conversation[0]['content']:
        #     outputs = '<smiles>' + outputs + '</smiles>'
        with open(f"{DEEPSEEK_ASSETS_DIR}/result.mmd", "w", encoding="utf-8") as afile:
            afile.write(outputs)

        if "line_type" in outputs:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            lines = eval(outputs)["Line"]["line"]

            line_type = eval(outputs)["Line"]["line_type"]
            # print(lines)

            endpoints = eval(outputs)["Line"]["line_endpoint"]

            fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(" -- ")[0])
                    p1 = eval(line.split(" -- ")[-1])

                    if line_type[idx] == "--":
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")

                    ax.scatter(p0[0], p0[1], s=5, color="k")
                    ax.scatter(p1[0], p1[1], s=5, color="k")
                except:
                    pass

            for endpoint in endpoints:
                label = endpoint.split(": ")[0]
                (x, y) = eval(endpoint.split(": ")[1])
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(1, 1),
                    textcoords="offset points",
                    fontsize=5,
                    fontweight="light",
                )

            try:
                if "Circle" in eval(outputs).keys():
                    circle_centers = eval(outputs)["Circle"]["circle_center"]
                    radius = eval(outputs)["Circle"]["radius"]

                    for center, r in zip(circle_centers, radius):
                        center = eval(center.split(": ")[1])
                        circle = Circle(
                            center, radius=r, fill=False, edgecolor="black", linewidth=0.8
                        )
                        ax.add_patch(circle)
            except Exception:
                pass

            plt.savefig(f"{DEEPSEEK_ASSETS_DIR}/geo.jpg")
            plt.close()

        result.save(f"{DEEPSEEK_ASSETS_DIR}/result_with_boxes.jpg")


"""
IMAGE_GPU=L40s modal run src/llm/vllm/vlm/ocr_deepseek.py --task stream_infer
"""


@app.local_entrypoint()
def main(task: str = "stream_infer"):
    tasks = {
        "stream_infer": stream_infer,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
