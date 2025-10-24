import os
import sys
import time
import uuid
import asyncio
import subprocess
from pathlib import Path
from threading import Thread
from PIL import Image

import modal

OCR_TAG = os.getenv("OCR_TAG", "llm_vllm_deepseek_ocr")
BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")  # only one
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
            "OCR_TAG": OCR_TAG,
        }
    )
)

if OCR_TAG == "llm_office_vllm_deepseek_ocr":
    vllm_image = vllm_image.run_commands(
        "pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly"
    ).env(
        {
            "VLLM_USE_V1": "1",
        }
    )

if BACKEND == "flashinfer":
    vllm_image = vllm_image.pip_install(
        f"flashinfer-python==0.2.2.post1",  # FlashInfer 0.2.3+ does not support per-request generators
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6",
    )

if APP_NAME == "achatbot":
    vllm_image = vllm_image.pip_install(
        f"achatbot==0.0.28.post2",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

if OCR_TAG == "llm_office_vllm_deepseek_ocr":
    vllm_image = vllm_image.pip_install("transformers==4.57.1")
else:
    vllm_image = vllm_image.pip_install("transformers==4.47.1")

# vllm_image = vllm_image.pip_install(
#   f"achatbot==0.0.28.post3",
#   extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
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
    import torch
    from tqdm import tqdm
    from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    MODEL_ID = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-OCR")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    DEEPSEEK_ASSETS_DIR = os.path.join(ASSETS_DIR, "DeepSeek-OCR-vllm")

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


def offline_infer(**kwargs):
    ocr_tag = os.getenv("OCR_TAG", "llm_vllm_deepseek_ocr")
    assert ocr_tag == "llm_office_vllm_deepseek_ocr"
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    from PIL import Image

    # Create model instance
    llm = LLM(
        model=MODEL_PATH,
        # block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,  # no chat case
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    # Prepare batched input with your image file
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    # prompt = "<image>\nFree OCR."
    image_files = [
        Image.new("RGB", (640, 640), color="white"),
        Image.open("/DeepSeek-OCR/assets/fig1.png").convert("RGB"),
        # use ORC detected Show pictures, detect again :)
        Image.open("/DeepSeek-OCR/assets/show1.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show2.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show3.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show4.jpg"),
    ]
    model_input = []
    for image_obj in image_files:
        model_input.append({"prompt": prompt, "multi_modal_data": {"image": image_obj}})

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )
    # Generate output
    model_outputs = llm.generate(model_input, sampling_param)

    # Print output
    for output in model_outputs:
        print(output.outputs[0].text)
        print("----" * 20)


async def stream_generate(image=None, prompt=""):
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}
        )
    ]  # whitelist: <td>, </td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        # ignore_eos=False,
    )

    request_id = f"request-{int(time.time())}"

    printed_length = 0

    if image and "<image>" in prompt:
        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
    elif prompt:
        request = {"prompt": prompt}
    else:
        assert False, f"prompt is none!!!"
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end="", flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print("\n")

    return final_output


async def stream_infer(**kwargs):
    sys.path.insert(0, "/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm")
    from process.image_process import DeepseekOCRProcessor
    from run_dpsk_ocr_image import load_image, re_match, process_image_with_refs
    import config

    config.OUTPUT_PATH = DEEPSEEK_ASSETS_DIR

    CROP_MODE = True

    image_path = "/DeepSeek-OCR/assets/fig1.png"
    # image_path = "/DeepSeek-OCR/assets/show1.jpg"
    image = load_image(image_path).convert("RGB")

    PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    # PROMPT = "<image>\nFree OCR."

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
        with open(f"{DEEPSEEK_ASSETS_DIR}/result.md", "w", encoding="utf-8") as afile:
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
                except Exception:
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


async def achatbot_stream_infer(**kwargs):
    from achatbot.processors.vision.ocr_processor import OCRProcessor
    from achatbot.modules.vision.ocr import VisionOCREnvInit
    from achatbot.common.session import SessionCtx, Session
    from achatbot.types.frames.data_frames import UserImageRawFrame
    from achatbot.thirdparty.deepseek_ocr_vllm.model import BASE_SIZE, IMAGE_SIZE, CROP_MODE, PROMPT
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    serv_args = dict(
        model=MODEL_PATH,
        # block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,  # no chat case
    )

    ocr_tag = os.getenv("OCR_TAG", "llm_vllm_deepseek_ocr")
    if ocr_tag == "llm_office_vllm_deepseek_ocr":
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        serv_args["mm_processor_cache_gb"] = 0
        serv_args["logits_processors"] = [NGramPerReqLogitsProcessor]
    else:
        from vllm.model_executor.models.registry import ModelRegistry
        # from achatbot.thirdparty.deepseek_ocr_vllm.model.deepseek_ocr import DeepseekOCRForCausalLM

        # ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        serv_args["hf_overrides"] = {"architectures": ["DeepseekOCRForCausalLM"]}

    ocr = VisionOCREnvInit.initVisionOCREngine(ocr_tag, {"serv_args": serv_args})
    session = Session(**SessionCtx(str(uuid.uuid4())).__dict__)
    processor = OCRProcessor(ocr=ocr, session=session)
    image_files = [
        Image.new("RGB", (640, 640), color="white"),
        # Image.open("/DeepSeek-OCR/assets/fig1.png"),
        # use ORC detected Show pictures, detect again :)
        # Image.open("/DeepSeek-OCR/assets/show1.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show2.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show3.jpg"),
        # Image.open("/DeepSeek-OCR/assets/show4.jpg"),
    ]
    for image_obj in image_files:
        image_obj: Image.Image = image_obj
        frame = UserImageRawFrame(
            image=image_obj.tobytes(),
            size=image_obj.size,
            format=image_obj.format,  # from frame bytes, no save format, need add a save format e.g.: JPEG,PNG,
            mode=image_obj.mode,  # default: RGB
            user_id=session.ctx.client_id,
        )
        iter = processor.run_detect(frame)
        async for textFrame in iter:
            print(textFrame)


"""
IMAGE_GPU=L40s modal run src/llm/vllm/vlm/ocr_deepseek.py --task stream_infer
IMAGE_GPU=L40s OCR_TAG=llm_office_vllm_deepseek_ocr modal run src/llm/vllm/vlm/ocr_deepseek.py --task offline_infer
APP_NAME=achatbot IMAGE_GPU=L40s OCR_TAG=llm_vllm_deepseek_ocr modal run src/llm/vllm/vlm/ocr_deepseek.py --task achatbot_stream_infer
APP_NAME=achatbot IMAGE_GPU=L40s OCR_TAG=llm_office_vllm_deepseek_ocr modal run src/llm/vllm/vlm/ocr_deepseek.py --task achatbot_stream_infer
"""


@app.local_entrypoint()
def main(task: str = "stream_infer"):
    tasks = {
        "stream_infer": stream_infer,
        "offline_infer": offline_infer,
        "achatbot_stream_infer": achatbot_stream_infer,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
