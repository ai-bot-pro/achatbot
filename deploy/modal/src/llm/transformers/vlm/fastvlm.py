import os
import subprocess
from threading import Thread
from time import perf_counter


import modal


app = modal.App("fastvlm")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/ml-fastvlm.git",
        "cd /ml-fastvlm && git checkout 6012455cb7723d6be90cb9eaf462325e6d4a1849",
        "cd /ml-fastvlm && pip install -e .",
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
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "llava-fastvithd_1.5b_stage3"),
            "MOBILE_CLIP_MODEL_CONFIG": "/root/.achatbot/models/mobileclip_l.json",
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


with img.imports():
    import torch
    from PIL import Image
    from transformers.generation.streamers import BaseStreamer, TextIteratorStreamer

    from llava.utils import disable_torch_init
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )

    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func()


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


def dump_model():
    for model_name in [
        "llava-fastvithd_0.5b_stage3",
        "llava-fastvithd_1.5b_stage3",
        "llava-fastvithd_7b_stage3",
    ]:
        model_path = os.path.join(HF_MODEL_DIR, model_name)
        # Load model
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        print(model_name)
        device = "mps" if torch.backends.mps.is_available() else "cuda"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, device=device
        )
        model = model.eval()
        print(f"{model.config=}")
        print(f"{tokenizer=}")
        print_model_params(model, f"LlavaQwen2ForCausalLM_{model_name}")

        del model
        torch.cuda.empty_cache()


# https://github.com/apple/ml-fastvit
# https://github.com/apple/ml-mobileclip
# https://github.com/apple/ml-fastvlm
def predict():
    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)
    print(f"{model_name=}")
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"device {device}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device=device
    )
    model = model.eval()
    print(f"model device {model.device}")
    print(f"{model.config=}")
    print(f"{tokenizer=}")
    # print(f"{model=}")

    # Construct prompt
    qs = "请用中文描述图片内容"
    qs = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 描述下图片内容"
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv_mode = "qwen_2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"{prompt=}")

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(torch.device(device))
    )

    image_file = "/ml-fastvlm/docs/acc_vs_latency_qwen-2.png"
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)


# https://github.com/apple/ml-fastvit
# https://github.com/apple/ml-mobileclip
# https://github.com/apple/ml-fastvlm
def predict_stream():
    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)
    print(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device=device
    )
    model = model.eval()
    print(f"{model.config=}")
    print(f"{tokenizer=}")
    print(f"{model=}")

    # Construct prompt
    qs = "Please reply to my message in Chinese simplified(简体中文), Do not use Markdown format. 描述下图片内容"
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv_mode = "qwen_2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"{prompt=}")
    print(f"{conv.messages=}")

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(torch.device(device))
    )
    print(input_ids)

    image_file = "/ml-fastvlm/docs/acc_vs_latency_qwen-2.png"
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    print(f"{image_tensor.shape=}")

    for i in range(2):
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.half().to(torch.device(device)),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            repetition_penalty=1.5,
            max_new_tokens=1024,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        for new_text in streamer:
            times.append(perf_counter() - start)
            print(new_text, end="", flush=True)
            generated_text += new_text
            start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


"""
IMAGE_GPU=T4 modal run src/llm/transformers/vlm/fastvlm.py --task dump_model
IMAGE_GPU=T4 modal run src/llm/transformers/vlm/fastvlm.py --task predict
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "predict": predict,
        "predict_stream": predict_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
