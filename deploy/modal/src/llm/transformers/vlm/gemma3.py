import os
import subprocess
from threading import Thread
from time import perf_counter


import modal


app = modal.App("gemma3")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "transformers",
        "accelerate",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "google/gemma-3-4b-it"),
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


with img.imports():
    import torch
    from transformers import pipeline

    from PIL import Image
    from transformers import (
        AutoProcessor,
        Gemma3ForConditionalGeneration,
        AutoModelForImageTextToText,
    )
    from transformers.generation.streamers import TextIteratorStreamer


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

    func(gpu_prop)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


# gemama https://arxiv.org/abs/2403.08295
# gemama2 https://arxiv.org/abs/2408.00118
# gemma3 https://arxiv.org/abs/2503.19786
# https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
def dump_model(gpu_prop):
    for model_name in [
        # "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ]:
        model_path = os.path.join(HF_MODEL_DIR, model_name)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        ).to("cuda")

        model = model.eval()
        print(f"{model.config=}")
        # processor = AutoProcessor.from_pretrained(model_path)
        # print(f"{processor=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


def predict(gpu_prop):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")
    print_model_params(model, f"{model_name}")

    text = "Describe this image in detail."
    text = "请用中文描述下图片内容"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    print(messages)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            top_k=1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=256,
        )
        generated_ids = generated_ids[0][input_len:]

    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids,
        # skip_special_tokens=True,
    )
    print(generated_text)


def predict_stream(gpu_prop):
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")

    model = model.eval()
    print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")

    # Construct prompt
    image_file = os.path.join(ASSETS_DIR, "bee.jpg")
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")

    text = "Describe this image in detail."
    text = "请用中文描述下图片内容"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    for i in range(3):
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            do_sample=True,
            temperature=0.2,
            top_k=10,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=256,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        with torch.inference_mode():
            for new_text in streamer:
                times.append(perf_counter() - start)
                print(new_text, end="", flush=True)
                generated_text += new_text
                start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


def predict_med(gpu_prop):
    import requests

    model_id = "google/medgemma-4b-it"
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_id)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "你是一名放射科医生"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这个 X-射线图像"},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)


def predict_med_stream(gpu_prop):
    # Load model
    import requests

    model_id = "google/medgemma-4b-it"
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_id)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一名放射科医生,进行语音聊天咨询， 你会进行诊断，并给出建议,言简意赅。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这个X-射线图像"},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    for i in range(3):
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            do_sample=True,
            temperature=0.2,
            top_k=10,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=256,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        with torch.inference_mode():
            for new_text in streamer:
                times.append(perf_counter() - start)
                print(new_text, end="", flush=True)
                generated_text += new_text
                start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


"""
Gemma 3 Medical:
Text only: 27b
Multimodal: 4b

Gemma 3:
Text only: 1b
Multimodal: 4b, 12b, 27b

Gemma 2:
Text only: 2b-v2, 9b, 27b

Gemma:
Text only: 2b, 7b


    Raw (GB)	    Quantized (GB)
Model	bf16	Int4	Int4(blocks=32) SFP8
1B	    2.0	    0.5	    0.7	            1.0
+KV 	2.9	    1.4	    1.6	            1.9
4B	    8.0	    2.6	    2.9	            4.4
+KV 	12.7	7.3	    7.6	            9.1
12B 	24.0	6.6	    7.1	            12.4
+KV 	38.9	21.5	22.0	        27.3
27B 	54.0	14.1	15.3	        27.4
+KV 	72.7	32.8	34.0	        46.1

IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/gemma3.py --task dump_model

IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict_stream

LLM_MODEL=google/gemma-3-4b-it-qat-q4_0-unquantized IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict
LLM_MODEL=google/gemma-3-4b-it-qat-q4_0-unquantized IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict_stream

LLM_MODEL=google/gemma-3-12b-it IMAGE_GPU=L40s modal run src/llm/transformers/vlm/gemma3.py --task predict
LLM_MODEL=google/gemma-3-12b-it IMAGE_GPU=L40s modal run src/llm/transformers/vlm/gemma3.py --task predict_stream

LLM_MODEL=google/gemma-3-27b-it-qat-q4_0-unquantized IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/gemma3.py --task predict_stream
LLM_MODEL=google/gemma-3-27b-it IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/gemma3.py --task predict
LLM_MODEL=google/gemma-3-27b-it IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/gemma3.py --task predict_stream

# medgemma
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict_med
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/gemma3.py --task predict_med_stream
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "predict": predict,
        "predict_stream": predict_stream,
        "predict_med": predict_med,
        "predict_med_stream": predict_med_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
