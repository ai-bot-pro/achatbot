# author: weedge (weege007@gmail.com)

import io
import math
import os
import re
import subprocess
from threading import Thread
from time import perf_counter
from typing import Optional
import uuid


import modal
import requests

APP_NAME = os.getenv("APP_NAME", "")
app = modal.App("skywork_r1-VL")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "transformers",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    )
    .pip_install("wheel", "packaging")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    # .run_commands(
    #    "pip install git+https://github.com/huggingface/transformers@17b3c96c00cd8421bff85282aec32422bdfebd31"
    # )
    .pip_install("accelerate", "av", "timm")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Skywork/Skywork-R1V3-38B"),
            "ROUND": os.getenv("ROUND", "1"),
            "IS_OUTPUT_THINK": os.getenv("IS_OUTPUT_THINK", "1"),
            "IMAGE_GPU": os.getenv("IMAGE_GPU", ""),
            "TEST_CHAT": os.getenv("TEST_CHAT", ""),
        }
    )
)

if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.21.post1",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


with img.imports():
    from PIL import Image
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from transformers.generation.streamers import TextIteratorStreamer


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func, thinking):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func(gpu_prop, thinking)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def dump_model(gpu_prop, thinking):
    """
    vlm text model use Skywork/Skywork-R1V3-38B
    """
    for model_name in [
        "Skywork/Skywork-R1V3-38B",
    ]:
        MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
        processor = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"{type(processor)=}", processor)

        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True if gpu_prop.major >= 8 else False,
            # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        )
        model = model.eval()
        print(f"{model.config=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


def get_prompt(conv_template, messages, thinking=True):
    assert len(messages) % 2 == 0
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"

    for i, message in enumerate(messages):
        assert message.get("content")
        assert len(message.get("content")) > 0
        print(f"{message=}")

        if message["role"] == "system":
            if message["content"][0]["text"]:
                conv_template.system_message = message["content"][0]["text"]
        elif message["role"] == "user":
            query = ""
            text = ""
            for item in message["content"]:
                if item["type"] == "text":
                    text = item["text"]
                if item["type"] == "image":
                    query += "<image>\n"
            query += text
            if i == len(messages) - 1:
                query = query.replace("<image>", "<IMAGE>")
            conv_template.append_message("user", query)
        elif message["role"] == "assistant":
            answer = ""
            for item in message["content"]:
                if item["type"] == "text":
                    answer += item["text"]
            print(f"{answer=}")
            conv_template.append_message("assistant", answer)
    conv_template.append_message("assistant", None)

    prompt = conv_template.get_prompt()
    if not prompt.endswith("\n<think>"):
        prompt += "\n<think>"
    if thinking is False:
        prompt = re.sub(r"\n<think>", "", prompt, count=1)

    return prompt


def tokenize(gpu_prop, thinking=True):
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # https://huggingface.co/Skywork/Skywork-R1V3-38B/blob/main/conversation.py
    print(f"{model.conv_template.sep=}")

    chat_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个中文的AI，你回答问题的时候不要使用特殊字符，不要使用英文。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图中展示了一个撕裂效果的画面 外面是橙黄色的背景 内部展现了一位裹着彩色条纹毯子 红头发的背对观众的人物 她坐在开满粉红色花朵的山坡上 前方是连绵的山谷和山脉 远处天空中有阳光洒落",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg"),
                },
                {"type": "text", "text": "图中有几个人"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一个人。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字？"},
            ],
        },
    ]

    for i in range(1, 4):
        round = i * 2
        messages = chat_messages[:round]
        tpl = model.conv_template.copy()
        prompt = get_prompt(tpl, messages, thinking)
        print(f"{round=} {prompt=}")

        pixel_values = []
        for item in messages[-1]["content"]:
            if item["type"] == "image":
                image_file = item["image"]
                if item["image"].startswith("https://") or item["image"].startswith("http://"):
                    image_file = io.BytesIO(requests.get(item["image"]).content)
                pixel_values.append(load_image(image_file))
        num_patches_list = [img.size(0) for img in pixel_values]
        # print(pixel_values, num_patches_list)

        # replace <image> placeholder
        # NOTE: just process curr user image query
        for num_patches in num_patches_list:
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
            )
            prompt = prompt.replace("<IMAGE>", image_tokens, 1)

        print(f"{round=} {prompt=}")


def test(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=False,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    eos_token_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    # https://huggingface.co/Skywork/Skywork-R1V3-38B/blob/main/conversation.py
    print(f"{model.conv_template.sep=}")
    model.conv_template.roles = ["user", "assistant"]
    model.conv_template.system_message = (
        "你是一个中文的AI，你回答问题的时候不要使用特殊字符，不要使用英文。"
    )

    text = f"请用中文描述图片内容，不要使用特殊字符回复。"
    # image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    # image="https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
    # image_file = io.BytesIO(requests.get(image).content)
    text = "<image>\n" + text

    image_file = os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")
    pixel_values = load_image(image_file).to(torch.bfloat16).to(model.device)
    num_patches_list = [pixel_values.shape[0]]
    print(f"{pixel_values.shape=}")

    test_chat = os.getenv("TEST_CHAT")
    if test_chat:
        generation_config = dict(
            max_new_tokens=64000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
        )
        # https://huggingface.co/Skywork/Skywork-R1V3-38B/blob/main/modeling_skywork_chat.py#L253
        response = model.chat(
            tokenizer,
            pixel_values,
            text,
            generation_config,
            num_patches_list=num_patches_list,
            mode="no-think" if thinking is False else "think",
        )
        print(response)
    else:
        model.conv_template.append_message(model.conv_template.roles[0], text)
        model.conv_template.append_message(model.conv_template.roles[1], None)
        prompt = model.conv_template.get_prompt()
        if thinking is False:
            prompt = re.sub(r"\n<think>", "", prompt, count=1)

        # replace <image> placeholder
        # NOTE: just process curr user image query
        for num_patches in num_patches_list:
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
            )
            prompt = prompt.replace("<image>", image_tokens, 1)

        inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
        for key, value in inputs.items():
            print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
                f"{key}: {value} {value.dtype=}"
            )
        input_ids = inputs["input_ids"]
        prompt = tokenizer.decode(input_ids[0])
        print(f"{prompt=}")

        generated_ids = model.generate(
            **inputs,
            pixel_values=pixel_values,
            max_new_tokens=64000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
        )
        print(f"{generated_ids.shape=}")
        generated_text = tokenizer.decode(
            generated_ids[0],
            # skip_special_tokens=True,
        )
        print(generated_text)


@torch.inference_mode()
def predict_text(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # predict with no instruct tpl, use base model | sft-it | rl-it
    text = "你叫什么名字？"
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    # https://huggingface.co/Skywork/Skywork-R1V3-38B/blob/main/modeling_skywork_chat.py#L316
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=128,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_text(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # Construct prompt
    text = "你叫什么名字？"
    messages = [
        {
            "role": "system",
            "content": "你是一个非常棒的聊天助手，不要使用特殊字符回复。",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    # use instruct sft | rl model
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def predict(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct prompt
    # text = "请用中文描述图片内容"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
    ]

    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        pixel_values=pixel_values,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.no_grad()
def predict_stream(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct prompt
    # text = "请用中文描述图片内容"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
    ]

    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    eos_token_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    for i in range(3):
        print("---" * 20)
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        temperature = 0.6
        generation_kwargs = dict(
            **inputs,
            pixel_values=pixel_values,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            # top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=1024,
            # use_cache=True,
            streamer=streamer,
            eos_token_id=eos_token_id,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        is_output_think = os.getenv("IS_OUTPUT_THINK", "1") == "1"
        for new_text in streamer:
            times.append(perf_counter() - start)
            print(new_text, end="", flush=True)
            if is_output_think is False:
                if "</think>" in new_text:
                    new_text = new_text.replace("</think>", "").strip("\n")
                    is_output_think = True
                else:
                    continue
            generated_text += new_text
            start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


@torch.inference_mode()
def chat(gpu_prop, thinking):
    """
    multi images need more GPU, have some bug
    """
    # Load model
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct history chat messages
    text = "讲一个故事"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    # 高分辨率的图片需要更多的GPU BHM
                    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图中展示了一个撕裂效果的画面 外面是橙黄色的背景 内部展现了一位裹着彩色条纹毯子 红头发的背对观众的人物 她坐在开满粉红色花朵的山坡上 前方是连绵的山谷和山脉 远处天空中有阳光洒落",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    # 高分辨率的图片需要更多的GPU BHM
                    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {"type": "text", "text": "图片中有几个人"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一个人。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                },
                {
                    "type": "image",
                    # 高分辨率的图片需要更多的GPU BHM
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": text},
            ],
        },
    ]

    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        pixel_values=pixel_values,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def text_vision_chat(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct history chat messages
    text = "讲一个故事"
    chat_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "我是图片描述助手。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                },
                {"type": "text", "text": "图片中有几个人"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一个人。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    ]

    round = int(os.getenv("ROUND", "1")) * 2 if int(os.getenv("ROUND", "1")) > 0 else 2
    messages = chat_messages[:round]
    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        pixel_values=pixel_values,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_tool(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    text = "北京的天气"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个智能助手，不要使用特殊字符回复。"
                    "\n 提供的工具如下：\n"
                    "\n Tool: get_weather \n"
                    "\n Description: Get weather of an location, the user shoud supply a location first \n"
                    "\n Arguments: location\n\n"
                    "\n 根据用户的问题选择合适的工具。如果不需要工具，直接回复。\n",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                # {
                #    "type": "image",
                #    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                # },
                {"type": "text", "text": text},
            ],
        },
    ]

    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    eos_token_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        pixel_values=pixel_values,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
        eos_token_id=eos_token_id,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_json_mode(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    # gpu = os.getenv("IMAGE_GPU")
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    device_map = split_model(MODEL_PATH)
    print(device_map)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True if gpu_prop.major >= 8 else False,
        device_map=device_map,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    system_prompt = """
    The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 
    
    EXAMPLE INPUT: 
    Which is the highest mountain in the world? Mount Everest.
    
    EXAMPLE JSON OUTPUT:
    {
        "question": "Which is the highest mountain in the world?",
        "answer": "Mount Everest"
    }
    """
    text = "Which is the longest river in the world? The Nile River."
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                # {
                #    "type": "image",
                #    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                # },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        },
    ]

    tpl = model.conv_template.copy()
    prompt = get_prompt(tpl, messages, thinking)
    print(f"{round=} {prompt=}")

    pixel_values = []
    for item in messages[-1]["content"]:
        if item["type"] == "image":
            image_file = item["image"]
            if item["image"].startswith("https://") or item["image"].startswith("http://"):
                image_file = io.BytesIO(requests.get(item["image"]).content)
            pixel_values.append(load_image(image_file))
    num_patches_list = [img.size(0) for img in pixel_values]
    pixel_values = (
        torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(model.device)
        if len(pixel_values) > 0
        else None
    )
    pixel_values is not None and print(f"{pixel_values.shape=}")

    # replace <image> placeholder
    # NOTE: just process curr user image query
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        prompt = prompt.replace("<IMAGE>", image_tokens, 1)

    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value} {value.dtype=}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        pixel_values=pixel_values,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0],
        # skip_special_tokens=True,
    )
    print(generated_text)


def achatbot_generate(gpu_prop, thinking):
    from achatbot.core.llm.transformers.manual_vision_skyworkr1v import (
        TransformersManualVisionSkyworkR1V,
    )
    from achatbot.types.llm.transformers import TransformersLMArgs
    from achatbot.common.types import MODELS_DIR, SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    model = TransformersManualVisionSkyworkR1V(
        **TransformersLMArgs(
            lm_model_name_or_path=PATH,
            init_chat_prompt="你是一个中文语音智能助手，不要使用特殊字符回复，请使用中文回复。",
            lm_device="cuda",
            lm_gen_temperature=0.6,
            lm_gen_thinking=thinking,
            lm_gen_repetition_penalty=1.1,
        ).__dict__
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
    image_bytes = requests.get(url).content
    img = Image.open(io.BytesIO(image_bytes))
    # chat_texts = ["这张图片的内容是什么", "你叫什么名字", "讲个故事"]
    chat_texts = ["这张图片的内容是什么"]
    for chat_text in chat_texts:
        session.ctx.state["prompt"] = [
            {"type": "image", "image": img},
            {"type": "text", "text": chat_text},
        ]
        for result_text in model.generate(session, thinking=thinking):
            print(result_text, flush=True, end="")
    img.close()


"""
https://huggingface.co/Skywork/Skywork-R1V3-38B

- default thinking mode

# 0. download model
modal run src/download_models.py --repo-ids "Skywork/Skywork-R1V3-38B"

# 1. dump model
IMAGE_GPU=L40s modal run src/llm/transformers/vlm/skywork_r1v.py --task dump_model
IMAGE_GPU=None modal run src/llm/transformers/vlm/skywork_r1v.py --task tokenize

# test
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task test thinking
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task test --no-thinking

# 2. text model
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task predict_text
IMAGE_GPU=A100-80GB LLM_MODEL=THUDM/GLM-4.1V-9B-Base modal run src/llm/transformers/vlm/skywork_r1v.py --task predict_text
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task chat_text

# 3. vision text/image chat case need 68GB GPU HBM
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict 
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict --no-thinking
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict --thinking
IMAGE_GPU=L40s:2 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict 
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task predict 
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task predict --no-thinking
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task predict --thinking

IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict_stream
IMAGE_GPU=L40s:2 modal run src/llm/transformers/vlm/skywork_r1v.py --task predict_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task predict_stream

# multi images
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task chat 

ROUND=1 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task text_vision_chat
ROUND=2 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task text_vision_chat
ROUND=3 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task text_vision_chat


# 5. 不支持funciton_calling 需要微调
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task chat_tool

# 6. json_mode支持不够好, 输出markdown格式```json {xxx} ``` xxx 需要截断/微调
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task chat_json_mode

# 7. use achatbot to generate
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task achatbot_generate
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task achatbot_generate --no-thinking
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/skywork_r1v.py --task achatbot_generate --thinking
"""


@app.local_entrypoint()
def main(task: str = "dump_model", thinking: Optional[bool] = None):
    print(task, thinking)
    tasks = {
        "dump_model": dump_model,
        "tokenize": tokenize,
        "test": test,
        "predict_text": predict_text,
        "chat_text": chat_text,
        "predict": predict,
        "predict_stream": predict_stream,
        "chat": chat,
        "text_vision_chat": text_vision_chat,
        "chat_tool": chat_tool,
        "chat_json_mode": chat_json_mode,
        "achatbot_generate": achatbot_generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task], thinking)
