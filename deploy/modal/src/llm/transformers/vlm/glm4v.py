# author: weedge (weege007@gmail.com)

import os
import subprocess
from threading import Thread
from time import perf_counter
from typing import Optional

import modal


app = modal.App("GLM4.1-VL")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    )
    .pip_install("wheel", "packaging")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers@17b3c96c00cd8421bff85282aec32422bdfebd31"
    )
    .pip_install("accelerate", "av")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            # "LLM_MODEL": os.getenv("LLM_MODEL", "THUDM/GLM-4.1V-9B-Base"),
            "LLM_MODEL": os.getenv("LLM_MODEL", "THUDM/GLM-4.1V-9B-Thinking"),
            "ROUND": os.getenv("ROUND", "1"),
            "IS_OUTPUT_THINK": os.getenv("IS_OUTPUT_THINK", "1"),
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


with img.imports():
    import torch
    from transformers import Glm4vForConditionalGeneration, AutoProcessor

    from transformers.generation.streamers import TextIteratorStreamer


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
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
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func(gpu_prop, thinking)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


def dump_model(gpu_prop, thinking):
    """
    vlm text model use qwen3 arch no MTP
    """
    for model_name in [
        "THUDM/GLM-4.1V-9B-Base",
        "THUDM/GLM-4.1V-9B-Thinking",
    ]:
        MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            use_fast=True,
            trust_remote_code=True,
        )
        model = Glm4vForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
            trust_remote_code=True,
        ).to("cuda")

        model = model.eval()
        print(f"{model.config=}")
        print(f"{processor=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


@torch.inference_mode()
def predict_text(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")
    model = model.eval()
    tokenizer = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    # predict with no instruct tpl, use base model | sft-it | rl-it
    text = "你叫什么名字？"
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
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
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")
    model = model.eval()
    tokenizer = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

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
    )
    print(f"{prompt=}")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
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
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct prompt
    # text = "请用中文描述图片内容"
    text = f"请用中文描述图片内容，不要使用特殊字符回复。"

    # don't to chat with smolvlm, just do vision task
    # text = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 你好"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
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
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode
def predict_stream(gpu_prop, thinking):
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")

    # Construct prompt
    # text = "请用中文描述图片内容"
    text = f"请用中文描述图片内容，不要使用特殊字符回复。"
    # text = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 描述下图片内容"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
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
        print("---" * 20)
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        # don't to do sampling
        generation_kwargs = dict(
            **inputs,
            do_sample=False,
            # do_sample=True,
            # temperature=0.2,
            # top_p=None,
            # num_beams=1,
            # repetition_penalty=1.5,
            max_new_tokens=1024,
            use_cache=True,
            streamer=streamer,
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
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()
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
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "画面中，几朵粉色的花朵在绿意环绕的环境中绽放，其中一朵花上停着一只蜜蜂，蜜蜂正忙碌地在花蕊间采蜜。花朵有的盛开，花瓣舒展，花蕊明黄；有的已凋谢，花瓣枯萎卷曲。画面中还能看到红色的花朵点缀其间，背景是模糊的绿色植物与草地，整体呈现出自然花园里生机与凋零交织的景象，色彩柔和，充满生活气息。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "图片中有几只蜜蜂"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一只蜜蜂。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
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

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def text_vision_chat(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()
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
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "图片中有几只蜜蜂"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一只蜜蜂。",  # remove think content
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
    inputs = processor.apply_chat_template(
        messages[:round],
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

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def text_video_chat(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # from meigen-multitalk generated video
    video_file = os.path.join(VIDEO_OUTPUT_DIR, "multi_long_exp.mp4")
    print(video_file)

    # Construct history chat messages
    text = "讲一个故事"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个视频描述助手，你可以用中文描述视频,图片内容，不要使用特殊字符回复。",
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
                    "text": "我是视频描述助手。",  # remove think content
                }
            ],
        },
        # {
        #    "role": "user",
        #    "content": [
        #        {
        #            "type": "image",
        #            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        #        },
        #        {"type": "text", "text": "图片中有几只蜜蜂"},
        #    ],
        # },
        # {
        #    "role": "assistant",
        #    "content": [
        #        {
        #            "type": "text",
        #            "text": "图片中有一只蜜蜂。",  # remove think content
        #        }
        #    ],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_file,
                    # "max_pixels": 480 * 480,
                    # "fps": 1.0,
                },
                {"type": "text", "text": "描述这个视频"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "画面展示了一个录音室场景，一位穿着浅灰色衬衫的男子和一位身着无袖亮片连衣裙的女子面对面站在专业麦克风前。两人身体微微前倾，靠近麦克风，面部表情专注且投入，似乎在共同演唱。录音室环境有黑色的隔音墙面（带有凸起吸音结构）和左侧的木质窗帘，整体氛围专业且温馨，呈现出两人合作演唱的画面。",  # remove think content
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
    # In Keye-VL, frame rate information is also input into the model to align with absolute time.
    inputs = processor.apply_chat_template(
        messages[:round],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device, dtype=torch.bfloat16)

    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_tool(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()

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
    if thinking is None:
        text += "/agentic_think"
    if thinking is True:
        text += "/think"
    if thinking is False:
        text += "/no_think"
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

    inputs = processor.apply_chat_template(
        messages,
        tools=tools,
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
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_json_mode(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()

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
    if thinking is True:
        text += "/think"
    if thinking is False:
        text += "/no_think"
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
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


"""
https://huggingface.co/THUDM/GLM-4.1V-9B-Base
https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking

TIPS:
- Base model processor does not have a chat template, use it to post training sft and rl
- Thinking model processor have chat template(think IT)

if want display thinking,see:
- https://github.com/THUDM/GLM-4.1V-Thinking/issues/38 (change chat template)
- https://github.com/THUDM/GLM-4.1V-Thinking/issues/22 (2 infer)

# download model
modal run src/download_models.py --repo-ids "GLM-4.1V-9B-Base"
modal run src/download_models.py --repo-ids "GLM-4.1V-9B-Thinking"
modal run src/download_models.py --repo-ids "THUDM/GLM-4.1V-9B-Thinking" --revision "refs/pr/9"

# dump model
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task dump_model

# text model
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task predict_text
IMAGE_GPU=L4 LLM_MODEL=THUDM/GLM-4.1V-9B-Base modal run src/llm/transformers/vlm/glm4v.py --task predict_text
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task chat_text

# vision text/image chat
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task predict 
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task predict_stream
IMAGE_GPU=L40s modal run src/llm/transformers/vlm/glm4v.py --task chat 

ROUND=1 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_vision_chat
ROUND=2 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_vision_chat
ROUND=3 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_vision_chat

# vision text/video chat
ROUND=1 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_video_chat
ROUND=2 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_video_chat
ROUND=3 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_video_chat
#ROUND=4 IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task text_video_chat

# 不支持funciton_calling 需要微调
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task chat_tool

# json_mode支持, GUI agents
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/glm4v.py --task chat_json_mode
"""


@app.local_entrypoint()
def main(task: str = "dump_model", thinking: Optional[bool] = None):
    print(task, thinking)
    tasks = {
        "dump_model": dump_model,
        "predict_text": predict_text,
        "chat_text": chat_text,
        "predict": predict,
        "predict_stream": predict_stream,
        "chat": chat,
        "text_vision_chat": text_vision_chat,
        "text_video_chat": text_video_chat,
        "chat_tool": chat_tool,
        "chat_json_mode": chat_json_mode,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task], thinking)
