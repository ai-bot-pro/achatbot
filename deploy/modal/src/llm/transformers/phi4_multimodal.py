import io
import requests
import os
import subprocess
from threading import Thread
from time import perf_counter


import modal


app = modal.App("phi4_multimodal")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "transformers==4.48.2",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("soundfile", "accelerate", "pillow", "scipy", "backoff", "peft")
    .pip_install("qwen_omni_utils")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct"),
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
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from transformers.generation.streamers import TextIteratorStreamer
    from qwen_omni_utils import process_mm_info

    # Define prompt structure
    USER_PROMPT = "<|user|>"
    ASSISTANT_PROMPT = "<|assistant|>"
    PROMPT_SUFFIX = "<|end|>"


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


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


def dump_model(gpu_prop):
    for model_name in [
        "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-4-multimodal-instruct",
    ]:
        model_path = os.path.join(HF_MODEL_DIR, model_name)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        ).to("cuda")

        model = model.eval()
        print(f"{model.config=}")
        print("model.config._attn_implementation:", model.config._attn_implementation)

        file_path = os.path.join(model_path, "model.txt")
        with open(file_path, "w") as f:
            print(f"{processor=}", file=f)
            print_model_params(model, f"{model_name}", f)

        # Load generation config
        generation_config = GenerationConfig.from_pretrained(model_path)
        print(f"{generation_config=}")

        del model
        torch.cuda.empty_cache()


def text2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    prompt = (
        f"{USER_PROMPT}what is the answer for 1+1? Explain it.{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    )
    inputs = processor(prompt, images=None, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids,
        # skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f">>> Response\n{response}")


def text2text_stream(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    prompt = (
        f"{USER_PROMPT}what is the answer for 1+1? Explain it.{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    )
    inputs = processor(prompt, images=None, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    for i in range(2):
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            do_sample=True,
            temperature=0.9,
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


def single_vision2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # single-image prompt
    prompt = f"{USER_PROMPT}<|image_1|>描述图片中的内容?{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def multi_vision2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # multi-image prompt
    images = []
    placeholder = ""
    for i in range(1, 5):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
        images.append(Image.open(requests.get(url, stream=True).raw))
        placeholder += f"<|image_{i}|>"
    messages = [
        {"role": "user", "content": placeholder + "Summarize the deck of slides."},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f">>> Prompt\n{prompt}")

    inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def single_turn_vision2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # chat template
    chat = [
        {"role": "system", "content": f"请用中文回复"},
        {"role": "user", "content": f"<|image_1|>请描述图片中的内容?"},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")
    print(f">>> Prompt\n{prompt}")

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def multi_turn_vision2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # chat template
    chat = [
        {"role": "system", "content": f"请用中文回复"},
        {"role": "user", "content": f"<|image_1|>What is shown in this image?"},
        {
            "role": "assistant",
            "content": "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
        },
        {"role": "user", "content": "<|image_2|>What is so special about this image"},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")

    print(f">>> Prompt\n{prompt}")

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=prompt, images=[image, image], return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def speech_vision2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # speech-image prompt
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    prompt = f"{USER_PROMPT}<|image_1|><|speech_1|>{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    speech = sf.read(AUDIO_FILE_1)
    inputs = processor(text=prompt, images=[image], speechs=[speech], return_tensors="pt").to(
        "cuda"
    )

    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def asr(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # single speech prompt
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
    prompt = f"{USER_PROMPT}<|audio_1|>{speech_prompt}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
    audio = sf.read(AUDIO_FILE_1)
    inputs = processor(text=prompt, audios=[audio], return_tensors="pt").to("cuda")
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def multi_asr(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # multi speech prompt
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    AUDIO_FILE_2 = os.path.join(model_path, "examples/what_is_shown_in_this_image.wav")
    audio_1 = sf.read(AUDIO_FILE_2)
    audio_2 = sf.read(AUDIO_FILE_1)
    chat = [
        {
            "role": "user",
            "content": f"<|audio_1|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
        },
        {
            "role": "assistant",
            "content": "What is shown in this image.",
        },
        {
            "role": "user",
            "content": f"<|audio_2|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
        },
    ]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")

    print(f">>> Prompt\n{prompt}")

    inputs = processor(text=prompt, audios=[audio_1, audio_2], return_tensors="pt").to("cuda")

    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def multi_turn_vision_speech2text(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # multi speech prompt
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    AUDIO_FILE_2 = os.path.join(model_path, "examples/what_is_shown_in_this_image.wav")
    audio_1 = sf.read(AUDIO_FILE_2)
    audio_2 = sf.read(AUDIO_FILE_1)
    chat = [
        {"role": "user", "content": f"<|image_1|><|audio_1|>"},
        {
            "role": "assistant",
            "content": "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
        },
        {"role": "user", "content": f"<|audio_2|>"},
    ]
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")

    print(f">>> Prompt\n{prompt}")

    inputs = processor(
        text=prompt,
        images=[image],
        # audios=[(audio_1[0], 16000), (audio_2[0], 16000)],
        audios=[audio_1, audio_2],
        return_tensors="pt",
    ).to("cuda")

    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f">>> Response\n{response}")


def multi_turn_vision_speech2text_stream(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")
    model = model.eval()

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_path)
    print(f"{generation_config=}")

    # multi turn speech + image prompt
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    AUDIO_FILE_2 = os.path.join(model_path, "examples/what_is_shown_in_this_image.wav")
    audio_1 = sf.read(AUDIO_FILE_2)
    audio_2 = sf.read(AUDIO_FILE_1)
    chat = [
        {"role": "user", "content": f"<|image_1|><|audio_1|>"},
        {
            "role": "assistant",
            "content": "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
        },
        {"role": "user", "content": f"<|audio_2|>"},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")

    print(f">>> Prompt\n{prompt}")

    inputs = processor(
        text=prompt, images=[image], audios=[audio_1, audio_2], return_tensors="pt"
    ).to("cuda")

    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    for i in range(2):
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            do_sample=True,
            temperature=0.9,
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


def cover_chat(chat: list):
    # print(chat)
    audio_cn = 0
    image_cn = 0
    for item in chat:
        tmp_text = ""
        tmp_content = ""
        sub_audio_cn = 0
        sub_image_cn = 0
        if "content" in item and isinstance(item["content"], list):
            for c_item in item["content"]:
                assert isinstance(c_item, dict)
                if "text" in c_item:
                    tmp_text = c_item["text"]
                if "image" in c_item:
                    sub_image_cn += 1
                if "audio" in c_item:
                    sub_audio_cn += 1
            image_cn += sub_image_cn
            audio_cn += sub_audio_cn
            for i in range(image_cn - sub_image_cn, image_cn):
                tmp_content += f"<|image_{i+1}|>"
            for i in range(audio_cn - sub_audio_cn, audio_cn):
                tmp_content += f"<|audio_{i+1}|>"
            if tmp_text:
                tmp_content += tmp_text
            item["content"] = tmp_content

    return chat


def tokenize(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image)
    AUDIO_FILE_1 = os.path.join(model_path, "examples/what_is_the_traffic_sign_in_the_image.wav")
    AUDIO_FILE_2 = os.path.join(model_path, "examples/what_is_shown_in_this_image.wav")
    audio_1 = sf.read(AUDIO_FILE_1)
    audio_2 = sf.read(AUDIO_FILE_2)
    print(f"{audio_1[1]} {audio_1[0].shape=} {audio_1[0].reshape(-1).shape=}")
    print(f"{audio_2[1]} {audio_2[0].shape=} {audio_2[0].reshape(-1).shape=}")
    chat = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_1[0].reshape(-1)},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio_2[0].reshape(-1)}],
        },
    ]
    # mm prompt
    audios, images, videos = process_mm_info(chat, use_audio_in_video=False)
    {print(f"audios[{i}]: {item.shape}") for i, item in enumerate(audios)} if audios else print(
        audios
    )
    {print(f"images[{i}]: {item}") for i, item in enumerate(images)} if images else print(images)
    {print(f"videos[{i}]: {item.shape}") for i, item in enumerate(videos)} if videos else print(
        videos
    )

    # text promt
    chat = cover_chat(chat)
    print(chat)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")
    print(f">>> Prompt\n{prompt}")

    # tokenize (tokens -> token_ids)
    new_audios = []
    for audio in audios:
        new_audios.append((audio, 16000))
    inputs = processor(text=prompt, images=images, audios=new_audios, return_tensors="pt").to(
        "cuda" if gpu_prop else "cpu", dtype=torch.bfloat16
    )
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]

    # text token
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")


def tokenize_debug(gpu_prop):
    model_name = os.getenv("LLM_MODEL", "microsoft/Phi-4-multimodal-instruct")
    model_path = os.path.join(HF_MODEL_DIR, model_name)
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    print(image)
    chat = [
        {"role": "system", "content": "请用中文交流"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你好"},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! Is there something specific you would like to know or discuss?",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字"},
                {"type": "image", "image": image},
            ],
        },
    ]
    # mm prompt
    audios, images, videos = process_mm_info(chat, use_audio_in_video=False)
    {print(f"audios[{i}]: {item.shape}") for i, item in enumerate(audios)} if audios else print(
        audios
    )
    {print(f"images[{i}]: {item}") for i, item in enumerate(images)} if images else print(images)
    {print(f"videos[{i}]: {item.shape}") for i, item in enumerate(videos)} if videos else print(
        videos
    )

    # text promt
    chat = cover_chat(chat)
    # print(chat)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    prompt = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
    if prompt.endswith("<|endoftext|>"):
        prompt = prompt.rstrip("<|endoftext|>")
    print(f"{prompt=}")

    # tokenize (tokens -> token_ids)
    if audios:
        new_audios = []
        for audio in audios:
            new_audios.append((audio, 16000))
        audios = new_audios if new_audios else None
    inputs = processor(text=prompt, images=images, audios=audios, return_tensors="pt").to(
        "cuda" if gpu_prop else "cpu", dtype=torch.bfloat16
    )
    for key, value in inputs.items():
        if value is not None:
            print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]

    # text token
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")


"""
language:
- Text: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian
- Vision: English 
- Audio: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese

# multimmodal input formats:
- https://huggingface.co/microsoft/Phi-4-multimodal-instruct#input-formats

modal run src/download_models.py --repo-ids "microsoft/Phi-4-mini-instruct,microsoft/Phi-4-multimodal-instruct"

IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task tokenize_debug
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task tokenize
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task dump_model

IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task text2text
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task text2text_stream

# NOTE: more vision(image) + speech encode need more memory
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task asr
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task multi_asr
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task single_vision2text 
IMAGE_GPU=L4 modal run src/llm/transformers/phi4_multimodal.py --task multi_vision2text 
IMAGE_GPU=T4 modal run src/llm/transformers/phi4_multimodal.py --task single_turn_vision2text
IMAGE_GPU=L4 modal run src/llm/transformers/phi4_multimodal.py --task multi_turn_vision2text 
IMAGE_GPU=L4 modal run src/llm/transformers/phi4_multimodal.py --task speech_vision2text
IMAGE_GPU=L4 modal run src/llm/transformers/phi4_multimodal.py --task multi_turn_vision_speech2text
IMAGE_GPU=L4 modal run src/llm/transformers/phi4_multimodal.py --task multi_turn_vision_speech2text_stream
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "tokenize_debug": tokenize_debug,
        "tokenize": tokenize,
        "text2text": text2text,
        "text2text_stream": text2text_stream,
        "single_vision2text": single_vision2text,
        "multi_vision2text": multi_vision2text,
        "single_turn_vision2text": single_turn_vision2text,
        "multi_turn_vision2text": multi_turn_vision2text,
        "speech_vision2text": speech_vision2text,
        "asr": asr,
        "multi_asr": multi_asr,
        "multi_turn_vision_speech2text": multi_turn_vision_speech2text,
        "multi_turn_vision_speech2text_stream": multi_turn_vision_speech2text_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
