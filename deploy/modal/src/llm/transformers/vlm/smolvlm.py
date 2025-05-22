import os
import subprocess
from threading import Thread
from time import perf_counter


import modal


app = modal.App("smolvlm")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "num2words",
        "transformers",
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
            "LLM_MODEL": os.getenv("LLM_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
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
    from transformers import AutoProcessor, AutoModelForImageTextToText
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


def dump_model(gpu_prop):
    for model_name in [
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    ]:
        model_path = os.path.join(HF_MODEL_DIR, model_name)
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        ).to("cuda")

        model = model.eval()
        print(f"{model.config=}")
        print(f"{processor=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


# https://github.com/huggingface/smollm/tree/main/vision
def predict(gpu_prop):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")

    model = model.eval()
    print(f"{model.config=}")
    print(f"{processor=}")
    # print(f"{model=}")
    print_model_params(model, f"{model_name}")

    # Construct prompt
    text = "请用中文描述图片内容"
    # text = f"请用中文描述图片内容，不要使用Markdown格式回复。"

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

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # 如果要使用重复惩罚, 虽然会解决重复, 但是会降低生成质量
        #  bad case: 在椒花的突出体部上, 有一个尖细而干净的电触噬兽在吸收野生产成员. 被目标传通了一条线急忟利场非市主义人类对至代表力量和发展术预示所形成的弗雾化作为秋季戈巴斐協筝歌曲、旭辰四星点时间以前到现在是最大限度控制整个社区许可下载后立法定行列应执行这项技能指导方便分析管理模型的研修奖得名委员会负担问题； 查看网址：www.shen
        # repetition_penalty=1.5,
        max_new_tokens=256,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


# https://github.com/huggingface/smollm/tree/main/vision
def predict_stream(gpu_prop):
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, os.getenv("LLM_MODEL"))
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
    ).to("cuda")

    model = model.eval()
    print(f"{model.config=}")
    print(f"{processor=}")
    # print(f"{model=}")

    # Construct prompt
    text = "请用中文描述图片内容"
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
        for new_text in streamer:
            times.append(perf_counter() - start)
            print(new_text, end="", flush=True)
            generated_text += new_text
            start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


"""
IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task dump_model
IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict
IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict_stream
LLM_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict
LLM_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict_stream
LLM_MODEL=HuggingFaceTB/SmolVLM2-256M-Video-Instruct IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict
LLM_MODEL=HuggingFaceTB/SmolVLM2-256M-Video-Instruct IMAGE_GPU=T4 modal run src/llm/transformers/vlm/smolvlm.py --task predict_stream

IMAGE_GPU=L4 modal run src/llm/transformers/vlm/smolvlm.py --task predict
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/smolvlm.py --task predict_stream
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
