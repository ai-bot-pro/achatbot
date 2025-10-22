import io
import os
import sys
import uuid
import asyncio
import subprocess
from threading import Thread
from PIL import Image


import modal


app = modal.App("deepseek-ocr")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
BACKEND = os.getenv("BACKEND", None)
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

if BACKEND == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.27.dev8",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
    )


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
    from transformers.generation.streamers import TextStreamer, TextIteratorStreamer
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
            eos_text = self.tokenizer.decode(
                [self.tokenizer.eos_token_id], skip_special_tokens=False
            )
            text = text.replace(eos_text, "\n")
            # print(text, flush=True, end="")
            stream_end and print("stream_end is True", flush=True)

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    # prompt = "<image>\nFree OCR. "
    # document: <image>\n<|grounding|>Convert the document to markdown.
    # other image: <image>\n<|grounding|>OCR this image.
    # without layouts: <image>\nFree OCR.
    # figures in document: <image>\nParse the figure.
    # general: <image>\nDescribe this image in detail.
    # rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
    # '先天下之忧而忧'
    # .......

    # create dummy image file
    dummy_img = Image.new("RGB", (640, 640), color="white")
    ioBuff = io.BytesIO()
    dummy_img.save(ioBuff, format="PNG")
    ioBuff.seek(0)

    image_files = [
        dummy_img,
        ioBuff,
        "/DeepSeek-OCR/assets/fig1.png",
        # use ORC detected Show pictures, detect again :)
        "/DeepSeek-OCR/assets/show1.jpg",
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
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            # skip_special_tokens=True,
            skip_special_tokens=False,
        )
        # https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/refs%2Fpr%2F23/modeling_deepseekocr.py#L707
        gen_kwargs = dict(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_file,
            # output_path=DEEPSEEK_ASSETS_DIR,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,  # open save results, u can push to S3 or other storage for cdn :)
            test_compress=False,
            eval_mode=False,
            # streamer=NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
            streamer=streamer,
            verbose=False,
        )
        thread = Thread(target=model.infer, kwargs=gen_kwargs)
        thread.start()
        for new_text in streamer:
            if new_text is not None:
                print(new_text, flush=True, end="")  # note: flush with buff to print


@torch.inference_mode()
def infer_filter(**kwargs):
    class NoEOSTextStreamer(TextStreamer):
        def on_finalized_text(self, text: str, stream_end: bool = False):
            eos_text = self.tokenizer.decode(
                [self.tokenizer.eos_token_id], skip_special_tokens=False
            )
            text = text.replace(eos_text, "\n")
            # print(text, flush=True, end="")
            stream_end and print("stream_end is True", flush=True)

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        # torch_dtype=torch.bfloat16,
        # device_map="cuda" if torch.cuda.is_available() else "auto",  # need accelerate>=0.26.0
    )
    model = model.eval().cuda().to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # NOTE: use FIED task prompt
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    # prompt = "<image>\n<|grounding|>Convert the document to text. " # have issue
    # prompt = "<image>\nFree OCR. "
    # document: <image>\n<|grounding|>Convert the document to markdown.
    # other image: <image>\n<|grounding|>OCR this image.
    # without layouts: <image>\nFree OCR.
    # figures in document: <image>\nParse the figure.
    # general: <image>\nDescribe this image in detail.
    # rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
    # '先天下之忧而忧'
    # .......

    # create dummy image file
    dummy_img = Image.new("RGB", (640, 640), color="white")
    ioBuff = io.BytesIO()
    dummy_img.save(ioBuff, format="PNG")
    ioBuff.seek(0)

    image_files = [
        dummy_img,
        ioBuff,
        "/DeepSeek-OCR/assets/fig1.png",
        # use ORC detected Show pictures, detect again :)
        "/DeepSeek-OCR/assets/show1.jpg",
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
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            # skip_special_tokens=True,
            skip_special_tokens=False,
        )
        # https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/refs%2Fpr%2F23/modeling_deepseekocr.py#L707
        gen_kwargs = dict(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_file,
            # output_path=DEEPSEEK_ASSETS_DIR,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,  # open save results, u can push to S3 or other storage for cdn :)
            test_compress=False,
            eval_mode=False,
            # streamer=NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
            streamer=streamer,
            verbose=False,
        )
        thread = Thread(target=model.infer, kwargs=gen_kwargs)
        thread.start()
        is_ref_det = False
        det_text = ""
        for new_text in streamer:
            if new_text is None:
                continue
            print(new_text, flush=True, end="")  # note: flush with buff to print
            if "<|ref|>" in new_text:
                is_ref_det = True

            if "<|/det|>" in new_text:
                is_ref_det = False
                new_text = new_text.split("<|/det|>")[1]

            if "<｜end▁of▁sentence｜>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<｜end▁of▁sentence｜>")[0]
                print("\n" + 20 * "----")
            if "<|end▁of▁sentence|>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<|end▁of▁sentence|>")[0]
                print("\n" + 20 * "----")

            if is_ref_det is False:
                det_text += new_text
        print("det_text:", det_text)
        torch.cuda.empty_cache()


async def achatbot_infer(**kwargs):
    from achatbot.processors.vision.ocr_processor import OCRProcessor
    from achatbot.modules.vision.ocr import VisionOCREnvInit
    from achatbot.common.session import SessionCtx, Session
    from achatbot.types.frames.data_frames import UserImageRawFrame
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    ocr = VisionOCREnvInit.initVisionOCREngine(
        "llm_transformers_manual_vision_deepseek_ocr",
        {
            "lm_model_name_or_path": MODEL_PATH,
            "lm_device": "cuda",
            "ocr_base_size": 1024,
            "ocr_image_size": 640,
            "ocr_crop_mode": True,
            "ocr_prompt": "<image>\n<|grounding|>Convert the document to markdown. ",
        },
    )
    session = Session(**SessionCtx(str(uuid.uuid4())).__dict__)
    processor = OCRProcessor(ocr=ocr, session=session)
    image_files = [
        Image.new("RGB", (640, 640), color="white"),
        Image.open("/DeepSeek-OCR/assets/fig1.png"),
        # use ORC detected Show pictures, detect again :)
        Image.open("/DeepSeek-OCR/assets/show1.jpg"),
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
modal run src/download_models.py --repo-ids "deepseek-ai/DeepSeek-OCR" --revision "refs/pr/23"

IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek.py --task dump_model
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek.py --task infer
IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek.py --task infer_filter
BACKEND=achatbot IMAGE_GPU=L4 modal run src/llm/transformers/vlm/ocr_deepseek.py --task achatbot_infer
"""


@app.local_entrypoint()
def main(task: str = "dump_model"):
    tasks = {
        "dump_model": dump_model,
        "infer": infer,
        "infer_filter": infer_filter,
        "achatbot_infer": achatbot_infer,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
    )
