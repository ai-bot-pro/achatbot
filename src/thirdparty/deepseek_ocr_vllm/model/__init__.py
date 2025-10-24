# https://docs.vllm.ai/en/latest/api/vllm/config/index.html#vllm.config.ModelConfig
# maybe config in ModelConfig mm_processor_kwargs, not good design
# this is default config const, now use env to config

import os

from transformers import AutoTokenizer

from src.common.types import MODELS_DIR

BASE_SIZE = int(os.getenv("DS_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("DS_OCR_IMAGE_SIZE", "640"))
CROP_MODE = bool(os.getenv("DS_OCR_CROP_MODE", "1"))
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

PROMPT = os.getenv("DS_OCR_PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......

MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.
VERBOSE = bool(os.getenv("DS_OCR_VERBOSE", ""))

MODEL_PATH = os.getenv(
    "LLM_MODEL_NAME_OR_PATH", os.path.join(MODELS_DIR, "deepseek-ai/DeepSeek-OCR")
)


TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
