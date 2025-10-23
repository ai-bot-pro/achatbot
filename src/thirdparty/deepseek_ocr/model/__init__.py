# https://docs.vllm.ai/en/latest/api/vllm/config/index.html#vllm.config.ModelConfig
# maybe config in ModelConfig mm_processor_kwargs, not good design
# this is default config const

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
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
VERBOSE = False
