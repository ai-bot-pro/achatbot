import logging
from threading import Thread
import time

import torch
from PIL import Image

from src.core.llm.base import BaseLLM
from src.common.session import Session
from src.common.interface import IVisionOCR
from src.types.vision.ocr.transformers_got import TransformersGoTOCRArgs

try:
    from qwen_vl_utils import fetch_image
    from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        f"In order to use GOT OCR2.0 language models., you need to `pip install achatbot[vision_transformers_got_ocr]`,"
        f"use awq model need to `pip install achatbot[vision_transformers_got_ocr]`")
    raise Exception(f"Missing module: {e}")


class TransformersGOTOCRLM(BaseLLM, IVisionOCR):
    """
    the General OCR Theory (GOT), ViTDet(vision encoder) + qwen2 0.5B
    the ViTDet backbone available at:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py

    """
    TAG = "vision_transformers_got_ocr"

    def __init__(self, **args) -> None:

        self.args = TransformersGoTOCRArgs(**args)

        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        if self.args.lm_device_map:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            ).eval().to(self.args.lm_device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True)
        self._streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.warmup()

    def count_tokens(self, text: str | bytes) -> int:
        return len(self._tokenizer.encode(text)) if self._tokenizer else 0

    def warmup(self):
        pass

    def _generate(self, session: Session):
        if 'ocr_img' not in session.ctx.state:
            yield None
            return

        ocr_image = session.ctx.state['ocr_img']
        if isinstance(ocr_image, dict) \
            and all(
                isinstance(k, str)
                and isinstance(v, (str, Image.Image)) for k, v in ocr_image.items()):
            image = fetch_image(ocr_image)
        elif isinstance(ocr_image, Image.Image):
            image = ocr_image
        else:
            raise ValueError(
                f"Unrecognized image input: PIL.Image or dict key:image/image_url, val:support local path, http url, base64 and PIL.Image, got {ocr_image}")

        generation_kwargs = dict(
            ocr_type=self.args.ocr_type,
            ocr_box=self.args.ocr_box,
            ocr_color=self.args.ocr_color,
            stream_flag=True,
            gradio_input=True,
            streamer=self._streamer,
        )
        thread = Thread(
            target=self._model.chat,
            args=(self._tokenizer, image),
            kwargs=generation_kwargs)
        thread.start()

        for new_text in self._streamer:
            yield new_text

    def generate(self, session: Session):
        if self.args.lm_stream is False:
            res = ""
            for text in self._generate(session):
                res += text
            yield res
        else:
            res = ""
            for text in self._generate(session):
                res += text
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[:pos + 1]
                    res = res[pos + 1:]
            if len(res) > 0:
                yield res + "."
