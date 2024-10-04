import logging
from threading import Thread
import time

import torch

from src.core.llm.base import BaseLLM
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.interface import ILlm
from src.types.vision.ocr.transformers_got import TransformersGoTOCRArgs

try:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        f"In order to use GOT OCR2.0 language models., you need to `pip install achatbot[lm_transformers_manual_vision_got_ocr]`,"
        f"use awq model need to `pip install achatbot[lm_transformers_manual_vision_got_ocr]`")
    raise Exception(f"Missing module: {e}")



class TransformersGOTOCRLM(BaseLLM, ILlm):
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

    def generate(self, session: Session):
        if 'ocr_img' not in session.ctx.state:
            yield None
            return

        image = session.ctx.state['ocr_img']
        generation_kwargs = dict(
            streamer=self._streamer,
            ocr_type=self.args.ocr_box,
            ocr_box=self.args.ocr_box)
        thread = Thread(
            target=self._model.chat,
            args=(self._tokenizer, image),
            kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self._streamer:
            generated_text += new_text
            yield new_text

    def chat_completion(self, session: Session):
        if self.args.lm_stream is False:
            res = ""
            for text in self.generate(session):
                res += text
            yield res
        else:
            res = ""
            for text in self.generate(session):
                res += text
                pos = self._have_special_char(res)
                if pos > -1:
                    yield res[:pos + 1]
                    res = res[pos + 1:]
            if len(res) > 0:
                yield res
