import logging
import re
import uuid
from time import perf_counter
import asyncio
import copy

from PIL import Image
import torch

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from src.thirdparty.deepseek_ocr_vllm.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from src.thirdparty.deepseek_ocr_vllm.process.image_process import DeepseekOCRProcessor
    from src.thirdparty.deepseek_ocr_vllm.process import load_image
    from src.thirdparty.deepseek_ocr_vllm.model import BASE_SIZE, IMAGE_SIZE, CROP_MODE, PROMPT
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.interface import ILlm, IVisionOCR
from src.common.session import Session
from src.common.types import SessionCtx
from src.core.llm.vllm.base import VllmEngineBase


class VllmDeepSeekOCR(VllmEngineBase, IVisionOCR):
    """
    deepseek ocr with vllm
    - https://github.com/deepseek-ai/DeepSeek-OCR/tree/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm
    """

    TAG = "llm_vllm_deepseek_ocr"

    def __init__(self, **kwargs) -> None:
        if self.TAG == "llm_vllm_deepseek_ocr":
            from src.thirdparty.deepseek_ocr_vllm.model.deepseek_ocr import (
                DeepseekOCRForCausalLM,
            )  # import to register

        self.base_size = BASE_SIZE
        self.image_size = IMAGE_SIZE
        self.crop_mode = CROP_MODE
        self.prompt = kwargs.pop("ocr_prompt", PROMPT)

        super().__init__(**kwargs)

    def init(self):
        pass

    async def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        session.ctx.state["ocr_img"] = dummy_pil_image
        for i in range(self.args.warmup_steps):
            logging.info(f"{i} warmup start")
            async for result_text in self.async_generate(session):
                pass

    def set_task_prompt(self, prompt: str):
        self.prompt = prompt

    async def async_generate(self, session: Session, **kwargs):
        """
        session.ctx.state["ocr_img"] PIL.Image.Image
        """
        ocr_img = session.ctx.state["ocr_img"]
        image = load_image(ocr_img).convert("RGB")

        # extract image features from  PIL.Image.Image
        image_features = None
        if "<image>" in self.prompt:
            image_features = DeepseekOCRProcessor(
                tokenizer=self.tokenizer,
                image_size=self.image_size,
                base_size=self.base_size,
            ).tokenize_with_images(
                self.prompt, images=[image], bos=True, eos=True, cropping=self.crop_mode
            )

        # for vLLM V0
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}
            )
        ]  # whitelist: <td>, </td>
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
            # ignore_eos=False,
        )

        request = {"prompt": self.prompt}
        if image_features and "<image>" in self.prompt:
            request = {"prompt": self.prompt, "multi_modal_data": {"image": image_features}}

        generated_text = ""
        start = perf_counter()
        times = []
        is_ref_det = False
        sentence = ""
        printed_length = 0
        # TODO: extract detect image to storage with s3 (use callback)
        request_id = str(uuid.uuid4().hex)
        async for request_output in self.engine.generate(request, sampling_params, request_id):
            if request_output.outputs is None or len(request_output.outputs) == 0:
                continue
            times.append(perf_counter() - start)

            generated_text = request_output.outputs[0].text
            new_text = generated_text[printed_length:]
            printed_length = len(generated_text)

            if "<|ref|>" in new_text:
                is_ref_det = True

            if "<|/det|>" in new_text:
                is_ref_det = False
                new_text = new_text.split("<|/det|>")[1]

            if "<｜end▁of▁sentence｜>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<｜end▁of▁sentence｜>")[0]
            if "<|end▁of▁sentence|>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<|end▁of▁sentence|>")[0]

            if is_ref_det is False:
                sentence += new_text
                pos = self._have_special_char(sentence)
                if pos > -1:
                    yield sentence[: pos + 1]
                    sentence = sentence[pos + 1 :]
            start = perf_counter()
        if len(sentence) > 0:
            yield sentence + "."
        if times:
            logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()


class VllmOfficeDeepSeekOCR(VllmDeepSeekOCR):
    """
    officially supported in upstream vLLM.
    https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html
    # Until v0.11.1 release, you need to install vLLM from nightly build
    # pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    """

    TAG = "llm_office_vllm_deepseek_ocr"

    async def async_generate(self, session: Session, **kwargs):
        """
        session.ctx.state["ocr_img"] PIL.Image.Image
        """
        ocr_img = session.ctx.state["ocr_img"]
        image = load_image(ocr_img).convert("RGB")

        # NOTE: vLLM V1 does not support per request user provided logits processors.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )

        request = {"prompt": self.prompt}
        if "<image>" in self.prompt:
            request = {"prompt": self.prompt, "multi_modal_data": {"image": image}}

        generated_text = ""
        start = perf_counter()
        times = []
        is_ref_det = False
        sentence = ""
        printed_length = 0
        # TODO: extract detect image to storage with s3 (use callback)
        request_id = str(uuid.uuid4().hex)
        async for request_output in self.engine.generate(request, sampling_params, request_id):
            if request_output.outputs is None or len(request_output.outputs) == 0:
                continue
            times.append(perf_counter() - start)

            generated_text = request_output.outputs[0].text
            new_text = generated_text[printed_length:]
            printed_length = len(generated_text)

            if "<|ref|>" in new_text:
                is_ref_det = True

            if "<|/det|>" in new_text:
                is_ref_det = False
                new_text = new_text.split("<|/det|>")[1]

            if "<｜end▁of▁sentence｜>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<｜end▁of▁sentence｜>")[0]
            if "<|end▁of▁sentence|>" in new_text:
                if "<|/ref|>" not in new_text:
                    new_text = new_text.split("<|end▁of▁sentence|>")[0]

            if is_ref_det is False:
                sentence += new_text
                pos = self._have_special_char(sentence)
                if pos > -1:
                    yield sentence[: pos + 1]
                    sentence = sentence[pos + 1 :]
            start = perf_counter()
        if len(sentence) > 0:
            yield sentence + "."
        if times:
            logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()
