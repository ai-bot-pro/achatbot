import io
import logging
import os
import sys
from threading import Thread
from time import perf_counter
from typing import List
from dotenv import load_dotenv

from PIL import Image
import numpy as np

from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.common.utils.helper import get_device, print_model_params
from src.core.llm.transformers.base import TransformersBaseLLM
from src.types.llm.transformers import TransformersLMArgs
from src.types.speech.language import TO_LLM_LANGUAGE

load_dotenv(override=True)

try:
    import torch
    from transformers.generation.streamers import TextIteratorStreamer

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../FastVLM"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/FastVLM"))

    from deps.FastVLM.llava.utils import disable_torch_init
    from deps.FastVLM.llava.conversation import conv_templates
    from deps.FastVLM.llava.model.builder import load_pretrained_model
    from deps.FastVLM.llava.mm_utils import (
        tokenizer_image_token,
        process_images,
        get_model_name_from_path,
    )
    from deps.FastVLM.llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )

except ModuleNotFoundError as e:
    logging.error(
        "In order to use Fastvlm , you need to `pip install achatbot[llm_transformers_manual_vision_fastvlm]`."
    )
    raise Exception(f"Missing module: {e}")


class TransformersManualVisionFastvlm(TransformersBaseLLM):
    r"""
    https://github.com/apple/ml-fastvlm
    # FastViT(mobileCLIP) + Qwen2 0.5/1.5/7 B
    # https://github.com/apple/ml-fastvit
    # https://github.com/apple/ml-mobileclip
    # https://github.com/apple/ml-fastvlm
    """

    TAG = "llm_transformers_manual_vision_fastvlm"
    CONV_MODE = "qwen_2"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device(self.args.lm_device)
        logging.info(f"TransformersLMArgs: {self.args}")

        disable_torch_init()
        model_name = get_model_name_from_path(self.args.lm_model_name_or_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.args.lm_model_name_or_path,
            None,
            model_name,
            load_8bit=True if self.args.lm_bnb_quant_type == "int8" else False,
            load_4bit=True if self.args.lm_bnb_quant_type == "int4" else False,
            device=self.args.lm_device,
            device_map="auto" if self.args.lm_device_map is None else self.args.lm_device_map,
            use_flash_attn=True if self.args.lm_attn_impl == "flash_attention_2" else False,
        )
        self._model = model.eval()
        print_model_params(self._model, self.TAG)

        self._tokenizer = tokenizer
        self.context_len = context_len
        self._image_processor = image_processor

        # self._chat_history = ChatHistory(self.args.chat_history_size)
        self.warmup()

    def preprocess(self, text: str, images: List[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        # Construct prompt
        qs = text
        if self._model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.CONV_MODE].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Set the pad token id for generation
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        # Tokenize prompt
        input_ids = (
            tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(torch.device(self.args.lm_device))
        )

        # Load and preprocess image
        image_tensor = (
            process_images(
                images,
                self._image_processor,
                self._model.config,
            )
            .half()
            .to(torch.device(self.args.lm_device))
        )

        return input_ids, image_tensor

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        dummy_input_text = self.args.warnup_prompt
        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        input_ids, image_tensor = self.preprocess(dummy_input_text, [dummy_pil_image])

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[dummy_pil_image.size],
            streamer=streamer,
            do_sample=True if self.args.lm_gen_temperature > 0 else False,
            temperature=self.args.lm_gen_temperature,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=128,
            use_cache=True,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    def get_prompt(self, session: Session) -> list:
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) >= 2
        for item in session.ctx.state["prompt"][:-1]:  # user images prompt
            assert isinstance(item, Image.Image)
        assert isinstance(session.ctx.state["prompt"][-1], str)  # user str prompt

        prompt = session.ctx.state["prompt"]
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        # torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = self.get_prompt(session)
        text = prompt[-1]
        text = "Don't use Markdown format to reply. " + text
        if (
            not self.args.lm_language_code
            or self.args.lm_language_code not in TO_LLM_LANGUAGE.keys()
        ):
            self.args.lm_language_code = "zh"
        text = (
            f"Please reply to my message in {TO_LLM_LANGUAGE[self.args.lm_language_code]}. " + text
        )

        input_ids, image_tensor = self.preprocess(text, prompt[:-1])
        image_sizes = [image.size for image in prompt[:-1]]

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            streamer=streamer,
            do_sample=True
            if kwargs.get("temperature", self.args.lm_gen_temperature) > 0
            else False,
            temperature=kwargs.get("temperature", self.args.lm_gen_temperature),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.args.lm_gen_repetition_penalty
            ),
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            use_cache=True,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        start = perf_counter()
        times = []
        for new_text in streamer:
            times.append(perf_counter() - start)
            generated_text += new_text
            yield new_text
            start = perf_counter()
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")

        # torch.cuda.empty_cache()
