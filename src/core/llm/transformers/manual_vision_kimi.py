import io
import logging
import os
import sys
from threading import Thread
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
    from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
    from qwen_vl_utils import process_vision_info

except ModuleNotFoundError as e:
    logging.error(
        "In order to use Kimi VL, you need to `pip install achatbot[llm_transformers_manual_vision_kimi]`."
    )
    raise Exception(f"Missing module: {e}")


def split_model(model_name):
    device_map = {}

    # splits layers into different GPUs (need use L4 for bfloat16 with flash attention)
    model_splits = {
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/config.json
        "moonshotai/Kimi-VL-A3B-Instruct": [13, 14],  # 2 GPU for 16b
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/blob/main/config.json
        "moonshotai/Kimi-VL-A3B-Thinking": [13, 14],  # 2 GPU for 16b
    }
    num_layers_per_gpu = model_splits[model_name]
    # num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    # exlude layer and first layer on cuda 0
    device_map["vision_tower"] = 0
    device_map["multi_modal_projector"] = 0
    # device_map["image_newline"] = 0
    # device_map["view_seperator"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.0"] = 0
    return device_map


class TransformersManualVisionKimi(TransformersBaseLLM):
    r"""
        Multimodal Understanding
        https://github.com/MoonshotAI/Kimi-VL

        _tokenizer.tokenizer.encode + AR LM model(MHA/MLA+MoE) + _tokenizer.tokenizer.decode

    🤗 For general multimodal perception and understanding, OCR, long video and long document, video perception, and agent uses, we recommend Kimi-VL-A3B-Instruct for efficient inference; for advanced text and multimodal reasoning (e.g. math), please consider using Kimi-VL-A3B-Thinking.

    🤗 对于一般的多模态感知与理解、OCR、长视频和长文档、视频感知以及代理用途，我们推荐使用 Kimi-VL-A3B-Instruct 进行高效推理；对于高级文本和多模态推理（例如数学），请考虑使用 Kimi-VL-A3B-Thinking 。
    """

    TAG = "llm_transformers_manual_vision_kimi"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)

        # moonvit(ViT SigLIP + 2D RoPE) + MLP (pixel shuffle) + deepseek V3 (MLA + MOE)
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking/blob/main/config.json
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/config.json
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/configuration_kimi_vl.py
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
        if self.args.lm_device_map is not None:
            if isinstance(self.args.lm_device_map, dict):
                customer_deivce_map = self.args.lm_device_map
                default_device_map = split_model(
                    "/".join(self.args.lm_model_name_or_path.split("/")[-2:])
                )
                self.args.lm_device_map = {**default_device_map, **customer_deivce_map}
            logging.info(f"TransformersLMArgs: {self.args}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                trust_remote_code=True,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                torch_dtype=self.args.lm_torch_dtype,
            ).eval()
        else:
            self.args.lm_device = self.args.lm_device or get_device()
            logging.info(f"TransformersLMArgs: {self.args}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=self.args.lm_attn_impl,
                torch_dtype=self.args.lm_torch_dtype,
            )
            self._model = self._model.eval().to(self.args.lm_device)
        logging.info(f"TransformersLMArgs: {self.args} model.device: {self._model.device}")
        print_model_params(self._model, self.TAG)

        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        dummy_input_text = self.args.warnup_prompt
        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        dummy_msgs = [
            {
                "role": self.args.user_role,
                "content": [
                    {"type": "text", "text": dummy_input_text},
                    {"type": "image", "image": dummy_pil_image},
                ],
            }
        ]

        # Preparation for inference
        text = self._tokenizer.apply_chat_template(
            dummy_msgs, add_generation_prompt=True, return_tensors="pt"
        )
        image_inputs, video_inputs = process_vision_info(dummy_msgs)
        model_inputs = self._tokenizer(
            images=image_inputs,
            text=text,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            model_inputs,
            streamer=streamer,
            do_sample=self.args.lm_gen_do_sample,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = session.ctx.state["prompt"]
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if isinstance(prompt, str):
                prompt = (
                    f"Please reply to my message in {TO_LLM_LANGUAGE[language_code]}. " + prompt
                )

        message = {"role": self.args.user_role, "content": prompt}
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        logging.debug(f"chat_history:{chat_history}")
        text = self._tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=True, return_tensors="pt"
        )
        image_inputs, video_inputs = process_vision_info(chat_history)
        logging.debug(f"image_inputs:{image_inputs},video_inputs:{video_inputs}")
        # https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/processing_kimi_vl.py#L73
        model_inputs = self._tokenizer(
            images=image_inputs,
            text=text,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **model_inputs,
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
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
        self._chat_history.append({"role": "assistant", "content": generated_text})

        torch.cuda.empty_cache()
