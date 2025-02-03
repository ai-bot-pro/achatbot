import logging
from threading import Thread

import torch

try:
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen2-VL, you need to `pip install achatbot[llm_transformers_manual_vision_qwen]`,"
        "use awq model need to `pip install achatbot[llm_transformers_manual_vision_qwen,autoawq]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionQwenLLM(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_qwen"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        if self.args.lm_device_map:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                Qwen2VLForConditionalGeneration.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=self.args.lm_torch_dtype,
                    attn_implementation=self.args.lm_attn_impl,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a
        # token count range of 256-1280, to balance speed and memory usage.
        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True,
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": self.args.init_chat_prompt,
                }
            )

        self.warmup()

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt
        dummy_msgs = [{"role": self.args.user_role, "content": dummy_input_text}]

        # Preparation for inference
        text = self._tokenizer.apply_chat_template(
            dummy_msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(dummy_msgs)
        model_inputs = self._tokenizer(
            images=image_inputs,
            text=[text],
            videos=video_inputs,
            padding=True,
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

    def generate(self, session: Session):
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
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(chat_history)
        logging.debug(f"image_inputs:{image_inputs},video_inputs:{video_inputs}")
        # https://github.com/huggingface/transformers/blob/8bd2b1e8c23234cd607ca8d63f53c1edfea27462/src/transformers/models/qwen2_vl/processing_qwen2_vl.py#L53
        model_inputs = self._tokenizer(
            images=image_inputs, text=[text], videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
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
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
        self._chat_history.append({"role": "assistant", "content": generated_text})
