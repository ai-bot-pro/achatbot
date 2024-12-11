from io import BytesIO
import logging
from threading import Thread

from PIL import Image
import torch

try:
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        TextIteratorStreamer,
        GenerationConfig,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Molmo multimodal language models., you need to `pip install achatbot[llm_transformers_manual_vision_molmo]`,"
        "use awq model need to `pip install achatbot[llm_transformers_manual_vision_molmo,autoawq]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionMolmoLLM(TransformersBaseLLM):
    r"""
    no chat template, no chat history, just image with text prompt
    """

    TAG = "llm_transformers_manual_vision_molmo"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation=self.args.lm_attn_impl,
                trust_remote_code=True,
            )
        else:
            self._model = (
                AutoModelForCausalLM.from_pretrained(
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
        self._processor = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True,
        )

        self.warmup()

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt

        dummy_image = Image.new(mode="RGB", size=(300, 300))
        img_obj = None
        with BytesIO() as buffered:
            dummy_image.save(buffered, "JPEG")
            img_obj = Image.open(BytesIO(buffered.getvalue()))

        # Preparation for inference
        inputs = self._processor.process(
            images=[img_obj],
            text=dummy_input_text,
        )
        # move inputs to the correct device and make a batch of size 1
        model_inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        warmup_gen_kwargs = dict(
            streamer=streamer,
            tokenizer=self._processor.tokenizer,
        )
        warmup_gen_args = (
            model_inputs,
            GenerationConfig(
                do_sample=self.args.lm_gen_do_sample,
                top_k=self.args.lm_gen_top_k,
                top_p=self.args.lm_gen_top_p,
                temperature=self.args.lm_gen_temperature,
                repetition_penalty=self.args.lm_gen_repetition_penalty,
                min_new_tokens=self.args.lm_gen_min_new_tokens,
                max_new_tokens=self.args.lm_gen_max_new_tokens,
                stop_strings="<|endoftext|>",
            ),
        )
        self._warmup(
            target=self._model.generate_from_batch,
            args=warmup_gen_args,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    def generate(self, session: Session):
        prompt = session.ctx.state["prompt"]
        text = ""
        image_inputs = None

        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, tuple):
            prompt, language_code = prompt
            if isinstance(prompt, str):
                if TO_LLM_LANGUAGE[language_code] == "zh":
                    prompt = "请用中文回复。" + prompt
                else:
                    prompt = (
                        f"Please reply to my message in {TO_LLM_LANGUAGE[language_code]}. " + prompt
                    )
            text = prompt
        elif isinstance(prompt, list):
            # no chat template so get image with textpromt
            image_inputs, _ = process_vision_info(
                [{"role": self.args.user_role, "content": prompt}]
            )
            for item in prompt:
                if "type" in item and item["type"] == "text":
                    text += item["text"]
        else:
            raise Exception(f"Unsupported prompt type: {type(prompt)}")

        # Preparation for inference
        inputs = self._processor.process(
            images=image_inputs,
            text=text,
        )
        # move inputs to the correct device and make a batch of size 1
        model_inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            streamer=streamer,
            tokenizer=self._processor.tokenizer,
        )
        gen_args = (
            model_inputs,
            GenerationConfig(
                do_sample=self.args.lm_gen_do_sample,
                top_k=self.args.lm_gen_top_k,
                top_p=self.args.lm_gen_top_p,
                temperature=self.args.lm_gen_temperature,
                repetition_penalty=self.args.lm_gen_repetition_penalty,
                min_new_tokens=self.args.lm_gen_min_new_tokens,
                max_new_tokens=self.args.lm_gen_max_new_tokens,
                stop_strings="<|endoftext|>",
            ),
        )
        thread = Thread(target=self._model.generate_from_batch, args=gen_args, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
