import logging
from threading import Thread
from PIL import Image
from time import perf_counter

try:
    from transformers import AutoProcessor, TextIteratorStreamer, AutoModelForImageTextToText
    import torch

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Smol-VLM, you need to `pip install achatbot[llm_transformers_manual_vision_smolvlm]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import get_device, print_model_params
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionSmolLM(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_smollm"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        gpu_prop = torch.cuda.get_device_properties("cuda")

        if self.args.lm_device_map:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=torch.bfloat16,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModelForImageTextToText.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        logging.info(f"TransformersLMArgs: {self.args}")
        print_model_params(self._model, self.TAG)
        self._tokenizer = AutoProcessor.from_pretrained(self.args.lm_model_name_or_path, use_fast=True)

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

        inputs = self._tokenizer.apply_chat_template(
            dummy_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self._model.device,
            dtype=torch.bfloat16,
        )

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=False,
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
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = session.ctx.state["prompt"]
        assert len(prompt) > 0
        text = prompt[0].get("text", "")
        text = self.args.init_chat_prompt + text
        if (
            not self.args.lm_language_code
            or self.args.lm_language_code not in TO_LLM_LANGUAGE.keys()
        ):
            self.args.lm_language_code = "zh"
        if self.args.lm_language_code == "zh":
            # NOTE: smol-vlm just do vision task, don't support to chat (Q/A task) need sft and do Pareto Front
            text = (
                self.args.init_chat_prompt if self.args.init_chat_prompt else "请用中文描述图片内容"
            )
        prompt[0]["text"] = text
        logging.info(f"{prompt[0]=}")

        message = {"role": self.args.user_role, "content": prompt}
        # logging.info(f"{message=}")
        inputs = self._tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self._model.device,
            dtype=torch.bfloat16,
        )

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=False,
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            # 如果要使用重复惩罚, 虽然会解决重复, 但是会降低生成质量
            repetition_penalty=1.1,
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
        torch.cuda.empty_cache()
