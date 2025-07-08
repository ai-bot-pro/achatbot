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
        "In order to use Mimo-VLM, you need to `pip install achatbot[llm_transformers_manual_vision_mimo]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import get_device, print_model_params
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TransformersManualVisionMimo(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_mimo"

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
        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path, use_fast=True
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        # not init, use default mimo system promt: You are MiMo, an AI assistant developed by Xiaomi.
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": [{"type": "text", "text": self.args.init_chat_prompt}],
                }
            )

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
        message = {"role": self.args.user_role, "content": prompt}
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        logging.debug(f"chat_history:{chat_history}")
        inputs = self._tokenizer.apply_chat_template(
            chat_history,
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
        is_output_think = self.args.lm_gen_think_output
        think_interval_time = self.args.lm_gen_think_interval_time
        for new_text in streamer:
            times.append(perf_counter() - start)
            if is_output_think is False:
                if "</think>" in new_text:
                    new_text = new_text.replace("</think>", "").strip("\n")
                    is_output_think = True
                else:
                    if think_interval_time > 0 and sum(times) > think_interval_time:  # tip once
                        think_interval_time = 0
                        yield "思考中，请稍等。"
                    else:
                        yield None
                    start = perf_counter()
                    continue
            generated_text += new_text
            yield new_text
            start = perf_counter()
        yield "."  # end the sentence for downstream process sentence, e.g.: tts
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()
        tmp_list = generated_text.split("</think>")
        if len(tmp_list) > 1:
            generated_text = tmp_list[1].strip("\n")
        self._chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]}
        )
