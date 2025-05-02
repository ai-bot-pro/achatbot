import logging
import os
import sys
from threading import Thread

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation.logits_process import LogitsProcessor
    from transformers.generation.utils import LogitsProcessorList
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Step-Audio-TTS, you need to `pip install achatbot[llm_transformers_manual_speech_step]`. "
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class RepetitionAwareLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1

        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size

        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


class TransformersManualSpeechStep(TransformersBaseLLM):
    """
    system prompt + (one shot: text->speech(audio vq code) prompt) + tts prompt -> tokenizer encode -> token ids -> StepForCausalLM -> audio vq tokens
    with TransformersLMArgs
    """

    TAG = "llm_transformers_manual_speech_step"
    DEFAULT_SYS_PROMPT = "Convert the text to speech"

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("args: %s", self.args)

        self.load_torch_optimus()

        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                trust_remote_code=True,
            ).eval()
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

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )
        self.end_token_id = 3
        end_token_ids = self._tokenizer.encode("<|EOT|>")
        if len(end_token_ids) >= 1:
            self.end_token_id = end_token_ids[-1]

        self.sys_prompt = self.DEFAULT_SYS_PROMPT

        self.warmup()

    def load_torch_optimus(self):
        # load optimus_ths for flash attention, make sure LD_LIBRARY_PATH has `nvidia/cuda_nvrtc/lib`
        # if not, please manually set LD_LIBRARY_PATH=xxx/python3.10/site-packages/nvidia/cuda_nvrtc/lib
        try:
            if torch.__version__ >= "2.5":
                torch.ops.load_library(
                    os.path.join(
                        self.args.lm_model_name_or_path,
                        "lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so",
                    )
                )
            elif torch.__version__ >= "2.3":
                torch.ops.load_library(
                    os.path.join(
                        self.args.lm_model_name_or_path,
                        "lib/liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so",
                    )
                )
            elif torch.__version__ >= "2.2":
                torch.ops.load_library(
                    os.path.join(
                        self.args.lm_model_name_or_path,
                        "lib/liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so",
                    )
                )
            print("Load optimus_ths successfully and flash attn would be enabled")
        except Exception as err:
            print(f"Fail to load optimus_ths and flash attn is disabled: {err}")

    def set_system_prompt(self, **kwargs):
        # session sys settings
        self.sys_prompt = kwargs.get("sys_prompt", self.sys_prompt)

    @torch.inference_mode()
    def warmup(self):
        if self.args.warmup_steps < 1:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")
        dummy_input_text = self.args.warnup_prompt.strip()
        prompt = f"<s><|BOT|><s> system\n{self.sys_prompt}"
        prompt += f"<|EOT|><|BOT|><s> human\n{dummy_input_text}"
        prompt += "<|EOT|><|BOT|><s> assistant\n"
        token_ids = self._tokenizer.encode(prompt)

        logging.debug(f"prompt:{prompt}")
        logging.debug(f"token_ids:{token_ids}")
        logging.debug(f"args:{self.args}")
        logging.debug(f"end_token_id:{self.end_token_id}")

        # inference token streamer
        streamer = TokenStreamer(skip_prompt=True)

        warmup_gen_kwargs = dict(
            input_ids=torch.tensor([token_ids]).to(torch.long).to(self._model.device),
            eos_token_id=self.end_token_id,
            streamer=streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            do_sample=True if self.args.lm_gen_temperature > 0.0 else False,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
            # repetition_penalty=self.args.lm_gen_repetition_penalty,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        system prompt + (one shot: text->speech(audio code) prompt) + tts prompt -> tokenizer encode -> token ids -> step lm -> audio vq tokens
        """
        prompt = session.ctx.state.get("prompt", "")
        token_ids = self._tokenizer.encode(prompt)
        logging.debug(f"prompt:{prompt}")
        logging.debug(f"tfeat/voiceoken_ids:{token_ids}")
        logging.debug(f"args:{self.args}")
        logging.debug(f"kwargs:{kwargs}")
        logging.debug(f"end_token_id:{self.end_token_id}")

        # inference token streamer
        streamer = TokenStreamer(skip_prompt=True)

        # inference token streamer
        generation_kwargs = dict(
            input_ids=torch.tensor([token_ids]).to(torch.long).to(self._model.device),
            eos_token_id=self.end_token_id,
            streamer=streamer,
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            do_sample=True if self.args.lm_gen_temperature > 0.0 else False,
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            temperature=kwargs.get("temperature", self.args.lm_gen_temperature),
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
            # repetition_penalty=self.args.lm_gen_repetition_penalty,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_id in streamer:
            yield token_id
