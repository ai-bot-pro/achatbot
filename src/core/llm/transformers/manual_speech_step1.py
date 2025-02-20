import logging
import os
import sys
from threading import Thread

import torch

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../StepAudio"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/StepAudio"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deps.StepAudio.tts import LogitsProcessorList, RepetitionAwareLogitsProcessor
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Step-Audio-TTS, you need to `pip install achatbot[llm_transformers_manual_speech_step1]`. "
    )
    raise Exception(f"Missing module: {e}")


from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersManualSpeechStep1(TransformersBaseLLM):
    """
    text / speech-tokens prompt -> Step1ForCausalLM -> (text/speech) output tokens
    with TransformersLMArgs
    """

    TAG = "llm_transformers_manual_speech_step1"

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        logging.info("args: %s", self.args)

        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                quantization_config=self.bnb_config if self.bnb_config else None,
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
                    quantization_config=self.bnb_config if self.bnb_config else None,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        # inference token streamer
        self.streamer = TokenStreamer(skip_prompt=True)

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps < 1:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")
        # dummy_input_text = self.args.warnup_prompt.strip()
        # NOTE: must use system prompt.
        # model_inputs = self._tokenizer([inputs], return_tensors="pt").to(self._model.device)
        model_inputs = (
            torch.tensor([self.test_audio_token_ids]).to(torch.long).to(self._model.device)
        )
        warmup_gen_kwargs = dict(
            input_ids=model_inputs,
            eos_token_id=3,
            streamer=self.streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            do_sample=True if self.args.lm_gen_temperature > 0.0 else False,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
            repetition_penalty=self.args.lm_gen_repetition_penalty,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=self.streamer,
        )

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session):
        """
        (text/speech prompt) input text + speech tokens -> glm -> (text/speech) output tokens
        """
        prompt = session.ctx.state["prompt"]
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        # print(prompt, inputs)
        # inference token streamer
        streamer = TokenStreamer(skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
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

        for token_id in streamer:
            # print(token_id, end=',', flush=True)
            yield token_id
