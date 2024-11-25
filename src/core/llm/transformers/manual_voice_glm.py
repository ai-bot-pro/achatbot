import logging
from threading import Thread
from queue import Queue

import torch
try:
    from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer
    from transformers.generation.streamers import BaseStreamer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        f"In order to use GLM-Voice, you need to `pip install achatbot[llm_transformers_manual_voice_glm]`,"
        f"use Int4 precision with 4-bit quantization, need to `pip install achatbot[llm_transformers_manual_voice_glm,bitsandbytes]`")
    raise Exception(f"Missing module: {e}")


from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class TransformersManualVoicGLM(TransformersBaseLLM):
    """
    text / speech-tokens prompt -> glm -> (text/speech) output tokens
    with TransformersLMArgs, if use int4, need to install bitsandbytes
    """
    TAG = "llm_transformers_manual_voice_glm"

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if self.args.lm_torch_dtype == "int4" else None

        if self.args.lm_device_map:
            self._model = AutoModel.from_pretrained(
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
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=self.args.lm_torch_dtype,
                attn_implementation=self.args.lm_attn_impl,
                quantization_config=self.bnb_config if self.bnb_config else None,
                trust_remote_code=True,
            ).eval().to(self.args.lm_device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True)
        self._streamer = TokenStreamer(skip_prompt=True)

        # self.warmup()

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt.strip()
        inputs = f"<|user|>\n{dummy_input_text}<|assistant|>streaming_transcription\n"
        model_inputs = self._tokenizer(
            [inputs], return_tensors="pt").to(self._model.device)

        warmup_gen_kwargs = dict(
            model_inputs,
            streamer=self._streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            do_sample=self.args.lm_gen_do_sample,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
        )

        self._warmup(target=self._model.generate, kwargs=warmup_gen_kwargs)

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session):
        """
        (text/speech prompt) input text + speech tokens -> glm -> (text/speech) output tokens
        """
        prompt = session.ctx.state['prompt']
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        generation_kwargs = dict(
            **inputs,
            streamer=self._streamer,
            do_sample=self.args.lm_gen_do_sample,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens)
        thread = Thread(
            target=self._model.generate,
            kwargs=generation_kwargs
        )
        thread.start()

        for token_id in self._streamer:
            yield token_id
