import logging
from threading import Thread
from queue import Queue

import torch

try:
    from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use GLM-Voice, you need to `pip install achatbot[llm_transformers_manual_voice_glm]`,"
        "use Int4 precision with 4-bit quantization, need to `pip install achatbot[llm_transformers_manual_voice_glm,bitsandbytes]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersManualVoicGLM(TransformersBaseLLM):
    """
    text / speech-tokens prompt -> glm -> (text/speech) output tokens
    with TransformersLMArgs, if use int4, need to install bitsandbytes
    """

    TAG = "llm_transformers_manual_voice_glm"
    DEFAULT_SYS_PROMPT = "User will provide you with a speech or text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        logging.info("TransformersManualVoicGLM args: %s", self.args)
        # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
        self.bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # load bfloat16 tensor to int-4bit tensor,
                #  torch_dtype see: config.json
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if self.args.lm_bnb_quant_type == "int4"
            else None
        )

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
            self._model = (
                AutoModel.from_pretrained(
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

        # self.warmup()

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt.strip()
        # NOTE: must use system prompt.
        inputs = f"<|system|>\n{self.DEFAULT_SYS_PROMPT}<|user|>\n{dummy_input_text}<|assistant|>streaming_transcription\n"
        model_inputs = self._tokenizer([inputs], return_tensors="pt").to(self._model.device)

        # inference token streamer
        streamer = TokenStreamer(skip_prompt=True)
        warmup_gen_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            do_sample=self.args.lm_gen_do_sample,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
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
