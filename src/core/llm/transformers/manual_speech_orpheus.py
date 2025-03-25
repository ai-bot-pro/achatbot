import logging
from threading import Lock, Thread

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[llm_transformers_manual_speech_orpheus]`,")
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersManualSpeechOrpheus(TransformersBaseLLM):
    """
    TTS: text -> llama3 -> vq code tokens
    """

    TAG = "llm_transformers_manual_speech_orpheus"
    DEFAULT_SYS_PROMPT = ""

    def __init__(self, **args):
        self.args = TransformersSpeechLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)
        self._model = AutoModelForCausalLM.from_pretrained(self.args.lm_model_name_or_path)
        self._model.eval().to(self.args.lm_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.args.lm_model_name_or_path)

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warnup_prompt:
            logging.info("no warmup!")
            return

        input_ids = self._tokenizer(self.args.warnup_prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text,
        modified_input_ids = torch.cat(
            [start_token, input_ids, end_tokens], dim=1
        )  # SOH SOT Text EOT EOH
        attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)

        streamer = TokenStreamer(skip_prompt=True)
        warmup_gen_kwargs = dict(
            input_ids=input_ids.to(self.args.lm_device),
            attention_mask=attention_mask.to(self.args.lm_device),
            streamer=streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            do_sample=self.args.lm_gen_do_sample,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            num_return_sequences=1,
            eos_token_id=128258,  # <custom_token_2>
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
        TTS: text -> llama3 -> vq code tokens
        """
        prompt = session.ctx.state["prompt"]  # tts text
        if "vq_code_prompt" in session.ctx.state and isinstance(
            session.ctx.state["vq_code_prompt"], torch.Tensor
        ):
            logging.warning(f"now don't support to process ref audio vq code tokens")

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text,
        modified_input_ids = torch.cat(
            [start_token, input_ids, end_tokens], dim=1
        )  # SOH SOT Text EOT EOH
        attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)

        streamer = TokenStreamer(skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids.to(self.args.lm_device),
            attention_mask=attention_mask.to(self.args.lm_device),
            streamer=streamer,
            min_new_tokens=kwargs["min_new_tokens"]
            if "min_new_tokens" in kwargs
            else self.args.lm_gen_min_new_tokens,
            max_new_tokens=kwargs["max_new_tokens"]
            if "max_new_tokens" in kwargs
            else self.args.lm_gen_max_new_tokens,
            top_k=kwargs["top_k"] if "top_k" in kwargs else self.args.lm_gen_top_k,
            top_p=kwargs["top_p"] if "top_p" in kwargs else self.args.lm_gen_top_p,
            do_sample=kwargs["do_sample"] if "do_sample" in kwargs else self.args.lm_gen_do_sample,
            temperature=kwargs["temperature"]
            if "temperature" in kwargs
            else self.args.lm_gen_temperature,
            repetition_penalty=kwargs["repetition_penalty"]
            if "repetition_penalty" in kwargs
            else self.args.lm_gen_repetition_penalty,
            num_return_sequences=1,
            eos_token_id=128258,  # <custom_token_2>
        )
        # print("generation_kwargs", generation_kwargs)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_id in streamer:
            yield token_id
