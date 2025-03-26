import logging
from threading import Thread


try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use TTS spark, you need to `pip install achatbot[llm_transformers_manual_speech_spark]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersGenerator(TransformersBaseLLM):
    """
    token_ids -> llm generate stream -> token_ids
    use transformers llm engine to generate token_ids
    """

    TAG = "llm_transformers_generator"

    def __init__(self, **args):
        self.args = TransformersSpeechLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)
        self._model = AutoModelForCausalLM.from_pretrained(self.args.lm_model_name_or_path)
        self._model.eval().to(self.args.lm_device)

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warnup_prompt:
            logging.info("no warmup!")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.lm_model_name_or_path)
        model_inputs = self.tokenizer([self.args.warnup_prompt], return_tensors="pt").to(
            self.args.lm_device
        )

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
    def generate(self, session: Session, **kwargs):
        """
        token_ids -> llm generate stream -> token_ids
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]
        input_ids = torch.IntTensor([token_ids]).to(self.device)

        # https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/text_generation
        streamer = TokenStreamer(skip_prompt=True)
        kwargs["max_length"] = kwargs.get("max_length", self.args.lm_gen_max_length)
        kwargs["max_new_tokens"] = kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens)
        kwargs["top_k"] = kwargs.get("top_k", self.args.lm_gen_top_k)
        kwargs["top_p"] = kwargs.get("top_p", self.args.lm_gen_top_p)
        kwargs["do_sample"] = (
            True if kwargs.get("temperature", self.args.lm_gen_temperature) > 0.0 else False
        )
        kwargs["temperature"] = kwargs.get("temperature", self.args.lm_gen_temperature)
        kwargs["repetition_penalty"] = kwargs.get(
            "repetition_penalty", self.args.lm_gen_repetition_penalty
        )
        kwargs["min_new_tokens"] = kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens)
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            **kwargs,
        )
        logging.debug("generation_kwargs", generation_kwargs)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_id in streamer:
            # print(token_id, end=",", flush=True)
            yield token_id
