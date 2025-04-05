import logging
from threading import Thread


try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[transformers]`,")
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.common.interface import ILlmGenerator
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersGenerator(TransformersBaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use transformers llm engine to generate token_ids
    """

    TAG = "llm_transformers_generator"

    def __init__(self, **args):
        self.args = TransformersLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)
        # https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
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
            min_new_tokens=0,
            max_new_tokens=3,
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
    async def generate(self, session: Session, **kwargs):
        """
        token_ids -> llm generate stream -> token_ids
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], (list, torch.Tensor))
        input_ids = session.ctx.state["token_ids"]
        input_ids = kwargs.pop("input_ids", input_ids)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(session.ctx.state["token_ids"])
            assert input_ids.dim() <= 2
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.args.lm_device)

        streamer = TokenStreamer(skip_prompt=True)
        if "attention_mask" in kwargs:
            assert isinstance(kwargs["attention_mask"], (list, torch.Tensor))
            if isinstance(kwargs["attention_mask"], list):
                kwargs["attention_mask"] = torch.IntTensor(kwargs["attention_mask"])
                assert kwargs["attention_mask"].dim() <= 2
                if kwargs["attention_mask"].dim() == 1:
                    kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)
            kwargs["attention_mask"] = kwargs["attention_mask"].to(self.args.lm_device)
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
        stop_ids = kwargs.pop("stop_ids", self.args.lm_gen_stop_ids)
        # notice: attention_mask is not used in generation_config
        attention_mask = kwargs.pop("attention_mask", None)
        generation_config = GenerationConfig(**kwargs)
        logging.debug(f"generation_config: {generation_config.to_dict()}")

        # https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/text_generation
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=kwargs.get("logits_processor", None),
            stopping_criteria=kwargs.get("stopping_criteria", None),
            prefix_allowed_tokens_fn=kwargs.get("prefix_allowed_tokens_fn", None),
            synced_gpus=kwargs.get("synced_gpus", None),
            assistant_model=kwargs.get("assistant_model", None),
            negative_prompt_ids=kwargs.get("negative_prompt_ids", None),
            negative_prompt_attention_mask=kwargs.get("negative_prompt_attention_mask", None),
            use_model_defaults=kwargs.get("use_model_defaults", None),
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_id in streamer:
            # print(token_id, end=",", flush=True)
            yield token_id
            if token_id in stop_ids:
                break


"""
MODEL=./models/Qwen/Qwen2.5-0.5B-Instruct python -m src.core.llm.transformers.generator
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import os
    import time
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    model_path = os.getenv("MODEL", "./models/Qwen/Qwen2.5-0.5B")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer("hello, my name is")
    # tokens = tokenizer("hello, my name is", return_tensors="pt")
    # tokens = tokenizer(["hello, my name is"])
    # tokens = tokenizer(["hello, my name is"], return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # generator
    generator = TransformersGenerator(
        **TransformersLMArgs(lm_model_name_or_path=model_path).__dict__
    )

    # generation_config
    generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
    generation_config.max_new_tokens = 30
    print(generation_config.to_dict())

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = input_ids
        first = True
        start_time = time.perf_counter()
        async for token_id in generator.generate(
            session,
            attention_mask=attention_mask,
            stop_ids=[13],
            **generation_config.to_dict(),
        ):
            if first:
                ttft = time.perf_counter() - start_time
                logging.info(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)

    asyncio.run(run())
