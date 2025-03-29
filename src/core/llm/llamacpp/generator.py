import logging

from src.common.interface import ILlmGenerator

from . import LLamacppLLM
from src.common.session import Session


class LlamacppGenerator(LLamacppLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    """

    TAG = "llm_llamacpp_generator"

    async def generate(self, session: Session, **kwargs):
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]
        # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.generate
        # AR generate model
        generator = self.model.generate(
            token_ids,
            temp=kwargs.get("temperature", self.args.llm_temperature)
            if kwargs.get("temperature", self.args.llm_temperature)
            else 0.80,
            top_k=kwargs.get("top_k", self.args.llm_top_k)
            if kwargs.get("top_k", self.args.llm_top_k)
            else 50,
            top_p=kwargs.get("top_p", self.args.llm_top_p)
            if kwargs.get("top_p", self.args.llm_top_p)
            else 0.9,
            min_p=kwargs.get("min_p", self.args.llm_min_p)
            if kwargs.get("min_p", self.args.llm_min_p)
            else 0.0,
            repeat_penalty=kwargs.get("repetition_penalty", self.args.llm_repeat_penalty)
            if kwargs.get("repetition_penalty", self.args.llm_repeat_penalty)
            else 1.0,
        )

        # todo: cache for multi generate, or use `create_completion`` metod
        # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion

        stop_ids = kwargs.get("stop_ids", self.args.llm_stop_ids)
        max_new_tokens = (
            kwargs.get("max_new_tokens", self.args.llm_max_tokens)
            if kwargs.get("max_new_tokens", self.args.llm_max_tokens)
            else 20
        )
        if max_new_tokens > self.args.n_ctx - len(token_ids):
            max_new_tokens = self.args.n_ctx - len(token_ids)
        token_cn = 0
        for token_id in generator:
            if token_cn >= max_new_tokens:
                break
            token_cn += 1
            yield token_id
            if token_id in stop_ids:
                break


"""
MODEL=./models/qwen2.5-0.5b-instruct-q8_0.gguf \
    TOKENIZER_PATH=./models/Qwen/Qwen2.5-0.5B-Instruct \
    python -m src.core.llm.llamacpp.generator 
"""
if __name__ == "__main__":
    import uuid
    import os
    import time
    import asyncio

    from transformers import AutoTokenizer
    from src.common.types import SessionCtx
    from src.common.types import LLamcppLLMArgs

    logging.basicConfig(level=logging.INFO)

    tokenizer_path = os.getenv("TOKENIZER_PATH", "./models/Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_path = os.getenv("MODEL", "./models/qwen2.5-0.5b-instruct-q8_0.gguf")
    generator = LlamacppGenerator(**LLamcppLLMArgs(model_path=model_path).__dict__)

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        # test max_new_tokens>n_ctx-len(token_ids)
        gen_iter = generator.generate(session, max_new_tokens=100, stop_ids=[13])
        start_time = time.perf_counter()
        first = True
        async for token_id in gen_iter:
            if first:
                ttft = time.perf_counter() - start_time
                logging.info(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)

    asyncio.run(run())
