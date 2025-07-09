import logging
import uuid


try:
    from fastdeploy import LLM, SamplingParams
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[fastdeploy]`")
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlmGenerator
from src.common.session import Session
from src.core.llm.base import BaseLLM


class FastdeployGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use fastdeploy llm engine to generate token_ids
    https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md
    the same as tensorrt_llm :)
    """

    TAG = "llm_fastdeploy_generator"

    def __init__(self, **kwargs):
        self.args = FastdeployEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://docs.fastdeploy.ai/en/stable/serving/engine_args.html#engine-args
        self.serv_args = AsyncEngineArgs(**self.args.serv_args)
        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.serv_args)

    async def generate(self, session: Session, **kwargs):
        """
        Generate new tokens using the LLM model.
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]


"""
MODEL=./models/ERNIE-4.5-0.3B python -m src.core.llm.fastdeploy.generator 
MODEL=./models/ERNIE-4.5-0.3B python -m src.core.llm.fastdeploy.generator 

"""
if __name__ == "__main__":
    import uuid
    import os
    import asyncio
    import time

    from transformers import AutoTokenizer
    from src.common.types import SessionCtx

    model = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
    generator = FastdeployGenerator(
        **FastdeployEngineArgs(serv_args=AsyncEngineArgs(model=model).__dict__).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        first = True
        start_time = time.perf_counter()
        async for token_id in generator.generate(session, max_new_tokens=20, stop_ids=[13]):
            if first:
                ttft = time.perf_counter() - start_time
                logging.info(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)

    asyncio.run(run())
