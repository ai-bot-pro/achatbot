import logging
import traceback
from typing import Generator
import uuid
import time

from src.types.llm.sampling import LMGenerateArgs
from src.common.interface import ILlmGenerator
from src.common.session import Session
from src.core.llm.base import BaseLLM

try:
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.engine.args_utils import EngineArgs
    from fastdeploy.engine.engine import LLMEngine
    from fastdeploy.engine.request import RequestOutput

    from src.types.llm.fastdeploy import FastDeployEngineArgs
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "you need to see https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"
    )
    raise Exception(f"Missing module: {e}")


class LLMEngineMonkey(LLMEngine):
    def _get_generated_tokens(self, request_id) -> Generator[RequestOutput, None, None]:
        """
        Get generated tokens for a specific request ID.
        This is a generator function that yields results until the generation is complete.

        Args:
            request_id (str): The ID of the request to get tokens for.

        Yields:
            RequestOutput: The generated tokens for the request.
        """
        finished = False
        while not finished and self.running:
            try:
                results = self.scheduler.get_results()
                if request_id in results:
                    contents = results[request_id]
                    for result in contents:
                        # print(request_id, result)
                        yield result
                        if result.finished:
                            finished = True
                            break
                if not finished:
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
            except Exception as e:
                logging.error(f"Error in _get_generated_tokens: {e}, {str(traceback.format_exc())}")
                break


class FastdeployGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use fastdeploy llm engine to generate token_ids
    https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md
    the same as tensorrt_llm :)
    """

    TAG = "llm_fastdeploy_generator"

    def __init__(self, **kwargs):
        self.args = FastDeployEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://docs.fastdeploy.ai/en/stable/serving/engine_args.html#engine-args
        self.serv_args = EngineArgs(**self.args.serv_args)
        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )
        self.engine = LLMEngineMonkey.from_engine_args(self.serv_args)

        if not self.engine.start():
            logging.error("Failed to initialize FastDeploy LLM engine, service exit now!")
            return
        logging.info(f"FastDeploy LLM engine initialized!")

    async def generate(self, session: Session, **kwargs):
        """
        Generate new tokens using the LLM model.
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)

        sampling_params = SamplingParams(
            n=1,
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.gen_args.lm_gen_repetition_penalty
            ),
            temperature=kwargs.get("temperature", self.gen_args.lm_gen_temperature),
            # top_k=kwargs.get("top_k", self.gen_args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.gen_args.lm_gen_top_p),
            max_tokens=kwargs.get("max_tokens", self.gen_args.lm_gen_max_tokens),
            reasoning_max_tokens=kwargs.get(
                "reasoning_max_tokens", self.gen_args.lm_gen_reasoning_max_tokens
            ),
            stop=kwargs.get("stop", self.gen_args.lm_gen_stops),
            stop_token_ids=kwargs.get("stop_token_ids", self.gen_args.lm_gen_stop_ids),
        )
        task = kwargs  # enable_thinking defualt True
        task["prompt_token_ids"] = session.ctx.state["token_ids"]
        task["request_id"] = session.ctx.client_id
        self.engine.add_requests(task, sampling_params)

        for result in self.engine._get_generated_tokens(task["request_id"]):
            if result.outputs and result.outputs.token_ids:
                yield result.outputs.token_ids

            if result.finished:
                break


"""
MODEL=./models/baidu/ERNIE-4.5-0.3B python -m src.core.llm.fastdeploy.generator 
MODEL=./models/baidu/ERNIE-4.5-VL-28B-A3B-Paddle python -m src.core.llm.fastdeploy.generator 
"""
if __name__ == "__main__":
    import uuid
    import os
    import asyncio
    import time

    from transformers import AutoTokenizer
    from src.common.types import SessionCtx

    model = os.getenv("MODEL", "baidu/ERNIE-4.5-0.3B")
    generator = FastdeployGenerator(
        **FastDeployEngineArgs(serv_args=EngineArgs(model=model).__dict__).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        first = True
        start_time = time.perf_counter()
        async for token_id in generator.generate(session, max_tokens=128, stop_token_ids=[23]):
            if first:
                ttft = time.perf_counter() - start_time
                logging.info(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)

    asyncio.run(run())
