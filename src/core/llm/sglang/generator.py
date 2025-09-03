import argparse
import logging
import threading
import uuid
import asyncio
import time

try:
    from sglang import Engine
    from src.types.llm.sglang import SGLangEngineArgs, ServerArgs, LMGenerateArgs
    from sglang.srt.managers.io_struct import GenerateReqInput
    from transformers import AutoTokenizer
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[sglang]`")
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlmGenerator
from src.core.llm.base import BaseLLM
from src.common.session import Session
from src.common.types import SessionCtx


class SGlangGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use sglang llm engine frontend asyncio api to generate token_ids
    https://docs.sglang.ai/basic_usage/offline_engine_api.html
    todo: maybe use sglang llm engine backend runtime method to generate token_ids
    """

    TAG = "llm_sglang_generator"

    def __init__(self, **kwargs):
        self.args = SGLangEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        self.serv_args = ServerArgs(**self.args.serv_args)

        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )

        # check if the current thread is the main thread,
        # engine must be initialized in the main thread with signal
        assert threading.current_thread() == threading.main_thread()
        self.engine = Engine(**self.serv_args.__dict__)

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warmup_prompt:
            logging.info("no warmup!")
            return

        import nest_asyncio

        nest_asyncio.apply()

        tokenizer = AutoTokenizer.from_pretrained(self.serv_args.model_path)
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

        async def run():
            for step in range(self.args.warmup_steps):
                session.ctx.state["token_ids"] = tokenizer.encode(self.args.warmup_prompt)
                start_time = time.perf_counter()
                first = True
                ttft = 0
                async for token_id in self.generate(session, max_new_tokens=64):
                    if first:
                        ttft = time.perf_counter() - start_time
                        first = False
                logging.info(
                    f"{step=} generate TTFT time: {ttft:.3f} s | total time: {time.perf_counter() - start_time:.3f} s"
                )

        asyncio.run(run())

    def close(self):
        logging.info(f"{self.__class__.__name__} close")
        self.engine.shutdown()

    async def generate(self, session: Session, **kwargs):
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]

        # https://docs.sglang.ai/backend/sampling_params.html
        sampling_params = {
            "n": 1,  # number of samples to generate
            "max_new_tokens": kwargs.get("max_new_tokens")
            if kwargs.get("max_new_tokens")
            else self.gen_args.lm_gen_max_new_tokens,
            "temperature": kwargs.get("temperature") or self.gen_args.lm_gen_temperature,
            "top_p": kwargs.get("top_p") or self.gen_args.lm_gen_top_p,
            "top_k": kwargs.get("top_k") or self.gen_args.lm_gen_top_k,
            "min_p": kwargs.get("min_p") or self.gen_args.lm_gen_min_p,
            # Penalizers
            "repetition_penalty": kwargs.get("repetition_penalty")
            or self.gen_args.lm_gen_repetition_penalty,
            "min_new_tokens": kwargs.get("min_new_tokens") or self.gen_args.lm_gen_min_new_tokens,
            "stop_token_ids": kwargs.get("stop_ids") or self.gen_args.lm_gen_stop_ids,
            "stop": kwargs.get("stop_tokens") or self.gen_args.lm_gen_stops,
        }
        obj = GenerateReqInput(
            input_ids=token_ids,
            sampling_params=sampling_params,
            rid=session.ctx.client_id,
            stream=True,
            return_logprob=True,
        )
        iterator = self.engine.tokenizer_manager.generate_request(obj, None)

        async for part in iterator:
            meta_info = part["meta_info"]
            if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"]) > 0:
                token_id = meta_info["output_token_logprobs"][-1][1]
                yield token_id


"""
python -m src.core.llm.sglang.generator --model-path ./models/Qwen/Qwen2.5-0.5B
python -m src.core.llm.sglang.generator --model-path ./models/Qwen/Qwen2.5-0.5B-Instruct
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    generator = SGlangGenerator(
        **SGLangEngineArgs(
            serv_args=server_args.__dict__,
        ).__dict__
    )

    tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        start_time = time.perf_counter()
        first = True
        async for token_id in generator.generate(session, max_new_tokens=3):
            if first:
                ttft = time.perf_counter() - start_time
                logging.info(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)

    asyncio.run(run())
