import logging
import uuid


try:
    from src.types.llm.vllm import VllmEngineArgs, AsyncEngineArgs, LMGenerateArgs
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlmGenerator
from src.common.session import Session
from src.core.llm.base import BaseLLM


class VllmGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use vllm llm engine frontend asyncio api to generate token_ids
    https://docs.vllm.ai/en/stable/models/generative_models.html
    todo: maybe use backend runtime method to generate token_ids
    """

    TAG = "llm_vllm_generator"

    def __init__(self, **kwargs):
        self.args = VllmEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://docs.vllm.ai/en/stable/serving/engine_args.html#engine-args
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

        # https://docs.vllm.ai/en/stable/api/inference_params.html#vllm.SamplingParams
        sampling_params = SamplingParams(
            n=1,
            seed=kwargs.get("seed") if kwargs.get("seed") else self.gen_args.lm_gen_seed,
            max_tokens=kwargs.get("max_new_tokens")
            if kwargs.get("max_new_tokens")
            else self.gen_args.lm_gen_max_new_tokens,
            temperature=kwargs.get("temperature")
            if kwargs.get("temperature")
            else self.gen_args.lm_gen_temperature,
            top_p=kwargs.get("top_p") if kwargs.get("top_p") else self.gen_args.lm_gen_top_p,
            top_k=kwargs.get("top_k") if kwargs.get("top_k") else self.gen_args.lm_gen_top_k,
            min_p=kwargs.get("min_p") if kwargs.get("min_p") else self.gen_args.lm_gen_min_p,
            # Penalizers,
            repetition_penalty=kwargs.get("repetition_penalty")
            if kwargs.get("repetition_penalty")
            else self.gen_args.lm_gen_repetition_penalty,
            min_tokens=kwargs.get("min_new_tokens")
            if kwargs.get("min_new_tokens")
            else self.gen_args.lm_gen_min_new_tokens,
            stop_token_ids=kwargs.get("stop_ids")
            if kwargs.get("stop_ids")
            else self.gen_args.lm_gen_stop_ids,
            stop=kwargs.get("stop_tokens")
            if kwargs.get("stop_tokens")
            else self.gen_args.lm_gen_stops,
        )
        # https://docs.vllm.ai/en/stable/api/offline_inference/llm.html#vllm.LLM.generate
        iterator = self.engine.generate(
            prompt=TokensPrompt(prompt_token_ids=token_ids),
            sampling_params=sampling_params,
            request_id=session.ctx.client_id,
        )
        async for part in iterator:
            if part.outputs:
                token_id = part.outputs[0].token_ids[-1]
                yield token_id


"""
MODEL=./models/Qwen/Qwen2.5-0.5B python -m src.core.llm.vllm.generator 
MODEL=./models/Qwen/Qwen2.5-0.5B-Instruct python -m src.core.llm.vllm.generator 

"""
if __name__ == "__main__":
    import uuid
    import os
    import asyncio
    import time

    from transformers import AutoTokenizer
    from src.common.types import SessionCtx

    model = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
    generator = VllmGenerator(
        **VllmEngineArgs(serv_args=AsyncEngineArgs(model=model).__dict__).__dict__,
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
