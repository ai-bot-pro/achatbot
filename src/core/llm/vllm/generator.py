import logging
import uuid


try:
    from src.types.llm.vllm import VllmEngineArgs, AsyncEngineArgs, LMGenerateArgs
    from vllm import AsyncLLMEngine, SamplingParams
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.session import Session
from src.common.device_cuda import CUDAInfo


class VllmGenerator:
    """
    token_ids -> llm generate stream -> token_ids
    use vllm llm engine frontend asyncio api to generate token_ids
    https://docs.vllm.ai/en/stable/models/generative_models.html
    todo: maybe use backend method to generate token_ids
    """

    TAG = "llm_vllm_generator"

    def __init__(self, **kwargs):
        self.args = VllmEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://docs.vllm.ai/en/stable/serving/engine_args.html#engine-args
        self.serv_args = AsyncEngineArgs(**self.args.serv_args)
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
            seed=kwargs.get("seed", self.gen_args.lm_gen_seed),
            max_tokens=kwargs.get(
                "max_new_tokens",
                self.gen_args.lm_gen_max_new_tokens,
            ),
            temperature=kwargs.get("temperature", self.gen_args.lm_gen_temperature),
            top_p=kwargs.get("top_p", self.gen_args.lm_gen_top_p),
            top_k=kwargs.get("top_k", self.gen_args.lm_gen_top_k),
            min_p=kwargs.get("min_p", self.gen_args.lm_gen_min_p),
            # Penalizers,
            repetition_penalty=kwargs.get(
                "repetition_penalty",
                self.gen_args.lm_gen_repetition_penalty,
            ),
            min_tokens=kwargs.get("min_new_tokens", self.gen_args.lm_gen_min_new),
        )
        # https://docs.vllm.ai/en/stable/api/offline_inference/llm.html#vllm.LLM.generate
        iterator = self.engine.generate(
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            request_id=session.ctx.client_id,
        )
        async for part in iterator:
            if part.outputs:
                token_id = part.outputs[0].token_ids[-1]
                yield token_id


"""
MODEL=./models/Qwen/Qwen2.5-0.5B python -m src.core.llm.vllm.generator 
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import os
    import asyncio

    generator = VllmGenerator(
        **VllmEngineArgs(
            serv_args=AsyncEngineArgs(model=os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")).__dict__
        ).__dict__,
    )

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        async for token_id in generator.generate(session, max_new_tokens=3):
            print(token_id)

    asyncio.run(run())
