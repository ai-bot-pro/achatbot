import logging
import uuid

try:
    from src.types.llm.tensorrt_llm import TensorRTLLMEngineArgs, LMGenerateArgs, LlmArgs
    from tensorrt_llm import LLM, SamplingParams
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[trtllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlmGenerator
from src.core.llm.base import BaseLLM
from src.common.session import Session


class TrtLLMGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use trtllm engine frontend asyncio api to generate token_ids
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """

    TAG = "llm_trtllm_generator"

    def __init__(self, **kwargs):
        self.args = TensorRTLLMEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.17.0/tensorrt_llm/llmapi/llm_utils.py#L368
        self.serv_args = LlmArgs.from_kwargs(**self.args.serv_args)
        # https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.__init__
        # Load HF model, convert to TensorRT, build TensorRT engine, load TensorRT engine
        self.engine = LLM(**self.serv_args.to_dict())

    async def generate(self, session: Session, **kwargs):
        """
        Generate new tokens using the LLM model.
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]

        # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.17.0/tensorrt_llm/sampling_params.py
        sampling_params = SamplingParams(
            n=1,
            seed=kwargs.get("seed") or self.gen_args.lm_gen_seed,
            max_tokens=kwargs.get("max_new_tokens") or self.gen_args.lm_gen_max_new_tokens,
            temperature=kwargs.get("temperature") or self.gen_args.lm_gen_temperature,
            top_p=kwargs.get("top_p") or self.gen_args.lm_gen_top_p,
            top_k=kwargs.get("top_k") or self.gen_args.lm_gen_top_k,
            # min_p need version > 0.17.0
            # min_p=kwargs.get("min_p") or self.gen_args.lm_gen_min_p,
            # Penalizers,
            repetition_penalty=kwargs.get("repetition_penalty")
            or self.gen_args.lm_gen_repetition_penalty,
            min_tokens=kwargs.get("min_new_tokens") or self.gen_args.lm_gen_min_new_tokens,
            stop_token_ids=kwargs.get("stop_ids") or self.gen_args.lm_gen_stop_ids,
            stop=kwargs.get("stop_tokens") or self.gen_args.lm_gen_stops,
            detokenize=False,
        )
        # https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
        generator = self.engine.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=True,
        )
        async for part in generator:
            if part.outputs:
                token_id = part.outputs[0].token_ids[-1]
                yield token_id


class TrtLLMRunnerGenerator:
    """
    token_ids -> llm generate stream -> token_ids
    use trtllm engine runtime runner to generate token_ids
    - https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner
    - https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunnerCpp
    """

    TAG = "llm_trtllm_runner_generator"


"""
MODEL=./models/Qwen/Qwen2.5-0.5B python -m src.core.llm.tensorrt_llm.generator 
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import os
    import asyncio
    import time
    from transformers import AutoTokenizer

    model = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
    generator = TrtLLMGenerator(
        **TensorRTLLMEngineArgs(serv_args=LlmArgs(model=model).to_dict()).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

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
