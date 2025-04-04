import logging
import uuid

import torch

try:
    from src.types.llm.tensorrt_llm import (
        TensorRTLLMEngineArgs,
        TensorRTLLMRunnerArgs,
        TensorRTLLMRunnerEngineArgs,
        LMGenerateArgs,
        LlmArgs,
    )
    from tensorrt_llm import LLM, SamplingParams, mpi_rank
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
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
        logging.info(
            f"before server args: {self.serv_args.to_dict()} | default generate args: {self.gen_args.__dict__}"
        )
        # https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py#L520
        kv_cache_config = KvCacheConfig()
        for key, value in self.serv_args.kv_cache_config.items():
            if hasattr(KvCacheConfig, key):
                setattr(kv_cache_config, key, value)
        self.serv_args.kv_cache_config = kv_cache_config
        logging.info(f"after server args: {self.serv_args.to_dict()}")
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


class TrtLLMRunnerGenerator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    use trtllm engine runtime runner to generate token_ids
    - https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner
    - https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunnerCpp
    """

    TAG = "llm_trtllm_runner_generator"

    def __init__(self, **kwargs):
        self.args = TensorRTLLMRunnerEngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        self.serv_args = TensorRTLLMRunnerArgs(**self.args.serv_args)
        self.serv_args.rank = mpi_rank()  # for multi gpu
        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )
        # load tensorrt engine
        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.from_dir
        self.engine = ModelRunner.from_dir(**self.serv_args.__dict__)

    async def generate(self, session: Session, **kwargs):
        """
        Generate new tokens using the LLM model.
        """
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        input_ids = session.ctx.state["token_ids"]
        input_ids = kwargs.pop("input_ids", input_ids)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(session.ctx.state["token_ids"])
            assert input_ids.dim() <= 2
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to("cuda")
        # https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/runtime/generation.html#SamplingConfig
        sampling_config = SamplingConfig(
            end_id=self.gen_args.lm_gen_end_id,
            pad_id=self.gen_args.lm_gen_pad_id,
            temperature=self.gen_args.lm_gen_temperature,
            repetition_penalty=self.gen_args.lm_gen_repetition_penalty,
            max_new_tokens=self.gen_args.lm_gen_max_new_tokens,
            top_k=self.gen_args.lm_gen_top_k,
            top_p=self.gen_args.lm_gen_top_p,
            # min_p version > 0.17.0
            # min_p=self.gen_args.lm_gen_min_p,
        )
        sampling_config.update(**kwargs)
        logging.debug(f"sampling_config:{sampling_config}")
        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.generate
        generator = self.engine.generate(
            input_ids,
            streaming=True,
            sampling_config=sampling_config,
        )
        stop_ids = kwargs.get("stop_ids", self.gen_args.lm_gen_stop_ids)
        for output in generator:
            output = output[:, 0]
            mask = output != sampling_config.pad_id
            output = output[mask]
            token_id = output[-1].item()
            yield token_id
            if token_id in stop_ids:
                break


"""
MODEL=./models/Qwen/Qwen2.5-0.5B python -m src.core.llm.tensorrt_llm.generator 

ENGINE=llm_trtllm_runner_generator ENGINE_DIR=./models/Qwen/Qwen2.5-0.5B-trtllm \
    python -m src.core.llm.tensorrt_llm.generator 
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import os
    import asyncio
    import time
    from transformers import AutoTokenizer

    engine_name = os.getenv("ENGINE", "llm_trtllm_generator")
    if engine_name == "llm_trtllm_generator":
        model = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B")
        generator = TrtLLMGenerator(
            **TensorRTLLMEngineArgs(serv_args=LlmArgs(model=model).to_dict()).__dict__,
        )
    if engine_name == "llm_trtllm_runner_generator":
        engine_dir = os.getenv("ENGINE_DIR", "./models/Qwen/Qwen2.5-0.5B-trtllm")
        generator = TrtLLMRunnerGenerator(
            **TensorRTLLMRunnerEngineArgs(
                serv_args=TensorRTLLMRunnerArgs(engine_dir=engine_dir)
            ).__dict__,
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
