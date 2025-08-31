import argparse
import logging
import threading

try:
    from src.types.llm.ctranslate2 import (
        Ctranslate2ModelArgs,
        Ctranslate2EngineArgs,
        LMGenerateArgs,
    )
    import ctranslate2
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[ctranslate2]`")
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlmGenerator
from src.core.llm.base import BaseLLM
from src.common.session import Session


class Ctranslate2Generator(BaseLLM, ILlmGenerator):
    """
    token_ids -> llm generate stream -> token_ids
    https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html
    """

    TAG = "llm_ctranslate2_generator"

    def __init__(self, **kwargs):
        self.args = Ctranslate2EngineArgs(**kwargs)
        self.gen_args = LMGenerateArgs(**self.args.gen_args)
        self.serv_args = Ctranslate2ModelArgs(**self.args.model_args)

        logging.info(
            f"server args: {self.serv_args.__dict__} | default generate args: {self.gen_args.__dict__}"
        )

        # check if the current thread is the main thread,
        # engine must be initialized in the main thread with signal
        assert threading.current_thread() == threading.main_thread()
        self.engine = ctranslate2.Generator(
            self.serv_args.model_path,
            device=self.serv_args.device,
            device_index=self.serv_args.device_index,
            compute_type=self.serv_args.compute_type,
            inter_threads=self.serv_args.inter_threads,
            intra_threads=self.serv_args.intra_threads,
            max_queued_batches=self.serv_args.max_queued_batches,
            flash_attention=self.serv_args.flash_attention,
            tensor_parallel=self.serv_args.tensor_parallel,
        )

    def close(self):
        logging.info(f"{self.__class__.__name__} close")

    async def generate(self, session: Session, **kwargs):
        assert session.ctx.state["tokens"] is not None
        assert isinstance(session.ctx.state["tokens"], list)
        start_tokens = session.ctx.state["tokens"]

        # https://opennmt.net/CTranslate2/generation.html
        # https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html#ctranslate2.Generator.generate_tokens
        step_results = self.engine.generate_tokens(
            start_tokens,
            end_token=kwargs.get("stop_tokens") or self.gen_args.lm_gen_stops or "</s>",
            max_length=kwargs.get("max_new_tokens") or self.gen_args.lm_gen_max_new_tokens,
            min_length=kwargs.get("min_new_tokens") or self.gen_args.lm_gen_min_new_tokens,
            sampling_temperature=kwargs.get("temperature") or self.gen_args.lm_gen_temperature,
            sampling_topk=kwargs.get("top_k") or self.gen_args.lm_gen_top_k,
            sampling_topp=kwargs.get("top_p") or self.gen_args.lm_gen_top_p,
            repetition_penalty=kwargs.get("repetition_penalty")
            or self.gen_args.lm_gen_repetition_penalty,
        )
        for step_result in step_results:
            # print(step_result)
            yield step_result.token_id


"""

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen/Qwen2.5-0.5B-Instruct
# linux
ct2-transformers-converter --model ./models/Qwen/Qwen2.5-0.5B-Instruct --output_dir ./models/Qwen/Qwen2.5-0.5B-Instruct_ctranslate

python -m src.core.llm.ctranslate2.generator
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import asyncio
    import time
    from transformers import AutoTokenizer

    generator = Ctranslate2Generator(
        **Ctranslate2EngineArgs(
            model_args=Ctranslate2ModelArgs(
                model_path="./models/Qwen/Qwen2.5-0.5B-Instruct_ctranslate"
            ).__dict__,
        ).__dict__
    )

    tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen2.5-0.5B-Instruct")

    async def run():
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        token_ids = tokenizer.encode("hello, my name is")
        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        session.ctx.state["tokens"] = start_tokens
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
