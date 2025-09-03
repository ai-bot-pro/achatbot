import datetime
import os
from pathlib import Path
import sys
import asyncio
import subprocess
import threading
import json
from time import perf_counter
import time


import modal


app = modal.App("seed-x_ctranslate2")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
APP_NAME = os.getenv("APP_NAME", "")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "ctranslate2",
        "transformers[torch]",
    )
    .env(
        {
            "LLM_MODEL": os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B"),
        }
    )
)

img = img.pip_install(
    "achatbot==0.0.24.post2",
    extra_index_url="https://test.pypi.org/simple/",
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)

with img.imports():
    import torch
    import ctranslate2
    import transformers

    MODEL_PATH = os.getenv("LLM_MODEL", "ByteDance-Seed/Seed-X-PPO-7B")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]
    ctranslate2_model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH + "_ctranslate2")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    secrets=[modal.Secret.from_name("achatbot")],
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def transformers_convert(**kwargs):
    quantization = kwargs.get("quantization", "")
    cmd = f"ct2-transformers-converter --model {model_path} --output_dir {ctranslate2_model_path}"
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        cmd = f"ct2-transformers-converter --model {model_path} --quantization {quantization} --output_dir {ctranslate2_model_path}.{quantization}"
    print(cmd)
    subprocess.run(cmd, shell=True)


def tokenize(**kwargs):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    print(tokenizer)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    token_ids = tokenizer.encode(messages)
    print(token_ids)

    tokens = tokenizer.decode(token_ids)
    print(tokens)

    start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(start_tokens)


def generate(**kwargs):
    print(f"{device=}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    quantization = kwargs.get("quantization", "")
    model_dir = ctranslate2_model_path
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        model_dir = f"{ctranslate2_model_path}.{quantization}"
    generator = ctranslate2.Generator(
        model_dir,
        device=device,
    )

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese simplified(简体中文):\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese simplified(简体中文) and explain it in detail:\nMay the force be with you <zh>",
    ]
    for message in messages:
        token_ids = tokenizer.encode(message)
        print(token_ids)

        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(start_tokens)

        # https://opennmt.net/CTranslate2/generation.html
        # https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html#ctranslate2.Generator.generate_tokens
        step_results = generator.generate_tokens(
            start_tokens,
            end_token=tokenizer.eos_token,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
        )

        first = True
        start_time = time.perf_counter()
        for step_result in step_results:
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            print(step_result)
            if step_result.token_id == tokenizer.bos_token_id:
                continue
            if step_result.token_id == tokenizer.eos_token_id:
                break
            print(step_result.token, end="", flush=True)


def generate_batch(**kwargs):
    print(f"{device=}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    quantization = kwargs.get("quantization", "")
    model_dir = ctranslate2_model_path
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        model_dir = f"{ctranslate2_model_path}.{quantization}"
    generator = ctranslate2.Generator(model_dir, device=device)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        # "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]

    def generation_callback(step_result: ctranslate2.GenerationStepResult) -> bool:
        print(step_result)
        return False

    for message in messages:
        token_ids = tokenizer.encode(message)
        print(token_ids)

        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(start_tokens)

        # use generate_batch to beam search, and callback to iter step
        # https://opennmt.net/CTranslate2/python/ctranslate2.Generator.html#ctranslate2.Generator.generate_batch
        res = generator.generate_batch(
            [start_tokens],
            max_batch_size=2,
            include_prompt_in_result=False,
            end_token=tokenizer.eos_token,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
            # beam_search
            beam_size=2,
            # callback=generation_callback, # beam_size=1 for greedy decoding, need callback
        )
        print(res)


async def run_achatbot_generator(**kwargs):
    import uuid
    import asyncio
    import time

    from transformers import AutoTokenizer
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.core.llm.ctranslate2.generator import (
        Ctranslate2Generator,
        Ctranslate2EngineArgs,
        Ctranslate2ModelArgs,
    )

    quantization = kwargs.get("quantization", "")
    model_dir = ctranslate2_model_path
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        model_dir = f"{ctranslate2_model_path}.{quantization}"

    generator = Ctranslate2Generator(
        **Ctranslate2EngineArgs(
            model_args=Ctranslate2ModelArgs(
                model_path=model_dir,
                device=device,
            ).__dict__,
        ).__dict__
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        # without CoT
        "Translate the following English sentence into Chinese:\nMay the force be with you <zh>",
        # with CoT
        "Translate the following English sentence into Chinese and explain it in detail:\nMay the force be with you <zh>",
    ]
    for message in messages:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        token_ids = tokenizer.encode(message)
        start_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        session.ctx.state["tokens"] = start_tokens
        start_time = time.perf_counter()
        first = True
        async for token_id in generator.generate(session, max_new_tokens=512):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            print(token_id, gen_text)


async def run_achatbot_processor(**kwargs):
    import uuid
    import asyncio
    import time

    from apipeline.pipeline.pipeline import Pipeline
    from apipeline.pipeline.runner import PipelineRunner
    from apipeline.pipeline.task import PipelineTask, PipelineParams
    from apipeline.processors.logger import FrameLogger
    from apipeline.frames import TextFrame, EndFrame
    from transformers import AutoTokenizer

    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.types.llm.ctranslate2 import (
        Ctranslate2ModelArgs,
        Ctranslate2EngineArgs,
    )
    from achatbot.core.llm import LLMEnvInit
    from achatbot.types.frames import TranslationStreamingFrame, TranslationFrame
    from achatbot.processors.translation.llm_translate_processor import LLMTranslateProcessor
    from achatbot.common.logger import Logger
    from achatbot.types.frames import TranscriptionFrame

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    quantization = kwargs.pop("quantization", "")
    model_dir = ctranslate2_model_path
    if quantization:
        # https://opennmt.net/CTranslate2/quantization.html
        model_dir = f"{ctranslate2_model_path}.{quantization}"

    generator = LLMEnvInit.initLLMEngine(
        tag="llm_ctranslate2_generator",
        kwargs=Ctranslate2EngineArgs(
            model_args=Ctranslate2ModelArgs(
                model_path=model_dir,
                device=device,
            ).__dict__,
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    src = kwargs.pop("src", "en")
    target = kwargs.pop("target", "zh")
    streaming = kwargs.pop("streaming", False)
    prompt_tpl = kwargs.pop("prompt_tpl", "seed-x")
    tl_processor = LLMTranslateProcessor(
        tokenizer=tokenizer,
        generator=generator,
        src=src,
        target=target,
        streaming=streaming,
        prompt_tpl=prompt_tpl,
    )

    task = PipelineTask(
        Pipeline(
            [
                FrameLogger(include_frame_types=[TextFrame]),
                tl_processor,
                FrameLogger(include_frame_types=[TranslationStreamingFrame, TranslationFrame]),
            ]
        ),
        params=PipelineParams(allow_interruptions=False),
    )
    await task.queue_frames(
        [
            TranscriptionFrame(
                user_id="",
                text="May the force be with you",
                timestamp="2025-08-27T15:24:31.687+00:00",
                language="zn",
            ),
            # TextFrame(text="May the force be with you"),
            # TextFrame(
            #    text="We are excited to introduce Seed-X, a powerful series of open-source multilingual translation language models, including an instruction model, a reinforcement learning model, and a reward model. It pushes the boundaries of translation capabilities within 7 billion parameters. We develop Seed-X as an accessible, off-the-shelf tool to support the community in advancing translation research and applications"
            # ),
        ]
    )

    async def end():
        await asyncio.sleep(10)
        await task.stop_when_done()

    asyncio.create_task(end())

    runner = PipelineRunner()
    await runner.run(task)


"""
modal run src/llm/ctranslate2/seed_x.py --task tokenize
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert --quantization int8
modal run src/llm/ctranslate2/seed_x.py --task transformers_convert --quantization int8_bfloat16

# CPU 
modal run src/llm/ctranslate2/seed_x.py --task generate
modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8
modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8_bfloat16

# L4 GPU
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate --quantization int8_bfloat16

# offline batch with beam search
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task generate_batch

# achatbot generator
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task run_achatbot_generator
# achatbot processor
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task run_achatbot_processor 
IMAGE_GPU=L4 modal run src/llm/ctranslate2/seed_x.py --task run_achatbot_processor --streaming
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
    quantization: str = "",
    # processor
    src: str = "en",
    target: str = "zh",
    streaming: bool = False,
):
    print(task)
    tasks = {
        "tokenize": tokenize,
        "transformers_convert": transformers_convert,
        "generate": generate,
        "generate_batch": generate_batch,
        "run_achatbot_generator": run_achatbot_generator,
        "run_achatbot_processor": run_achatbot_processor,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        quantization=quantization,
        src=src,
        target=target,
        streaming=streaming,
    )
