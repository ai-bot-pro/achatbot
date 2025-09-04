import os
import sys
import uuid
import time
import asyncio
import subprocess
from time import perf_counter


import modal


app = modal.App("hunyuan7b_trtllm")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm",
        add_python="3.12",  # modal install /usr/local/bin/python3.12.1 or 3.10.13
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .run_commands(
        "/usr/local/bin/python --version",
        "/usr/bin/python --version",
        "echo $PATH",
        "/usr/bin/pip list",
        "update-alternatives --install /usr/local/bin/python python3 /usr/local/bin/python3.12 1",
        "update-alternatives --install /usr/local/bin/python python3 /usr/bin/python3.12 2",
        "python --version",
        "pip list",
    )
    # https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html
    # .run_commands(
    #    "git lfs install",
    #    "git clone https://github.com/NVIDIA/TensorRT-LLM.git",
    #    "cd TensorRT-LLM && git checkout 064eb7a70f29f45a74b5b080aafd0f6a872ed4b5",
    #    "cd TensorRT-LLM && pip install -r requirements.txt",
    # )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "tencent/Hunyuan-MT-7B"),
            "LD_LIBRARY_PATH": "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH",
        }
    )
    .run_commands(
        "cat /modal_requirements.txt",
        "pip list",
        "pip install -r /modal_requirements.txt",
    )
    .pip_install(
        "fastapi==0.115.4",
        "pydantic==2.9.1",
        "cloudpickle>=3.0.0",
    )
)


img = img.pip_install(
    f"achatbot==0.0.24.post3",
    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
).run_commands(
    "ldconfig -p",
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


with img.imports():
    import torch

    # sys.path.insert(0, "/TensorRT-LLM")
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, TrtLlmArgs, LoadFormat

    MODEL_ID = os.getenv("LLM_MODEL", "tencent/Hunyuan-MT-7B")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def generate(**kwargs):
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate
    """

    prompts = [
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n你好<|extra_0|>",
        """<|startoftext|>把下面的文本翻译成English，不要额外解释。\n\n9月3日上午九点，中国的抗日战争胜利纪念日暨世界反法西斯战争胜利80周年纪念活动正式登场。\n 升旗仪式后，中国领导人习近平发表讲话称，“全军将士要忠实履行神圣职责，加快建设世界一流军队，坚决维护国家主权统一、领土完整。为实现中华民族伟大复兴提供战略支撑，为世界和平与发展作出更大贡献。”\n 相比十年之前的“9·3 ”阅兵，习近平此次的讲话并未透露更多信息，当时他在讲话中强调，中国永不称霸，并宣布将裁军30万。\n 十年间，中国经历了被认为是中共建政以来最大力度的军队体制改革，服役人数稳定在 200万人，从七大军区改为五大战区，大量将领被整肃，军费则上涨超过70%。\n 此次纪念大会和阅兵式是十年来第二次“九三阅兵”，也是习近平上任后第三次天安门阅兵。阅兵式上，展示的新式武器更多，尤其是能搭载核弹头的战略导弹，以及大量无人作战武器。\n<|extra_0|>""",
    ]

    # load hf model, flashinfer.jit compile
    # https://nvidia.github.io/TensorRT-LLM/1.0.0rc1/
    llm = LLM(
        model=MODEL_PATH,
        max_batch_size=2,
        max_seq_len=512,
        kv_cache_config={"free_gpu_memory_fraction": 0.5},
    )
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7, top_k=20, top_p=0.6, max_tokens=256, repetition_penalty=1.05
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # Print the outputs.
    for output in outputs:
        print(output)

    llm.shutdown()


async def async_gen_stream(**kwargs):
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n你好<|extra_0|>",
    ]

    # load hf model, flashinfer.jit compile
    # https://nvidia.github.io/TensorRT-LLM/1.0.0rc1/
    llm = LLM(
        model=MODEL_PATH,
        max_batch_size=2,
        max_seq_len=512,
        kv_cache_config={"free_gpu_memory_fraction": 0.5},
    )
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.6,
        repetition_penalty=1.05,
        max_tokens=64,
        detokenize=True,
    )

    for i, prompt in enumerate(prompts):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for output in generator:
            print(output)

    llm.shutdown()


async def async_batch_stream(**kwargs):
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    import asyncio
    import uuid

    from tensorrt_llm import LLM, SamplingParams

    # Prompts to generate
    prompts = [
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n你好<|extra_0|>",
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n奥利给<|extra_0|>",
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n我爱中国<|extra_0|>",
        f"<|startoftext|>把下面的文本翻译成English，不要额外解释。 \n\n9月3日，看了阅兵仪式，东风快递好牛叉！<|extra_0|>",
    ]

    # load hf model, flashinfer.jit compile
    # https://nvidia.github.io/TensorRT-LLM/1.0.0rc1/
    llm = LLM(
        model=MODEL_PATH,
        max_batch_size=4,
        # max_batch_size=2,
        max_seq_len=512,
        kv_cache_config={"free_gpu_memory_fraction": 0.5},
    )

    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.6,
        repetition_penalty=1.05,
        max_tokens=64,
        detokenize=True,
    )

    lock = asyncio.Lock()

    async def run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex)):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for item in generator:
            async with lock:
                print(f"[{request_id}] tokenId: {item.outputs[0].token_ids[-1]} {item} ")
                # u can send this response to a request queue/channle

    tasks = [
        run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex))
        for prompt in prompts
    ]
    await asyncio.gather(*tasks)

    llm.shutdown()


async def run_achatbot_generator(**kwargs):
    from transformers import AutoTokenizer, GenerationConfig

    from achatbot.core.llm.tensorrt_llm.generator import (
        TrtLLMPyTorchGenerator,
        TensorRTLLMEngineArgs,
        LMGenerateArgs,
        LlmArgs,
    )
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    generator = TrtLLMPyTorchGenerator(
        **TensorRTLLMEngineArgs(
            serv_args={
                "model": MODEL_PATH,
                # kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
                "kv_cache_config": {"free_gpu_memory_fraction": 0.7},
            },
            gen_args=LMGenerateArgs(lm_gen_stops=None).__dict__,
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    generation_config = {}
    if os.path.exists(os.path.join(MODEL_PATH, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH, "generation_config.json"
        ).to_dict()

    prompt_cases = [
        {
            "prompt": "<|startoftext|>把下面的文本翻译成English，不要额外解释。\n\n9月3日，看了阅兵仪式，东风快递好牛叉！<|extra_0|>",
            "kwargs": {"max_new_tokens": 64, "stop_ids": [127960]},
        },
        {
            "prompt": """<|startoftext|>把下面的文本翻译成English，不要额外解释。\n\n9月3日上午九点，中国的抗日战争胜利纪念日暨世界反法西斯战争胜利80周年纪念活动正式登场。\n 升旗仪式后，中国领导人习近平发表讲话称，“全军将士要忠实履行神圣职责，加快建设世界一流军队，坚决维护国家主权统一、领土完整。为实现中华民族伟大复兴提供战略支撑，为世界和平与发展作出更大贡献。”\n 相比十年之前的“9·3 ”阅兵，习近平此次的讲话并未透露更多信息，当时他在讲话中强调，中国永不称霸，并宣布将裁军30万。\n 十年间，中国经历了被认为是中共建政以来最大力度的军队体制改革，服役人数稳定在 200万人，从七大军区改为五大战区，大量将领被整肃，军费则上涨超过70%。\n 此次纪念大会和阅兵式是十年来第二次“九三阅兵”，也是习近平上任后第三次天安门阅兵。阅兵式上，展示的新式武器更多，尤其是能搭载核弹头的战略导弹，以及大量无人作战武器。\n<|extra_0|>""",
            "kwargs": {"max_new_tokens": 1024, "stop_ids": [127960]},
        },
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens.pop("input_ids")
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
        gen_kwargs = case["kwargs"]
        print("gen_kwargs:", gen_kwargs)
        first = True
        start_time = time.perf_counter()
        gen_texts = ""
        async for token_id in generator.generate(session, **gen_kwargs):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            gen_texts += gen_text
            print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)

    generator.close()


"""
# https://github.com/Tencent-Hunyuan/Hunyuan-7B#tensorrt-llm

IMAGE_GPU=L4 modal run src/llm/trtllm/hunyuan_7b.py --task generate
IMAGE_GPU=L4 modal run src/llm/trtllm/hunyuan_7b.py --task async_gen_stream
IMAGE_GPU=L4 modal run src/llm/trtllm/hunyuan_7b.py --task async_batch_stream
IMAGE_GPU=L4 modal run src/llm/trtllm/hunyuan_7b.py --task run_achatbot_generator
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
):
    print(task)
    tasks = {
        "generate": generate,
        "async_gen_stream": async_gen_stream,
        "async_batch_stream": async_batch_stream,
        "run_achatbot_generator": run_achatbot_generator,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
