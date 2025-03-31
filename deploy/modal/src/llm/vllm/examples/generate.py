import os
import modal

app = modal.App("vllm-generate")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0",
        }
    )  # faster model transfers
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# PP need close v1
vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
    }
)

achatbot_vllm_image = vllm_image.pip_install(
    "achatbot==0.0.9.post1",
    extra_index_url="https://pypi.org/simple/",
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=vllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run_sync():
    from vllm import LLM, EngineArgs

    engine = LLM(
        model="/root/models/Qwen/Qwen2.5-0.5B",
        # dtype="half",
        # enforce_eager=True,
    )

    sampling_params = engine.get_default_sampling_params()
    print(sampling_params)
    sampling_params.n = 1
    sampling_params.seed = 42
    sampling_params.max_tokens = 2
    sampling_params.temperature = 0.9
    sampling_params.top_p = 1.0
    sampling_params.top_k = 50
    sampling_params.min_p = 0.0
    # Penalizers
    sampling_params.repetition_penalty = 1.1
    sampling_params.min_tokens = 0
    outputs = engine.generate(
        "Hello, my name is",
        sampling_params=sampling_params,
    )
    print(outputs)
    i = 1
    for part in outputs:
        print(part)
        if i == 3:
            print("stopping")
            break
        i += 1


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=vllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async():
    import asyncio
    import uuid

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    # from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model="/root/models/Qwen/Qwen2.5-0.5B",
            # dtype="half",
            # enforce_eager=True,
        )
    )

    sampling_params = SamplingParams(
        n=1,
        seed=42,
        max_tokens=2,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        min_p=0.0,
        # Penalizers
        repetition_penalty=1.1,
        min_tokens=0,
    )
    print(sampling_params)

    outputs_generator = engine.generate(
        "Hello, my name is", sampling_params=sampling_params, request_id=str(uuid.uuid4().hex)
    )
    i = 1
    async for part in outputs_generator:
        print(part)
        """
        RequestOutput(request_id=a80d0cce8c774cbaada1ae0bb993ddf5, prompt='Hello, my name is', prompt_token_ids=[9707, 11, 847, 829, 374], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' David', token_ids=[6798], cumulative_logprob=None, logprobs=None, finish_reason=None, stop_reason=None)], finished=False, metrics=None, lora_request=None, num_cached_tokens=None, multi_modal_placeholders={})
        """
        if part.outputs:
            print(part.outputs[0].token_ids[-1])
        if i == 3:
            print("stopping")
            break
        i += 1


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=achatbot_vllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_achatbot_generator():
    import uuid
    import os
    import asyncio
    import time

    from achatbot.core.llm.vllm.generator import VllmGenerator, VllmEngineArgs, AsyncEngineArgs
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from transformers import AutoTokenizer, GenerationConfig

    model = "/root/models/Qwen/Qwen2.5-0.5B"
    generator = VllmGenerator(
        **VllmEngineArgs(serv_args=AsyncEngineArgs(model=model).__dict__).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # async def run():
    generation_config = {}
    if os.path.exists(os.path.join(model, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model, "generation_config.json"
        ).to_dict()
    # async def run():
    prompt_cases = [
        {"prompt": "hello, my name is", "kwargs": {"max_new_tokens": 30, "stop_ids": []}},
        {
            "prompt": "hello, my name is",
            "kwargs": {"max_new_tokens": 30, "stop_ids": [13]},
        },  # prefill cache token test
        {"prompt": "hello, what your name?", "kwargs": {"max_new_tokens": 30, "stop_ids": [13]}},
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        session.ctx.state["token_ids"] = tokenizer.encode("hello, my name is")
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens} # hack test, vllm have some bug :)
        gen_kwargs = {**case["kwargs"], **tokens}
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

    # asyncio.run(run())


"""
modal run src/llm/vllm/examples/generate.py::run_sync (no stream)
modal run src/llm/vllm/examples/generate.py::run_async (stream)
modal run src/llm/vllm/examples/generate.py::run_achatbot_generator (stream)
"""


# @app.local_entrypoint()
# def main():
#    run_sync.remote()
