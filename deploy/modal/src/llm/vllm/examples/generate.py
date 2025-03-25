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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# PP need close v1
vllm_image = vllm_image.env(
    {
        "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
    }
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


"""
modal run src/llm/vllm/examples/generate.py::run_sync (no stream)
modal run src/llm/vllm/examples/generate.py::run_async (stream)
"""


# @app.local_entrypoint()
# def main():
#    run_sync.remote()
