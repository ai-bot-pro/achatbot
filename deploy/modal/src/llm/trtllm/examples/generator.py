# https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L548

# https://nvidia.github.io/TensorRT-LLM/index.html (nice doc)
"""
NeMo -------------
                  |
HuggingFace ------
                  |   convert                             build                    load
Modelopt ---------  ----------> TensorRT-LLM Checkpoint --------> TensorRT Engine ------> TensorRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
"""

import os
import modal

app_name = "qwen2.5-0.5B"

app = modal.App("trtllm-generator")

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        "tensorrt-llm==0.17.0.post1",
        # "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        # "tensorrt==10.8.0.43",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({})
)

MAX_BATCH_SIZE = 1024  # better throughput at larger batch sizes, limited by GPU RAM
MODEL_ID = "Qwen/Qwen2.5-0.5B"

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image.env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0"}),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run_sync():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate
    """
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine
    llm = LLM(model=MODEL_ID)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # Print the outputs.
    for output in outputs:
        print(output)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image.env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0"}),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async_stream():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2, detokenize=False)

    # load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine
    llm = LLM(model=MODEL_ID)

    for i, prompt in enumerate(prompts):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for output in generator:
            print(output)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image.env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0"}),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async_batch_stream():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    import asyncio
    import uuid

    from tensorrt_llm import LLM, SamplingParams

    # Prompts to generate
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=2, detokenize=True, return_perf_metrics=True
    )

    # Load HF model, convert to TensorRT, build TensorRT engine, load TensorRT engine
    llm = LLM(model=MODEL_ID)

    lock = asyncio.Lock()

    async def run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex)):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for item in generator:
            print(f"[{request_id}] tokenId: {item.outputs[0].token_ids[-1]} {item} ")
            async with lock:
                print(f"[{request_id}] {item}")
                print(item.outputs[0].token_ids[-1])
                # u can send this response to a request queue/channle

    tasks = [
        run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex))
        for prompt in prompts
    ]
    await asyncio.gather(*tasks)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image.env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0"}),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def runner_stream():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2, detokenize=False)

    # load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine
    llm = LLM(model=MODEL_ID)

    for i, prompt in enumerate(prompts):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for output in generator:
            print(output)


"""
# llmapi (LLM load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine)
modal run src/llm/trtllm/examples/generator.py::run_sync (no stream | batch prompts processing | throughput)
modal run src/llm/trtllm/examples/generator.py::run_async_stream (stream | single prompt async processing | latency)
modal run src/llm/trtllm/examples/generator.py::run_async_batch_stream (stream | batch prompts async processing | latency+throughput) (multiple prompts/request or one prompt/request)

# runner (load tensorrt engine to run generate)

"""
