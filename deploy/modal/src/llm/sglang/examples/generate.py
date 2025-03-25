import os
import modal

app = modal.App("sglang-generate")

sglang_image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install(  # add sglang and some Python dependencies
    # as per sglang website: https://sgl-project.github.io/start/install.html
    "flashinfer-python",
    "sglang[all]>=0.4.4.post1",
    extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/",
    extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "T4"),
    cpu=2.0,
    retries=0,
    image=sglang_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run_sync():
    from sglang import Engine

    engine = Engine(
        model_path="/root/models/Qwen/Qwen2.5-0.5B",
    )

    iterator = engine.generate(  # no request_id param :(
        prompt="Hello, my name is",
        sampling_params={
            "n": 1,  # number of samples to generate
            "max_new_tokens": 2,
            "temperature": 0.95,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            # Penalizers
            "repetition_penalty": 1.1,
            "min_new_tokens": 0,
        },
        stream=True,
        return_logprob=True,
    )
    i = 1
    for part in iterator:
        print(part)
        meta_info = part["meta_info"]
        if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"]) > 0:
            token_id = meta_info["output_token_logprobs"][-1][1]
            print(token_id)
        if i == 3:
            print("stopping")
            break
        i += 1


@app.function(
    gpu=os.getenv("IMAGE_GPU", "T4"),
    cpu=2.0,
    retries=0,
    image=sglang_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async():
    import uuid
    from sglang import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput

    engine = Engine(
        model_path="/root/models/Qwen/Qwen2.5-0.5B",
    )

    obj = GenerateReqInput(
        text="Hello, my name is",
        rid=str(uuid.uuid4().hex),
        sampling_params={
            "n": 1,  # number of samples to generate
            "max_new_tokens": 2,
            "temperature": 0.95,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            # Penalizers
            "repetition_penalty": 1.1,
            "min_new_tokens": 0,
        },
        stream=True,
        return_logprob=True,
    )
    generator = engine.tokenizer_manager.generate_request(obj, None)

    i = 1
    async for part in generator:
        print(part)
        meta_info = part["meta_info"]
        if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"]) > 0:
            token_id = meta_info["output_token_logprobs"][-1][1]
            print(token_id)
        if i == 3:
            print("stopping")
            break
        i += 1


"""
modal run src/llm/sglang/examples/generate.py::run_sync (stream)
modal run src/llm/sglang/examples/generate.py::run_async (stream)
"""
