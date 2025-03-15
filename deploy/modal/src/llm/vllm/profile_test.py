import os
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "kmod")
    .run_commands(
        # "git clone https://github.com/weedge/vllm.git",
        # "cd vllm &&  pip install -r requirements-cuda.txt",
    )
    .pip_install(
        "vllm==0.7.3",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)


# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_profile = modal.Volume.from_name("vllm_profile", create_if_missing=True)

PROFILE_DIR = "/root/vllm_profile"

vllm_image = vllm_image.env({"VLLM_USE_V1": "1", "VLLM_TORCH_PROFILER_DIR": PROFILE_DIR})

app = modal.App("vllm-profile")

IMAGE_GPU = os.getenv("IMAGE_GPU", "L4")

MINUTES = 60  # seconds


@app.function(
    image=vllm_image,
    gpu=IMAGE_GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        PROFILE_DIR: vllm_profile,
    },
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
)
def profile():
    import time
    import sys

    sys.path.insert(1, "/vllm")

    from vllm import LLM, SamplingParams

    os.environ["VLLM_TORCH_PROFILER_DIR"] = PROFILE_DIR

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model="facebook/opt-125m", tensor_parallel_size=1)

    print("Starting profiler...")
    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    print("Stop profiler...")
    llm.stop_profile()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


# modal run src/llm/vllm/profiling_test.py
@app.local_entrypoint()
def main():
    profile.remote()
