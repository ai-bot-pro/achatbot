import modal

bench_vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "unzip", "wget")
    .pip_install("vllm==0.7.3", "pandas", "datasets")
    .run_commands(
        "git clone https://github.com/vllm-project/vllm.git",
    )
)

MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"

BENCH_DIR = "/data/bench"
bench_dir = modal.Volume.from_name("bench", create_if_missing=True)


app = modal.App("vllm-bench-client")

MINUTES = 60  # seconds


# https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md
@app.function(
    image=bench_vllm_image,
    # gpu=f"T4",
    cpu=8.0,
    volumes={
        BENCH_DIR: bench_dir,
    },
    retries=0,
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
)
def bench(url: str, num_prompts: int = 2):
    import subprocess
    import os
    import time

    result = subprocess.run(
        [
            "python",
            "/vllm/benchmarks/benchmark_serving.py",
            "--backend",
            "vllm",
            "--base-url",
            url,
            "--model",
            MODEL_NAME,
            "--dataset-name",
            "sharegpt",
            "--dataset-path",
            "/data/bench/ShareGPT_V3_unfiltered_cleaned_split.json",
            "--profile",
            "--num-prompts",
            num_prompts,
        ],
        capture_output=True,
        text=True,
        cwd="/vllm",
    )
    print(result)
    with open(os.path.join(BENCH_DIR, f"ShareGPT_V3_unfiltered_cleaned_split_bench.log"), "w") as f:
        f.write(result.stdout)


"""
modal run src/llm/vllm/bench/client.py --url https://weedge--vllm-bench-serve-dev.modal.run
modal run src/llm/vllm/bench/client.py --url https://weedge--vllm-bench-serve-dev.modal.run --num-prompts 100
modal run src/llm/vllm/bench/client.py --url https://weedge--vllm-bench-serve-dev.modal.run --num-prompts 1000
"""


@app.local_entrypoint()
def main(url="http://localhost:8000", num_prompts=2):
    bench.remote(url, num_prompts)
