import modal

app = modal.App("ds-flash-mla")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

ds_flash_mla_image = (
    # https://hub.docker.com/r/pytorch/pytorch/tags
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel", add_python="3.11")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/deepseek-ai/FlashMLA.git",
        "cd FlashMLA && python setup.py install",
    )
    .pip_install()
)

BENCH_DIR = "/data/bench"
bench_dir = modal.Volume.from_name("bench", create_if_missing=True)


# modal run src/bench/run_flash_mla.py::compute_cap
@app.function(
    # gpu=["T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100"],
    gpu="H100",
    max_inputs=1,  # new container each input, so we re-roll the GPU dice every time
)
async def compute_cap():
    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(gpu)
    return gpu


@app.function(
    gpu="H100",
    retries=0,
    image=ds_flash_mla_image,
    volumes={BENCH_DIR: bench_dir},
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def flash_mla(bench_name="ds_flash_mla_bench") -> str:
    import subprocess
    import os

    cmd = "python tests/test_flash_mla.py".split(" ")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/FlashMLA")
    print(result)
    with open(os.path.join(BENCH_DIR, f"{bench_name}.log"), "w") as f:
        f.write(result.stdout)


# modal run src/bench/run_flash_mla.py
@app.local_entrypoint()
def main(bench_name="ds_flash_mla_bench"):
    flash_mla.remote(bench_name)
