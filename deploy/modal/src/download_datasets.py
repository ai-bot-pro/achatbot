import modal

app = modal.App("download_datasets")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .run_commands()
    .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"},  # hf-transfer for faster downloads
    )
)

HF_DATASET_DIR = "/datasets"
hf_dataset_vol = modal.Volume.from_name("datasets", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=0,
    cpu=8.0,
    image=download_image,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={HF_DATASET_DIR: hf_dataset_vol},
    timeout=1200,
    scaledown_window=1200,
)
def download_datasets(repo_ids: str, allow_patterns: str = "*"):
    import os

    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    for repo_id in repo_ids.split(","):
        local_dir = os.path.join(HF_DATASET_DIR, repo_id)
        print(f"{repo_id} datasets downloading")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            # ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            local_dir=local_dir,
            max_workers=8,
        )
        print(f"{repo_id} datasets to dir:{HF_DATASET_DIR} done")


"""
modal run src/download_datasets.py --repo-ids "VITA-MLLM/VITA-Audio-Data" --allow-patterns "*fixie-ai*"

"""


@app.local_entrypoint()
def main(repo_ids: str, allow_patterns: str = "*"):
    download_datasets.remote(repo_ids, allow_patterns)
