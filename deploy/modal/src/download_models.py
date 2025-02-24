import modal

app = modal.App("download_models")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install()
    .run_commands()
    .pip_install("huggingface_hub")
)

MODEL_DIR = "/root/models"
model_dir = modal.Volume.from_name("models", create_if_missing=True)

# ASSETS_DIR = "/root/assets"
# assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=2,
    cpu=8.0,
    image=download_image,
    volumes={MODEL_DIR: model_dir},
    timeout=1200,
    container_idle_timeout=1200,
)
def download_ckpt(repo_ids: str) -> str:
    import os

    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    for repo_id in repo_ids.split(","):
        local_dir = os.path.join(MODEL_DIR, repo_id)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns="*",
            local_dir=local_dir,
        )
        print(f"{repo_id} model to dir:{MODEL_DIR} done")


@app.local_entrypoint()
def main(repo_ids: str):
    download_ckpt.remote(repo_ids)
