import modal

app = modal.App("download_models")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install()
    .run_commands()
    .pip_install("hf-transfer", "huggingface_hub")
    .env(
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"},  # hf-transfer for faster downloads
    )
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)

# ASSETS_DIR = "/root/assets"
# assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=0,
    cpu=8.0,
    image=download_image,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={HF_MODEL_DIR: hf_model_vol},
    timeout=1200,
    scaledown_window=1200,
)
def download_ckpt(repo_ids: str) -> str:
    import os

    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    for repo_id in repo_ids.split(","):
        local_dir = os.path.join(HF_MODEL_DIR, repo_id)
        print(f"{repo_id} model downloading")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns="*",
            # ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            local_dir=local_dir,
            max_workers=8,
        )
        print(f"{repo_id} model to dir:{HF_MODEL_DIR} done")


"""
modal run src/download_models.py --repo-ids "stepfun-ai/Step-Audio-Chat,stepfun-ai/Step-Audio-TTS-3B,stepfun-ai/Step-Audio-Tokenizer"

modal run src/download_models.py --repo-ids "SWivid/F5-TTS"
modal run src/download_models.py --repo-ids "charactr/vocos-mel-24khz"

modal run src/download_models.py --repo-ids "SparkAudio/Spark-TTS-0.5B"
modal run src/download_models.py --repo-ids "mradermacher/SparkTTS-LLM-GGUF"

modal run src/download_models.py --repo-ids "canopylabs/orpheus-3b-0.1-ft"
modal run src/download_models.py --repo-ids "hubertsiuzdak/snac_24khz"

modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B"
"""


@app.local_entrypoint()
def main(repo_ids: str):
    download_ckpt.remote(repo_ids)
