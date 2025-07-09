import os
import modal

app = modal.App("download_models")

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
def download_ckpt(repo_ids: str, revision: str = None, local_dir: str = None) -> str:
    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    for repo_id in repo_ids.split(","):
        if local_dir is None:
            local_dir = os.path.join(HF_MODEL_DIR, repo_id)
        else:
            local_dir = os.path.join(HF_MODEL_DIR, local_dir)
        print(f"{repo_id} model downloading")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            allow_patterns="*",
            # ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            local_dir=local_dir,
            max_workers=8,
        )
        print(f"{repo_id} model to dir:{HF_MODEL_DIR} done")

    hf_model_vol.commit()


@app.function(
    # gpu="T4",
    retries=0,
    cpu=2.0,
    image=download_image,
    volumes={HF_MODEL_DIR: hf_model_vol},
    timeout=1200,
    scaledown_window=1200,
)
def download_ckpts(ckpt_urls: str) -> str:
    import os
    import subprocess
    import logging
    from urllib.parse import unquote

    logging.basicConfig(level=logging.INFO)

    for url in ckpt_urls.split(","):
        url = url.strip()
        if not url:
            continue

        filename = os.path.basename(url)
        local_path = os.path.join(HF_MODEL_DIR, filename)
        local_path = unquote(local_path)
        dir_path = os.path.dirname(local_path)
        os.makedirs(dir_path, exist_ok=True)

        try:
            if os.path.exists(local_path):
                logging.info(f"File {filename} already exists. download again.")

            logging.info(f"Downloading {url} to {local_path}")
            # Use curl to download the file with progress bar
            # `-O` to save the file with the same name as the URL
            # `-L` to follow redirects
            # `-#` to show progress bar
            result = subprocess.run(
                ["curl", "-L", "-#", url, "-o", local_path],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"Download of {url} complete.")
            logging.debug(f"curl output:{result.stdout}")
            # Check if the downloaded file is a zip or tar archive and extract it
            if filename.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    zip_ref.extractall(dir_path)
                os.remove(local_path)
                logging.info(f"Extracted and removed {filename}")
            elif (
                filename.endswith(".tar")
                or filename.endswith(".tar.gz")
                or filename.endswith(".tgz")
            ):
                import tarfile

                with tarfile.open(local_path, "r") as tar_ref:
                    tar_ref.extractall(dir_path)
                os.remove(local_path)
                logging.info(f"Extracted and removed {filename}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Error downloading {url}: {e}")
            logging.error(f"curl stderr: {e.stderr}")

    hf_model_vol.commit()

    return "All ckpt downloads complete."


"""
modal run src/download_models.py --repo-ids "stepfun-ai/Step-Audio-Chat,stepfun-ai/Step-Audio-TTS-3B,stepfun-ai/Step-Audio-Tokenizer"

modal run src/download_models.py --repo-ids "SWivid/F5-TTS"
modal run src/download_models.py --repo-ids "charactr/vocos-mel-24khz"

modal run src/download_models.py --repo-ids "SparkAudio/Spark-TTS-0.5B"
modal run src/download_models.py --repo-ids "mradermacher/SparkTTS-LLM-GGUF"

modal run src/download_models.py --repo-ids "canopylabs/orpheus-3b-0.1-ft"
modal run src/download_models.py --repo-ids "hubertsiuzdak/snac_24khz"

modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B"

modal run src/download_models.py --repo-ids "FunAudioLLM/SenseVoiceSmall"

SECRET_NAME=achatbot modal run src/download_models.py --repo-ids "google/gemma-3-1b-it,google/gemma-3-4b-it,google/gemma-3-12b-it,google/gemma-3-27b-it"

modal run src/download_models.py::download_ckpts --ckpt-urls "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
modal run src/download_models.py::download_ckpts --ckpt-urls "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip"

modal run src/download_models.py::download_ckpts --ckpt-urls "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar"
"""


@app.local_entrypoint()
def main(repo_ids: str, revision: str = None, local_dir: str = None):
    download_ckpt.remote(repo_ids, revision, local_dir)
