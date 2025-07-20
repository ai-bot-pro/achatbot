import modal
import os

app = modal.App("download_assets")

# Define the dependencies for the function using a Modal Image.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")  # Install curl for downloading files
    .pip_install("huggingface_hub")
)

ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=2,
    cpu=2.0,
    image=download_image,
    volumes={ASSETS_DIR: assets_dir},
    timeout=1200,
    scaledown_window=1200,
)
def download_assets(asset_urls: str) -> str:
    """
    Downloads files from the given URLs and saves them to the ASSETS_DIR.

    Args:
        asset_urls: A comma-separated string of URLs to download.

    Returns:
        A message indicating that the downloads are complete.
    """
    import subprocess
    import logging
    from urllib.parse import unquote

    logging.basicConfig(level=logging.INFO)

    for url in asset_urls.split(","):
        url = url.strip()
        if not url:
            continue

        filenames = url.split("/assets/")
        filename = os.path.basename(url) if len(filenames) < 2 else filenames[-1]
        local_path = os.path.join(ASSETS_DIR, filename)
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

    return "All asset downloads complete."


"""
modal run -e achatbot src/download_assets.py --asset-urls "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/speakers/Tingting%E5%93%BC%E5%94%B1_prompt.wav"

modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/SWivid/F5-TTS/refs/heads/main/src/f5_tts/infer/examples/vocab.txt"
modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/SWivid/F5-TTS/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/SWivid/F5-TTS/refs/heads/main/src/f5_tts/infer/examples/basic/basic_ref_zh.wav"

modal run src/download_assets.py --asset-urls "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/test/audio_files/asr_example_zh.wav"

modal run src/download_assets.py --asset-urls "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_assets.tar"
"""


@app.local_entrypoint()
def main(asset_urls: str):
    """
    Local entry point to trigger the asset download function.

    Args:
        asset_urls: A comma-separated string of URLs to download.
    """
    download_assets.remote(asset_urls)
