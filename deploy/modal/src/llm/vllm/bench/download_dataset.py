import modal
import os

app = modal.App("download_vllm_bench_dataset")

# Define the dependencies for the function using a Modal Image.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "wget", "unzip")  # Install curl for downloading files
    .pip_install("huggingface_hub")
)

BENCH_DIR = "/data/bench"
bench_dir = modal.Volume.from_name("bench", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=2,
    cpu=2.0,
    image=download_image,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={BENCH_DIR: bench_dir},
    timeout=1200,
    scaledown_window=1200,
)
def download_ShareGPT() -> str:
    """
    Downloads ShareGPT and saves them to the BENCH_DIR.

    Returns:
        A message indicating that the downloads are complete.
    """
    import subprocess
    import logging

    logging.basicConfig(level=logging.INFO)

    os.makedirs(BENCH_DIR, exist_ok=True)

    try:
        subprocess.run(
            [
                "wget",
                "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                "-O",
                "/data/bench/ShareGPT_V3_unfiltered_cleaned_split.json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"Download of ShareGPT complete.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading ShareGPT: {e}")
        logging.error(f"curl stderr: {e.stderr}")

    return "All ShareGPT downloads complete."


@app.function(
    # gpu="T4",
    retries=2,
    cpu=2.0,
    image=download_image,
    volumes={BENCH_DIR: bench_dir},
    # timeout=1200,
    # scaledown_window=1200,
)
def download_ShareGPT4V() -> str:
    """
    Downloads ShareGPT4V and saves them to the BENCH_DIR.

    Returns:
        A message indicating that the downloads are complete.
    """
    import subprocess
    import logging

    logging.basicConfig(level=logging.INFO)

    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(BENCH_DIR + "/coco", exist_ok=True)

    try:
        if os.path.exists("/data/bench/sharegpt4v_instruct_gpt4-vision_cap100k.json"):
            logging.info("File sharegpt4v_instruct_gpt4-vision_cap100k.json already exists.")
        else:
            subprocess.run(
                [
                    "wget",
                    "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json",
                    "-O",
                    "/data/bench/sharegpt4v_instruct_gpt4-vision_cap100k.json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"Download of ShareGPT complete.")
        if os.path.exists("/data/bench/coco/train2017.zip"):
            logging.info("File /data/bench/coco/train2017.zip already exists.")
        else:
            subprocess.run(
                [
                    "wget",
                    "http://images.cocodataset.org/zips/train2017.zip",
                    "-O",
                    "/data/bench/coco/train2017.zip",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        if os.path.exists("/data/bench/coco/train2017.zip.done"):
            logging.info("File /data/bench/coco/train2017.zip.done already exists.")
        else:
            logging.info("unzip /data/bench/coco/train2017.zip")
            subprocess.run(
                [
                    "unzip",
                    "/data/bench/coco/train2017.zip",
                    "-d",
                    "/data/bench/coco/",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            with open("/data/bench/coco/train2017.zip.done", "w"):
                logging.info("unzip /data/bench/coco/train2017.zip done")
                pass

        logging.info(f"Download of ShareGPT4V complete.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading ShareGPT: {e}")
        logging.error(f"curl stderr: {e.stderr}")

    return "All ShareGPT downloads complete."


@app.function(
    # gpu="T4",
    retries=2,
    cpu=2.0,
    image=download_image,
    volumes={BENCH_DIR: bench_dir},
    timeout=1200,
    scaledown_window=1200,
)
def download_BurstGPT() -> str:
    """
    Downloads BurstGPT and saves them to the BENCH_DIR.

    Returns:
        A message indicating that the downloads are complete.
    """
    import subprocess
    import logging

    logging.basicConfig(level=logging.INFO)

    os.makedirs(BENCH_DIR, exist_ok=True)

    try:
        subprocess.run(
            [
                "wget",
                "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv",
                "-O",
                "/data/bench/BurstGPT_without_fails_2.csv",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logging.info(f"Download of BurstGPT complete.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading BurstGPT: {e}")
        logging.error(f"curl stderr: {e.stderr}")

    return "All BurstGPT downloads complete."


"""
modal run src/llm/vllm/bench/download_dataset.py --dataset ShareGPT
modal run src/llm/vllm/bench/download_dataset.py --dataset ShareGPT4V
modal run src/llm/vllm/bench/download_dataset.py --dataset BurstGPT
"""


@app.local_entrypoint()
def main(dataset: str = "ShareGPT"):
    """
    Local entry point to trigger the asset download function.
    """
    if dataset == "ShareGPT":
        download_ShareGPT.remote()
    elif dataset == "ShareGPT4V":
        download_ShareGPT4V.remote()
    elif dataset == "BurstGPT":
        download_BurstGPT.remote()
