# https://github.com/triton-inference-server/client/tree/main/src/python/examples

import os
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-http-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install("numpy", "soundfile", "tritonclient[grpc]")
image = image.run_commands(
    "wget 'https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/example/prompt_audio.wav' -O /prompt_audio.wav",
    "ls -lh /prompt_audio.wav",
)
image = image.env({})

with image.imports():
    import numpy as np
    import soundfile as sf

    import tritonclient
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    from tritonclient.utils import InferenceServerException


TTS_GEN_AUDIO_DIR = "/tts_gen_audio"
tts_gen_audio_vol = modal.Volume.from_name("tts_gen_audio", create_if_missing=True)


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    volumes={
        TTS_GEN_AUDIO_DIR: tts_gen_audio_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def tts(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "output.wav",
):
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def health(server_url: str, verbose: bool = False):
    url = server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    try:
        triton_client = grpcclient.InferenceServerClient(url=url, verbose=verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        return

    # Health
    if not triton_client.is_server_live(headers={"test": "1", "dummy": "2"}):
        print("FAILED : is_server_live")
        return
    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        return

    # Metadata
    metadata = triton_client.get_server_metadata()
    if not (metadata.name == "triton"):
        print("FAILED : get_server_metadata")
        return
    print(metadata)

    # Health
    for model_name in [
        "audio_tokenizer",
        "tensorrt_llm",
        "vocoder",
        "spark_tts",
    ]:
        if not triton_client.is_model_ready(model_name):
            print("FAILED : is_model_ready")

        print("-" * 20)

        metadata = triton_client.get_model_metadata(model_name, headers={"test": "1", "dummy": "2"})
        if not (metadata.name == model_name):
            print("FAILED : get_model_metadata")
            return
        print(metadata)


"""
modal run src/llm/trtllm/tts_spark/client_grpc.py \
    --action health \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_grpc.py \
    --action tts \
    --server-url "weedge--tritonserver-serve-dev.modal.run"
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "output.wav",
    action: str = "health",
):
    print("[!NOTE] grpc now don't to support for modal")
    return
    if action == "tts":
        tts.remote(
            server_url, reference_audio, reference_text, target_text, model_name, output_audio
        )
    else:
        health.remote(server_url, verbose=True)
