import os
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-http-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install("numpy", "soundfile", "requests")
image = image.run_commands(
    "wget 'https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/example/prompt_audio.wav' -O /prompt_audio.wav",
    "ls -lh /prompt_audio.wav",
)
image = image.env({})

with image.imports():
    import numpy as np
    import soundfile as sf
    import requests

TTS_GEN_AUDIO_DIR = "/tts_gen_audio"
tts_gen_audio_vol = modal.Volume.from_name("tts_gen_audio", create_if_missing=True)


@app.function(
    cpu=8.0,
    retries=0,
    image=image,
    volumes={
        TTS_GEN_AUDIO_DIR: tts_gen_audio_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def http_cli(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "output.wav",
):
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{model_name}/infer"
    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration

    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    data = prepare_request(samples, reference_text, target_text, duration, sample_rate=sr)

    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": "0"},
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)

    audio_dir = os.path.join(TTS_GEN_AUDIO_DIR, app_name)
    os.makedirs(audio_dir, exist_ok=True)
    output_audio = os.path.join(audio_dir, output_audio)
    sf.write(output_audio, audio, 16000, "PCM_16")


"""
modal run src/llm/trtllm/tts_spark/client_http.py \
    --server-url "https://weedge--tritonserver-serve-dev.modal.run"
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "output.wav",
):
    http_cli.remote(
        server_url, reference_audio, reference_text, target_text, model_name, output_audio
    )


def prepare_request(
    waveform,
    reference_text,
    target_text,
    duration: float,
    sample_rate=16000,
    padding_duration: int = None,
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    if padding_duration:
        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate * ((int(duration) // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform
    else:
        samples = waveform

    samples = samples.reshape(1, -1).astype(np.float32)

    data = {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist(),
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [target_text]},
        ]
    }

    return data
