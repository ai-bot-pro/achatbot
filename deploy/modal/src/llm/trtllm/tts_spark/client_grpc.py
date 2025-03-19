# https://github.com/triton-inference-server/client/tree/main/src/python/examples

import os
import uuid
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-grpc-client")

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

    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype


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
    output_audio: str = "grpc_output.wav",
    verbose: bool = False,
):
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=verbose)
    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    inputs = prepare_grpc_sdk_request(
        samples, reference_text, target_text, duration, sample_rate=sr
    )

    # https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    outputs = [grpcclient.InferRequestedOutput("waveform")]
    response = triton_client.infer(
        model_name,
        inputs,
        request_id=str(uuid.uuid4()),
        outputs=outputs,
    )
    audio = response.as_numpy("waveform").reshape(-1)
    audio = np.array(audio, dtype=np.float32)

    audio_dir = os.path.join(TTS_GEN_AUDIO_DIR, app_name)
    os.makedirs(audio_dir, exist_ok=True)
    output_audio = os.path.join(audio_dir, output_audio)
    sf.write(output_audio, audio, 16000, "PCM_16")
    print(f"save audio to {output_audio}")

    triton_client.close()


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def health(server_url: str, verbose: bool = False):
    url = server_url

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
    --server-url "r15.modal.host:44161"

modal run src/llm/trtllm/tts_spark/client_grpc.py \
    --action tts \
    --server-url "r15.modal.host:44161"
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "grpc_output.wav",
    action: str = "health",
    verbose: bool = False,
):
    if action == "tts":
        tts.remote(
            server_url,
            reference_audio,
            reference_text,
            target_text,
            model_name,
            output_audio,
            verbose=verbose,
        )
    else:
        health.remote(server_url, verbose=verbose)


def prepare_grpc_sdk_request(
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

    """
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
    """
    inputs = [
        grpcclient.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        grpcclient.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
        grpcclient.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)

    input_data_numpy = np.array([reference_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)

    input_data_numpy = np.array([target_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[3].set_data_from_numpy(input_data_numpy)

    return inputs
