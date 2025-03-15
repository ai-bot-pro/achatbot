# https://github.com/triton-inference-server/client/tree/main/src/python/examples

import os
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-http-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install("numpy", "soundfile", "requests", "tritonclient[http]")
image = image.run_commands(
    "wget 'https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/example/prompt_audio.wav' -O /prompt_audio.wav",
    "ls -lh /prompt_audio.wav",
)
image = image.env({})

with image.imports():
    import numpy as np
    import soundfile as sf
    import requests
    import json
    import uuid

    # import tritonclient.http as httpclient
    import tritonclient.http.aio as httpclient
    from tritonclient.utils import InferenceServerException
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
        params={"request_id": uuid.uuid4()},
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)

    audio_dir = os.path.join(TTS_GEN_AUDIO_DIR, app_name)
    os.makedirs(audio_dir, exist_ok=True)
    output_audio = os.path.join(audio_dir, output_audio)
    sf.write(output_audio, audio, 16000, "PCM_16")
    print(f"save audio to {output_audio}")


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
async def cli_sdk_tts(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    output_audio: str = "output.wav",
    verbose: bool = False,
):
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=verbose)
    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    inputs = prepare_http_sdk_request(
        samples, reference_text, target_text, duration, sample_rate=sr
    )

    # https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    outputs = [httpclient.InferRequestedOutput("waveform")]
    response = await triton_client.infer(
        model_name,
        inputs,
        request_id=str(uuid.uuid4()),
        outputs=outputs,
        headers={"Content-Type": "application/json"},
    )
    audio = response.as_numpy("waveform").reshape(-1)
    audio = np.array(audio, dtype=np.float32)

    audio_dir = os.path.join(TTS_GEN_AUDIO_DIR, app_name)
    os.makedirs(audio_dir, exist_ok=True)
    output_audio = os.path.join(audio_dir, output_audio)
    sf.write(output_audio, audio, 16000, "PCM_16")
    print(f"save audio to {output_audio}")

    await triton_client.close()


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
def health_meta_statics(server_url: str, verbose: bool = False):
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    # Health
    resp = requests.get(f"{server_url}/v2/health/live")
    assert resp.status_code == 200, "don't live"

    resp = requests.get(f"{server_url}/v2/health/ready")
    assert resp.status_code == 200, "don't ready"

    resp = requests.get(f"{server_url}/v2")
    assert resp.status_code == 200, f"no serve meta info"
    res = json.dumps(resp.json(), indent=2)
    print(res)

    for model_name in [
        "audio_tokenizer",
        "tensorrt_llm",
        "vocoder",
        "spark_tts",
    ]:
        resp = requests.get(f"{server_url}/v2/models/{model_name}/ready")
        assert resp.status_code == 200, f"{model_name} don't ready"
        print(f"{model_name} meta" + ("-" * 20))
        resp = requests.get(f"{server_url}/v2/models/{model_name}")
        assert resp.status_code == 200, f"no {model_name} meta info"
        res = json.dumps(resp.json(), indent=2)
        print(res)

        print(f"{model_name} statics" + ("-" * 20))
        resp = requests.get(f"{server_url}/v2/models/{model_name}/stats")
        assert resp.status_code == 200, f"no {model_name} statics info"
        res = json.dumps(resp.json(), indent=2)
        print(res)

    print("all statics" + ("-" * 20))
    resp = requests.get(f"{server_url}/v2/models/stats")
    assert resp.status_code == 200, f"no statics info"
    res = json.dumps(resp.json(), indent=2)
    print(res)


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def cli_sdk_health_meta_statics(server_url: str, verbose: bool = False):
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=verbose)

    # Health
    res = await triton_client.is_server_live(query_params={"test_1": 1, "test_2": 2})
    assert res is True, print("FAILED : is_server_live")

    res = await triton_client.is_server_ready()
    assert res is True, print("FAILED : is_server_ready")

    # Metadata
    metadata = await triton_client.get_server_metadata()
    assert metadata["name"] == "triton", print("FAILED : get_server_metadata")
    # print(json.dumps(metadata, indent=2))

    for model_name in [
        "audio_tokenizer",
        "tensorrt_llm",
        "vocoder",
        "spark_tts",
    ]:
        res = await triton_client.is_model_ready(model_name)
        assert res is True, print("FAILED : is_model_ready")

        metadata = await triton_client.get_model_metadata(
            model_name, query_params={"test_1": 1, "test_2": 2}
        )
        assert metadata["name"] == model_name, print("FAILED : get_model_metadata")
        print(json.dumps(metadata, indent=2))

    # Passing incorrect model name
    try:
        metadata = await triton_client.get_model_metadata("wrong_model_name")
    except InferenceServerException as ex:
        assert "Request for unknown model" in ex.message(), print(
            "FAILED : get_model_metadata wrong_model_name"
        )
    else:
        print("FAILED : get_model_metadata wrong_model_name")
        return

    # inference statistics
    stats = await triton_client.get_inference_statistics()
    # print(json.dumps(stats, indent=2))

    for model_name in [
        "audio_tokenizer",
        "tensorrt_llm",
        "vocoder",
        "spark_tts",
    ]:
        stats = await triton_client.get_inference_statistics(model_name=model_name)
        print(json.dumps(stats, indent=2))

    await triton_client.close()


"""
modal run src/llm/trtllm/tts_spark/client_http.py \
    --action health_meta_statics \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_health_meta_statics \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action tts \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_tts \
    --output-audio cli_sdk_tts_output.wav \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

modal run src/llm/trtllm/tts_spark/client_http.py \
    --action cli_sdk_tts \
    --output-audio cli_sdk_tts_output.wav \
    --target-text "你好，请将一个儿童故事，文字不少于200字，故事充满童趣和奇妙幻想。" \
    --output-audio "child_story.wav" \
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
    action: str = "health_meta_statics",
):
    if action == "tts":
        tts.remote(
            server_url,
            reference_audio,
            reference_text,
            target_text,
            model_name,
            output_audio,
        )
    elif action == "cli_sdk_health_meta_statics":
        cli_sdk_health_meta_statics.remote(server_url, verbose=False)
    elif action == "cli_sdk_tts":
        cli_sdk_tts.remote(
            server_url,
            reference_audio,
            reference_text,
            target_text,
            model_name,
            output_audio,
            verbose=False,
        )
    else:
        health_meta_statics.remote(server_url, verbose=False)


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


def prepare_http_sdk_request(
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
        httpclient.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        httpclient.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        httpclient.InferInput("reference_text", [1, 1], "BYTES"),
        httpclient.InferInput("target_text", [1, 1], "BYTES"),
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
