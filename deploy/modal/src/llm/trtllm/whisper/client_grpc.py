#!/usr/bin/python
"""
This script supports to load audio files and sends it to the server
for decoding, in parallel.


Usage:
# For offlien whisper server
python3 client.py \
    --server-addr localhost \
    --model-name whisper_bls \
    --num-tasks $num_task \
    --whisper-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --audio-path $audio_path
"""

# https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.18.0/tools/whisper/client.py

import asyncio
import os
import queue
import sys
import time
import types
import uuid
import modal
import tritonclient
from tritonclient.grpc import InferenceServerException
from tritonclient.utils import np_to_triton_dtype


app_name = "whisper"
app = modal.App(f"{app_name}-grpc-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install("numpy", "soundfile", "tritonclient[grpc]")
image = image.run_commands(
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0001.wav' -O /1221-135766-0001.wav",
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav' -O /1221-135766-0002.wav",
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_zh/wav/long.wav' -O long.wav",
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_zh/wav/mid.wav' -O mid.wav",
    "ls -lh /*.wav",
)
image = image.env({})

with image.imports():
    import numpy as np
    import soundfile as sf

    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype


ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


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
        "whisper_bls",
    ]:
        if not triton_client.is_model_ready(model_name):
            print("FAILED : is_model_ready")

        print("-" * 20)

        metadata = triton_client.get_model_metadata(model_name, headers={"test": "1", "dummy": "2"})
        if not (metadata.name == model_name):
            print("FAILED : get_model_metadata")
            return
        print(metadata)


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    volumes={
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def asr(
    reference_audio: str = "/prompt_audio.wav",
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    server_url: str = "localhost:8000",
    model_name: str = "spark_tts",
    padding_duration: int = 10,
    streaming: bool = False,
    verbose: bool = False,
):
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=verbose)
    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    inputs = prepare_grpc_sdk_request(
        samples, text_prefix, duration, sample_rate=sr, padding_duration=padding_duration
    )
    outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

    if streaming:
        decoding_results = await infer_streaming(
            model_name,
            inputs,
            request_id=str(uuid.uuid4()),
            outputs=outputs,
            triton_client=triton_client,
        )
    else:
        decoding_results = await infer(
            model_name,
            inputs,
            request_id=str(uuid.uuid4()),
            outputs=outputs,
            triton_client=triton_client,
        )
    print(f"decoding_results: {decoding_results}")

    triton_client.close()


@app.function(
    cpu=8.0,
    retries=0,
    image=image,
    volumes={
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def bench_asr(
    server_url: str = "localhost:8000",
    model_name: str = "spark_tts",
    concurency_cn: int = 1,
    verbose: bool = False,
    streaming: bool = False,
):
    text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    reference_audio: str = ("/prompt_audio.wav",)
    text_prefix: str = (
        "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    )
    dps_list = [
        [
            {
                "audio_filepath": reference_audio,
                "text": "foo",
                "id": 0,
            }
        ]
    ] * concurency_cn

    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=verbose)

    tasks = []
    start_time = time.time()
    for i in range(concurency_cn):
        task = asyncio.create_task(
            send_whisper(
                dps=dps_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                log_interval=1,
                model_name=model_name,
                text_prefix=text_prefix,
                streaming=streaming,
            )
        )
        tasks.append(task)

    answer_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for answer in answer_list:
        total_duration += answer[0]
        results += answer[1]
        latency_data += answer[2]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"

    latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    s += f"latency_variance: {latency_variance:.2f}\n"
    s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
    s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
    s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
    s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
    s += f"average_latency_ms: {latency_ms:.2f}\n"

    print(s)


def prepare_grpc_sdk_request(
    waveform,
    text_prefix,
    duration: float,
    sample_rate=16000,
    padding_duration: int = None,
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    if padding_duration:
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
                "name": "TEXT_PREFIX",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [TEXT_PREFIX],
            },
            {
                "name": "WAV",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist(),
            },
            {
                "name": "WAV_LEN",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            }
        ]
    }
    """
    inputs = [
        grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
        grpcclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype)),
        grpcclient.InferInput("WAV_LEN", lengths.shape, np_to_triton_dtype(lengths.dtype)),
    ]
    input_data_numpy = np.array([text_prefix], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[0].set_data_from_numpy(input_data_numpy)
    inputs[1].set_data_from_numpy(samples)
    inputs[2].set_data_from_numpy(lengths)

    return inputs


async def infer(
    model_name,
    inputs,
    request_id,
    outputs,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
):
    response = await triton_client.infer(model_name, inputs, request_id=request_id, outputs=outputs)
    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if isinstance(decoding_results, np.ndarray):
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        decoding_results = decoding_results.decode("utf-8")
    return decoding_results


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


async def infer_streaming(
    model_name,
    inputs,
    request_id,
    outputs,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
):
    user_data = UserData()

    async def async_request_iterator():
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": request_id,
        }

    try:
        response_iterator = triton_client.stream_infer(
            inputs_iterator=async_request_iterator(),
            stream_timeout=None,
        )
        async for response in response_iterator:
            result, error = response
            if error:
                print(error)
                user_data._completed_requests.put(error)
            else:
                user_data._completed_requests.put(result)
    except InferenceServerException as error:
        print(error)
        sys.exit(1)

    results = []
    while True:
        try:
            data_item = user_data._completed_requests.get(block=False)
            if isinstance(data_item, InferenceServerException):
                sys.exit(1)
            else:
                decoding_results = data_item.as_numpy("TRANSCRIPTS")[0]
                if isinstance(decoding_results, np.ndarray):
                    decoding_results = b" ".join(decoding_results).decode("utf-8")
                else:
                    decoding_results = decoding_results.decode("utf-8")
                results.append(decoding_results)
        except Exception:
            break
    decoding_results = results[-1]
    return decoding_results


async def send_whisper(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    model_name: str,
    padding_duration: int = 10,  # padding to nearset 10 seconds
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    streaming: bool = False,
):
    total_duration = 0.0
    results = []
    latency_data = []
    task_id = int(name[5:])
    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sr = sf.read(dp["audio_filepath"])
        assert sr == 16000, "sample rate hardcoded in server"
        duration = int(len(waveform) / sr)
        # duration = sf.info(dp["audio_filepath"]).duration

        inputs = prepare_grpc_sdk_request(
            waveform, text_prefix, duration, sample_rate=sr, padding_duration=padding_duration
        )
        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]

        # Send request
        sequence_id = 100000000 + i + task_id * 100
        start = time.time()
        if streaming:
            decoding_results = await infer_streaming(
                model_name,
                inputs,
                request_id=str(sequence_id),
                outputs=outputs,
                triton_client=triton_client,
            )
        else:
            decoding_results = await infer(
                model_name,
                inputs,
                request_id=str(sequence_id),
                outputs=outputs,
                triton_client=triton_client,
            )
        end = time.time() - start
        latency_data.append((end, duration))
        total_duration += duration
        results.append(
            (
                dp["id"],
                dp["text"].split(),
                decoding_results.split(),
            )
        )
        print(results[-1])

    return total_duration, results, latency_data


"""
# health check
modal run src/llm/trtllm/whisper/client_grpc.py \
    --action health \
    --server-url "r15.modal.host:44161"

# single wav test
modal run src/llm/trtllm/whisper/client_grpc.py \
    --action asr \
    --server-url "r15.modal.host:44161"

# bench
modal run src/llm/trtllm/whisper/client_grpc.py \
    --action bench_asr \
    --server-url "r15.modal.host:44161"

# WER eval
see run.py
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    model_name: str = "whisper_bls",
    action: str = "health",
    concurency_cn: int = 1,
    verbose: bool = False,
    streaming: bool = False,
):
    if action == "asr":
        asr.remote(
            server_url,
            reference_audio,
            text_prefix,
            model_name,
            concurency_cn,
            verbose=verbose,
            streaming=streaming,
        )
    elif action == "bench_asr":
        bench_asr.remote(
            server_url,
            model_name,
            concurency_cn,
            verbose=verbose,
            streaming=streaming,
        )
    else:
        health.remote(server_url, verbose=verbose)
