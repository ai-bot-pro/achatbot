#!/usr/bin/python

# https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.18.0/tools/whisper/client.py

import asyncio
from collections import defaultdict
import logging
import os
import queue
import sys
import time
import types
from typing import Dict, Iterable, List, TextIO, Tuple
import uuid
import modal


app_name = "whisper"
app = modal.App(f"{app_name}-grpc-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install(
    "numpy",
    "soundfile",
    "tritonclient[all]",
    "kaldialign",  # WER
)
image = image.run_commands(
    # god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0001.wav' -O /1221-135766-0001.wav",
    # yet these thoughts affected hester prynne less with hope than apprehension
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav' -O /1221-135766-0002.wav",
    # 大学生利用漏洞免费吃肯德基获刑
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_zh/wav/long.wav' -O /long.wav",
    # 富士康在印度工厂出现大规模感染目前工厂产量已下降超50%
    "wget 'https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_zh/wav/mid.wav' -O /mid.wav",
    "ls -lh /*.wav",
)
image = image.env({"TRITON_PROTOCOL": os.getenv("TRITON_PROTOCOL", "grpc")})

audio_text_map = {
    "/1221-135766-0001.wav": "god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven.",
    "/1221-135766-0002.wav": "yet these thoughts affected hester prynne less with hope than apprehension.",
    "/mid.wav": "大学生利用漏洞免费吃肯德基获刑。",
    "/long.wav": "富士康在印度工厂出现大规模感染，目前工厂产量已下降超50%。",
}

with image.imports():
    import numpy as np
    import soundfile as sf

    # import tritonclient.grpc as protocol_client
    protocol = os.getenv("TRITON_PROTOCOL", "grpc")
    print(f"use protocol: {protocol}")
    if protocol == "grpc":
        import tritonclient.grpc.aio as protocol_client
    else:
        import tritonclient.http.aio as protocol_client
    from tritonclient.grpc import InferenceServerException
    from tritonclient.utils import np_to_triton_dtype
    import kaldialign


ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    cpu=2.0,
    retries=0,
    image=image,
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def health(
    server_url: str,
    verbose: bool = False,
    model_names: str = "whisper_bls,whisper_tensorrt_llm",
):
    url = server_url

    try:
        triton_client = protocol_client.InferenceServerClient(url=url, verbose=verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        return

    # Health
    if not await triton_client.is_server_live(headers={"test": "1", "dummy": "2"}):
        print("FAILED : is_server_live")
        return
    if not await triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        return

    # Metadata
    metadata = await triton_client.get_server_metadata()
    if isinstance(metadata, dict):
        metadata = types.SimpleNamespace(**metadata)
    if not (metadata.name == "triton"):
        print("FAILED : get_server_metadata")
        return
    print(metadata)

    # Health
    for model_name in model_names.split(","):
        if not await triton_client.is_model_ready(model_name):
            print(f"{model_name} FAILED : is_model_ready")

        print("-" * 20)

        metadata = await triton_client.get_model_metadata(
            model_name, headers={"test": "1", "dummy": "2"}
        )
        if isinstance(metadata, dict):
            metadata = types.SimpleNamespace(**metadata)
        if not (metadata.name == model_name):
            print(f"{model_name} FAILED : get_model_metadata")
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
    server_url: str = "localhost:8000",
    reference_audio: str = "/1221-135766-0002.wav",
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    model_name: str = "whisper_bls",
    padding_duration: int = 10,
    streaming: bool = False,
    verbose: bool = False,
):
    triton_client = protocol_client.InferenceServerClient(url=server_url, verbose=verbose)
    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    inputs = prepare_grpc_sdk_request(
        samples, text_prefix, duration, sample_rate=sr, padding_duration=padding_duration
    )
    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]

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
    print(f"ref_text____________: {audio_text_map[reference_audio]}")
    print(f"asr_decoding_results: {decoding_results}")

    tot_err_rate = write_error_stats(
        None,
        test_set_name="test-asr",
        results=[
            (
                reference_audio,
                audio_text_map[reference_audio].lower(),
                decoding_results.strip().lower(),
            )
        ],
        enable_log=False,
    )
    print("total_err_rate:", tot_err_rate)

    await triton_client.close()


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
    reference_audio: str = "/1221-135766-0002.wav",
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    model_name: str = "whisper_bls",
    padding_duration: int = 10,  # padding to nearset 10 seconds
    concurency_cn: int = 1,
    batch_size: int = 1,
    verbose: bool = False,
    streaming: bool = False,
):
    dps_list = [
        [
            {
                "wav_id": reference_audio,
                "ref_audio_filepath": reference_audio,
                "ref_text": audio_text_map[reference_audio],
            }
        ]
        * batch_size
    ] * concurency_cn

    triton_client = protocol_client.InferenceServerClient(url=server_url, verbose=verbose)

    tasks = []
    start_time = time.time()
    for i in range(concurency_cn):
        task = asyncio.create_task(
            send_whisper(
                dps=dps_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=1,
                model_name=model_name,
                padding_duration=padding_duration,
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
        protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
        protocol_client.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype)),
        protocol_client.InferInput("WAV_LEN", lengths.shape, np_to_triton_dtype(lengths.dtype)),
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
    triton_client: "protocol_client.InferenceServerClient",
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
    triton_client: "protocol_client.InferenceServerClient",
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
    triton_client: "protocol_client.InferenceServerClient",
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

        waveform, sr = sf.read(dp["ref_audio_filepath"])
        assert sr == 16000, "sample rate hardcoded in server"
        duration = int(len(waveform) / sr)
        # duration = sf.info(dp["ref_audio_filepath"]).duration

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
                dp["wav_id"],
                dp["ref_text"].split(),  # label
                decoding_results.split(),  # prediction
            )
        )
        print(results[-1])

    return total_duration, results, latency_data


"""
# gRPC
## health check for whisper
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --model-name whisper \
    --model-names whisper \
    --server-url "r15.modal.host:34101"

## health check for whisper_bls,whisper_tensorrt_llm
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --server-url "r15.modal.host:34101"

## health check for whisper_infer_bls, whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --action health \
    --model-name whisper_infer_bls \
    --model-names whisper_infer_bls,whisper_tensorrt_llm_cpprunner \
    --server-url "r28.modal.host:38535"


## single wav test for whisper
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --model-name whisper \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --model-name whisper \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## single wav test for whisper_bls,whisper_tensorrt_llm
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r24.modal.host:44175"

## single wav test for whisper_infer_bls,whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --model-name whisper_infer_bls \
    --server-url "r28.modal.host:38535"
modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action asr \
    --model-name whisper_infer_bls \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper_infer_bls \
    --reference-audio /1221-135766-0001.wav \
    --text-prefix "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"
modal run src/llm/trtllm/whisper/client.py \
    --action asr \
    --model-name whisper_infer_bls \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## bench for whisper
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --model-name whisper \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

## bench (concurency_cn:1->2->4->8->16 | batch_size(requests_cn):1->2->4->8)
## bench throughput and latency, grpc just test, modal support http, grpc now use tunnel
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "r28.modal.host:33695"

modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "r18.modal.host:41787"

modal run src/llm/trtllm/whisper/client.py \
    --streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r18.modal.host:41787"

## bench for whisper_infer_bls,whisper_tensorrt_llm_cpprunner
modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --model-name whisper_infer_bls \
    --concurency-cn 4 \
    --batch-size 4 \
    --reference-audio /long.wav \
    --text-prefix "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --server-url "r15.modal.host:34101"

# http
## health check
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --action health --verbose \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

## single wav test
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action asr \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

## bench (concurency_cn:1->2->4->8->16 | batch_size(requests_cn):1->2->4->8)
## bench throughput and latency, grpc just test, modal support http, grpc now use tunnel
TRITON_PROTOCOL=http modal run src/llm/trtllm/whisper/client.py \
    --no-streaming \
    --action bench_asr \
    --concurency-cn 4 \
    --batch-size 4 \
    --server-url "weedge--tritonserver-serve-dev.modal.run"

# other the same as grpc :-)

# WER eval
see run.py to change
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/1221-135766-0002.wav",
    text_prefix: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    model_name: str = "whisper_bls",
    action: str = "health",
    concurency_cn: int = 1,
    batch_size: int = 1,
    padding_duration: int = 10,  # padding to nearset 10 seconds , max 30 seconds
    verbose: bool = False,
    streaming: bool = False,
    model_names: str = "whisper_bls,whisper_tensorrt_llm",
):
    if action == "asr":
        asr.remote(
            server_url,
            reference_audio,
            text_prefix,
            model_name,
            padding_duration,
            verbose=verbose,
            streaming=streaming,
        )
    elif action == "bench_asr":
        """
        bench throughput and latency, grpc just test, modal support http, grpc now use tunnel
        """
        bench_asr.remote(
            server_url,
            reference_audio,
            text_prefix,
            model_name,
            padding_duration,
            concurency_cn,
            batch_size,
            verbose=verbose,
            streaming=streaming,
        )
    else:
        health.remote(server_url, verbose=verbose, model_names=model_names)


def store_transcripts(filename: os.PathLike, texts: Iterable[Tuple[str, str, str]]) -> None:
    """Save predicted results and reference transcripts to a file.
    https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str, str]],  # wav_id, label, prediction
    enable_log: bool = True,
) -> float:
    """Write statistics based on predicted results and reference transcripts.
    https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted([(sum(v[1:]), k, v) for k, v in words.items()], reverse=True):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)
