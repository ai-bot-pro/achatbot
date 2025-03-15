# https://github.com/triton-inference-server/client/tree/main/src/python/examples

import os
import modal

app_name = "tts-spark"
app = modal.App(f"{app_name}-http-client")

# Define the dependencies for the function using a Modal Image.
image = modal.Image.debian_slim(python_version="3.10").apt_install("git", "wget")
image = image.pip_install(
    "numpy",
    "librosa",
    "soundfile",
    "requests",
    "tritonclient[all]",
    "datasets",
)
image = image.run_commands(
    "wget 'https://raw.githubusercontent.com/SparkAudio/Spark-TTS/refs/heads/main/example/prompt_audio.wav' -O /prompt_audio.wav",
    "ls -lh /prompt_audio.wav",
)
image = image.env(
    {
        "CLI_MODE": os.getenv("CLI_MODE", "http"),
        # https://huggingface.co/docs/huggingface_hub/quick-start#login
        # NOTE: need set in secret env, just for test, priority use secret env :) don't do this~
        # "HF_TOKEN": os.getenv("HF_TOKEN", "set your hf token here"),
    }
)

with image.imports():
    import os
    import time
    import types  # type: ignore
    from pathlib import Path
    import asyncio
    import json

    import requests
    import numpy as np
    import soundfile as sf
    import tritonclient
    from tritonclient.utils import np_to_triton_dtype
    from tritonclient.utils import InferenceServerException

    cli_mode = os.getenv("CLI_MODE", "http")
    print(f"cli_mode: {cli_mode}")
    import tritonclient.http.aio as client

    if cli_mode == "grpc":
        import tritonclient.grpc.aio as client

HF_CACHE_DIR = "/root/.cache/huggingface"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

TTS_GEN_AUDIO_DIR = "/tts_gen_audio"
tts_gen_audio_vol = modal.Volume.from_name("tts_gen_audio", create_if_missing=True)

BENCH_LOG_DIR = "/bench"
bench_log_vol = modal.Volume.from_name("bench", create_if_missing=True)


@app.function(
    cpu=8.0,  # for more concurrecy, need add more cpus
    retries=0,
    image=image,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        BENCH_LOG_DIR: bench_log_vol,
        HF_CACHE_DIR: hf_cache_vol,
        TTS_GEN_AUDIO_DIR: tts_gen_audio_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
)
async def bench(
    server_url: str = "localhost:8000",
    reference_audio: str = "",
    reference_text: str = "",
    target_text: str = "",
    model_name: str = "spark_tts",
    huggingface_dataset: str = "yuekai/seed_tts",  # https://huggingface.co/datasets/yuekai/seed_tts
    split_name: str = "wenetspeech4tts",  # ["wenetspeech4tts", "test_zh", "test_en", "test_hard"]
    manifest_path: str = "",
    num_tasks: int = 1,
    log_interval: int = 5,
    compute_wer: bool = False,
    batch_size: int = 1,
):
    mode = os.getenv("CLI_MODE", "http")
    triton_client = client.InferenceServerClient(url=server_url, verbose=False)

    if reference_audio:
        num_tasks = 1
        log_interval = 1
        manifest_item_list = [
            {
                "reference_text": reference_text,
                "target_text": target_text,
                "audio_filepath": reference_audio,
                "target_audio_path": "test",
            }
        ]
    elif huggingface_dataset:
        import datasets

        dataset = datasets.load_dataset(
            huggingface_dataset,
            split=split_name,
            trust_remote_code=True,
        )
        manifest_item_list = []
        for i in range(len(dataset)):
            manifest_item_list.append(
                {
                    "audio_filepath": dataset[i]["prompt_audio"],
                    "reference_text": dataset[i]["prompt_text"],
                    "target_audio_path": dataset[i]["id"],
                    "target_text": dataset[i]["target_text"],
                }
            )
    else:
        manifest_item_list = load_manifests(manifest_path)

    num_tasks = min(num_tasks, len(manifest_item_list))
    manifest_item_list = split_data(manifest_item_list, num_tasks)

    log_dir = os.path.join(BENCH_LOG_DIR, app_name)
    os.makedirs(log_dir, exist_ok=True)
    audio_save_dir = os.path.join(TTS_GEN_AUDIO_DIR, app_name)
    os.makedirs(audio_save_dir, exist_ok=True)

    tasks = []
    start_time = time.time()
    for i in range(num_tasks):
        task = asyncio.create_task(
            send(
                manifest_item_list[i],
                f"task-{i}",
                triton_client,
                log_interval,
                model_name,
                audio_save_dir=audio_save_dir,
                padding_duration=None,
                mode=mode,
                server_url=server_url,
            )
        )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        latency_data += ans[1]

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
    if manifest_path:
        name = Path(manifest_path).stem
    elif split_name:
        name = split_name
    with open(f"{log_dir}/rtf-{name}.txt", "w") as f:
        f.write(s)

    if mode == "grpc":
        stats = await triton_client.get_inference_statistics(model_name="", as_json=True)
    if "http" in mode:
        stats = await triton_client.get_inference_statistics(model_name="")
    write_triton_stats(stats, f"{log_dir}/stats_summary-{name}.txt")

    if mode == "grpc":
        metadata = await triton_client.get_model_config(model_name=model_name, as_json=True)
    if "http" in mode:
        metadata = await triton_client.get_model_config(model_name=model_name)
    with open(f"{log_dir}/model_config-{name}.json", "w") as f:
        json.dump(metadata, f, indent=4)

    await triton_client.close()


"""
# single test
CLI_MODE=http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "/prompt_audio.wav" \
    --huggingface-dataset ""

# hf dataset with (wenetspeech4tts 26 rows) with concurrency 1 task to bench
CLI_MODE=http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 1 

# single test used by http client sdk
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "/prompt_audio.wav" \
    --huggingface-dataset ""

# hf dataset with (wenetspeech4tts 26 rows) with concurrency 1 task to bench used by http client sdk
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 1 

# hf dataset with (wenetspeech4tts 26 rows) concurrency min(num_tasks,26) task to bench used by http client sdk
# num-tasks: 2->4->8->16->26
CLI_MODE=cli_sdk_http modal run src/llm/trtllm/tts_spark/bench.py \
    --server-url "weedge--tritonserver-serve-dev.modal.run" \
    --reference-audio "" \
    --huggingface-dataset "yuekai/seed_tts" \
    --split-name wenetspeech4tts \
    --num-tasks 2
"""


@app.local_entrypoint()
def main(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    huggingface_dataset: str = "yuekai/seed_tts",
    split_name: str = "wenetspeech4tts",  # ["wenetspeech4tts", "test_zh", "test_en", "test_hard"]
    manifest_path: str = None,  # unuse
    num_tasks: str = "1",
    log_interval: str = "5",
    compute_wer: str = "",
    batch_size: str = "1",
):
    bench.remote(
        server_url,
        reference_audio,
        reference_text,
        target_text,
        model_name,
        huggingface_dataset,
        split_name,
        manifest_path,
        int(num_tasks),
        int(log_interval),
        bool(compute_wer),
        int(batch_size),
    )


def write_triton_stats(stats, summary_file):
    with open(summary_file, "w") as summary_f:
        model_stats = stats["model_stats"]
        # write a note, the log is from triton_client.get_inference_statistics(), to better human readability
        summary_f.write(
            "The log is parsing from triton_client.get_inference_statistics(), to better human readability. \n"
        )
        summary_f.write("To learn more about the log, please refer to: \n")
        summary_f.write(
            "1. https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md \n"
        )
        summary_f.write("2. https://github.com/triton-inference-server/server/issues/5374 \n\n")
        summary_f.write(
            "To better improve throughput, we always would like let requests wait in the queue for a while, and then execute them with a larger batch size. \n"
        )
        summary_f.write(
            "However, there is a trade-off between the increased queue time and the increased batch size. \n"
        )
        summary_f.write(
            "You may change 'max_queue_delay_microseconds' and 'preferred_batch_size' in the model configuration file to achieve this. \n"
        )
        summary_f.write(
            "See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching for more details. \n\n"
        )
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = int(model_inference_stats["queue"]["ns"]) / 1e9
            total_infer_time_s = int(model_inference_stats["compute_infer"]["ns"]) / 1e9
            total_input_time_s = int(model_inference_stats["compute_input"]["ns"]) / 1e9
            total_output_time_s = int(model_inference_stats["compute_output"]["ns"]) / 1e9
            summary_f.write(
                f"queue time {total_queue_time_s:<5.2f} s, compute infer time {total_infer_time_s:<5.2f} s, compute input time {total_input_time_s:<5.2f} s, compute output time {total_output_time_s:<5.2f} s \n"  # noqa
            )
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert compute_infer["count"] == compute_output["count"] == compute_input["count"]
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"execuate inference with batch_size {batch_size:<2} total {batch_count:<5} times, total_infer_time {compute_infer_time_ms:<9.2f} ms, avg_infer_time {compute_infer_time_ms:<9.2f}/{batch_count:<5}={compute_infer_time_ms/batch_count:.2f} ms, avg_infer_time_per_sample {compute_infer_time_ms:<9.2f}/{batch_count:<5}/{batch_size}={compute_infer_time_ms/batch_count/batch_size:.2f} ms \n"  # noqa
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms/batch_count:.2f} ms, "  # noqa
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms/batch_count:.2f} ms \n"  # noqa
                )


def load_audio(wav_path, target_sample_rate=16000):
    assert target_sample_rate == 16000, "hard coding in server"
    if isinstance(wav_path, dict):
        waveform = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate


async def send(
    manifest_item_list: list,
    name: str,
    triton_client: object,
    log_interval: int,
    model_name: str,
    audio_save_dir: str = "./",
    padding_duration: int = None,
    mode="http",
    server_url: str = "localhost:8000",
):
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])

    print(f"manifest_item_list({len(manifest_item_list)}): {manifest_item_list}")
    for i, item in enumerate(manifest_item_list):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(manifest_item_list)}")
        sequence_id = 100000000 + i + task_id * 10
        if mode in ["grpc", "cli_sdk_http"]:
            (audio, estimated_target_duration, end) = await cli_sdk_tts(
                sequence_id,
                triton_client,
                reference_audio=item["audio_filepath"],
                reference_text=item["reference_text"],
                target_text=item["target_text"],
                model_name=model_name,
                padding_duration=padding_duration,
            )
        else:
            (audio, estimated_target_duration, end) = http_tts(
                sequence_id,
                server_url=server_url,
                reference_audio=item["audio_filepath"],
                reference_text=item["reference_text"],
                target_text=item["target_text"],
                model_name=model_name,
                padding_duration=padding_duration,
            )

        latency_data.append((end, estimated_target_duration))
        total_duration += estimated_target_duration

        audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")
        sf.write(audio_save_path, audio, 16000, "PCM_16")

    return total_duration, latency_data


def load_manifests(manifest_path):
    with open(manifest_path, "r") as f:
        manifest_list = []
        for line in f:
            assert len(line.strip().split("|")) == 4
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            utt = Path(utt).stem
            # gt_wav = os.path.join(os.path.dirname(manifest_path), "wavs", utt + ".wav")
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(manifest_path), prompt_wav)
            manifest_list.append(
                {
                    "audio_filepath": prompt_wav,
                    "reference_text": prompt_text,
                    "target_text": gt_text,
                    "target_audio_path": utt,
                }
            )
    return manifest_list


def split_data(data, k):
    n = len(data)
    if n < k:
        print(
            f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}."
        )
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


def prepare_http_request(
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


def http_tts(
    sequence_id: int,
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    padding_duration: int = None,
):
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{model_name}/infer"
    waveform, sr = load_audio(reference_audio, target_sample_rate=16000)
    duration = len(waveform) / sr

    assert sr == 16000, "sample rate hardcoded in server"

    estimated_target_duration = duration / len(reference_text) * len(target_text)

    samples = np.array(waveform, dtype=np.float32)
    data = prepare_http_request(
        samples,
        reference_text,
        target_text,
        duration,
        sample_rate=sr,
        padding_duration=padding_duration,
    )

    start = time.time()
    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": str(sequence_id)},
    )
    end = time.time() - start
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)

    return (audio, estimated_target_duration, end)


async def cli_sdk_tts(
    sequence_id: int,
    triton_client: object,
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts",
    padding_duration: int = None,
):
    waveform, sample_rate = load_audio(reference_audio, target_sample_rate=16000)
    duration = len(waveform) / sample_rate
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    estimated_target_duration = duration / len(reference_text) * len(target_text)

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

    inputs = [
        client.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        client.InferInput("reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)),
        client.InferInput("reference_text", [1, 1], "BYTES"),
        client.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)

    input_data_numpy = np.array([reference_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)

    input_data_numpy = np.array([target_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[3].set_data_from_numpy(input_data_numpy)

    outputs = [client.InferRequestedOutput("waveform")]

    start = time.time()
    response = await triton_client.infer(
        model_name, inputs, request_id=str(sequence_id), outputs=outputs
    )
    end = time.time() - start

    audio = response.as_numpy("waveform").reshape(-1)

    return (audio, estimated_target_duration, end)
