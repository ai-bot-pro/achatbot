import time
import wave

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.fileio import File
from modelscope.outputs import OutputKeys
import typer
import numpy as np

from src.common.utils.audio_resample import (
    resample_bytes2bytes,
    resample_file,
    resample_bytes2numpy,
)
from src.modules.speech.help.audio_mock import generate_white_noise

app = typer.Typer()


"""
- https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal
"""


def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b"RIFF")
    header.extend(riff_chunk_size.to_bytes(4, byteorder="little"))
    header.extend(b"WAVE")

    # fmt subchunk
    header.extend(b"fmt ")
    header.extend(fmt_chunk_size.to_bytes(4, byteorder="little"))
    header.extend((1).to_bytes(2, byteorder="little"))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder="little"))
    header.extend(sample_rate.to_bytes(4, byteorder="little"))
    header.extend(byte_rate.to_bytes(4, byteorder="little"))
    header.extend(block_align.to_bytes(2, byteorder="little"))
    header.extend(bits_per_sample.to_bytes(2, byteorder="little"))

    # data subchunk
    header.extend(b"data")
    header.extend(data_chunk_size.to_bytes(4, byteorder="little"))

    return bytes(header) + dataflow


@app.command()
def online_stream():
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model="iic/speech_dfsmn_ans_psm_48k_causal",  # output 48K auido sample rate
        stream_mode=True,
    )
    block_bytes_size = 160 * 2  # 模型每次处理的音频片段不能小于block_size

    start = time.time()
    white_noise = generate_white_noise(duration=0.1, sr=16000)
    audio_data = white_noise.astype(np.int16).tobytes()
    audio_48k_bytes = resample_bytes2bytes(
        audio_data,
        orig_freq=16000,
        new_freq=48000,
    )
    result = ans(audio_48k_bytes)
    out_pcm = result[OutputKeys.OUTPUT_PCM]
    print(f"warmup {len(out_pcm)=} cost: {time.time() - start} s")

    # audio_path = "./records/speech_with_noise_48k.wav"

    # input_path = "./records/speech_with_noise16k.wav"
    # audio_path = "./records/speech_with_noise48k.wav"
    # resample_file(input_path, audio_path)

    audio_path = "./records/speech_with_noise16k.wav"

    audio_pcm = b""
    begin = time.time()
    with open(os.path.join(os.getcwd(), audio_path), "rb") as f:
        # audio = f.read(block_bytes_size)
        audio = f.read(block_bytes_size + 44)[44:]  # remove RIFF header
        while len(audio) > 0:
            start = time.time()
            print("before", len(audio))
            audio_48k = resample_bytes2bytes(
                audio,
                orig_freq=16000,
                new_freq=48000,
            )
            print("resample after", len(audio_48k))
            result = ans(audio_48k)
            out_pcm = result[OutputKeys.OUTPUT_PCM]
            print("ans after", len(out_pcm))
            if len(out_pcm) > 0:
                out_pcm = resample_bytes2bytes(
                    out_pcm,
                    orig_freq=48000,
                    new_freq=16000,
                )
                print("resample back", len(out_pcm))
                audio_pcm += out_pcm
            print(f"cost: {time.time() - start} s")
            audio = f.read(block_bytes_size)

    print(f"total cost: {time.time() - begin} s")
    out_path = "./records/output_dfsmn_enhancer_stream.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_pcm)
        print(f"{audio_path=} save to {out_path}")


@app.command()
def offline_stream():
    ans = pipeline(Tasks.acoustic_noise_suppression, model="iic/speech_dfsmn_ans_psm_48k_causal")

    # audio_path = 'https://modelscope.cn/api/v1/models/damo/speech_dfsmn_ans_psm_48k_causal/repo?Revision=master&FilePath=examples/speech_with_noise_48k.wav',
    audio_path = "./records/speech_with_noise_48k.wav"

    if audio_path.startswith("http"):
        import io

        file_bytes = File.read(audio_path)
        audiostream = io.BytesIO(file_bytes)
    else:
        audiostream = open(audio_path, "rb")

    window = 1 * 48000 * 2  # 2 秒的窗口大小，以字节为单位
    outputs = b""
    total_bytes_len = 0
    audiostream.read(44)
    for dataflow in iter(lambda: audiostream.read(window), ""):
        print(len(dataflow))
        total_bytes_len += len(dataflow)
        if len(dataflow) == 0:
            break
        result = ans(
            create_wav_header(dataflow, sample_rate=48000, num_channels=1, bits_per_sample=16)
        )
        output = result["output_pcm"]
        outputs = outputs + output
    audiostream.close()

    outputs = outputs[:total_bytes_len]
    output_path = "dfsmn_enhancer_output_stream.wav"
    with open(output_path, "wb") as out_wave:
        out_wave.write(
            create_wav_header(outputs, sample_rate=48000, num_channels=1, bits_per_sample=16)
        )


@app.command()
def offline():
    ans = pipeline(Tasks.acoustic_noise_suppression, model="iic/speech_dfsmn_ans_psm_48k_causal")

    audio_path = "./records/speech_with_noise16k.wav"
    # audio_path = "./records/speech_with_noise_48k.wav"

    result = ans(
        # 'https://modelscope.cn/api/v1/models/damo/speech_dfsmn_ans_psm_48k_causal/repo?Revision=master&FilePath=examples/speech_with_noise_48k.wav',
        audio_path,
        output_path="dfsmn_enhancer_output.wav",
    )
    print(len(result["output_pcm"]))


"""
python -m demo.modelscope.ans_dfsmn_enhancer offline
python -m demo.modelscope.ans_dfsmn_enhancer offline-stream
python -m demo.modelscope.ans_dfsmn_enhancer online-stream
"""
if __name__ == "__main__":
    import torch
    import os

    thread_count = os.cpu_count()
    print(f"thread_count: {thread_count}")

    torch.set_num_threads(min(8, thread_count))
    torch.set_num_interop_threads(min(8, thread_count))

    app()
