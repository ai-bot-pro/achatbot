import os
import time
import traceback
import wave

import numpy as np
from pyrnnoise import RNNoise
import typer

from src.common.utils.audio_resample import (
    resample_bytes2bytes,
    resample_file,
    resample_bytes2numpy,
)

app = typer.Typer()


@app.command()
def random_np_test():
    denoiser = RNNoise(sample_rate=48000)

    try:
        i = 0
        for i in range(10000):
            audio_data = np.random.randint(-32768, 32767, (1, 512), dtype=np.int16)
            is_last = i == 1000
            for speech_prob, denoised_audio in denoiser.denoise_chunk(audio_data, is_last):
                print(f"{denoised_audio.shape}")
                print(f"Speech probability: {speech_prob}")
                # Process denoised_audio as needed
            if is_last is True:
                denoiser.reset()
    except Exception as e:
        print(i, e, traceback.format_exc())


@app.command()
def file_test_48k():
    denoiser = RNNoise(sample_rate=48000)
    in_path = "./records/speech_with_noise48k.wav"
    out_path = "./records/output_rnnoise_enhancer_stream_48k.wav"
    for speech_prob in denoiser.denoise_wav(in_path, out_path):
        print(speech_prob)
    print(f"{in_path=} save to {out_path}")


@app.command()
def file_test_16k():
    """
    NOTE:
    - 16K to 48K end save 16k denoise not good
    - need 16k to 48k resample first
    """
    denoiser = RNNoise(sample_rate=48000)
    raw_path = "./records/speech_with_noise16k.wav"
    in_path = "./records/speech_with_noise16k_48k.wav"
    resample_file(raw_path, in_path)
    out_path = "./records/output_rnnoise_enhancer_stream16k_48k.wav"
    for speech_prob in denoiser.denoise_wav(in_path, out_path):
        print(speech_prob)
    print(f"{in_path=} save to {out_path}")


@app.command()
def online_stream():
    denoiser = RNNoise(sample_rate=48000)

    # 模型每次处理的音频片段不能小于block_size
    # NOTE:
    # - rnnoise min support 10ms for 48000 sample rate
    # - if not 10ms frame size, AudioGraph to buffering until 10ms frame size
    # SEE:
    # - https://av.basswood-io.com/docs/stable/api/filter.html#bv.filter.graph.Graph
    # https://ffmpeg.org/ffmpeg-filters.html#abuffer

    # in_audio_path = "./records/speech_with_noise_48k.wav"
    # block_bytes_size = 480 * 2  # 10ms
    # block_bytes_size = 48 * 2  # 1ms

    in_audio_path = "./records/speech_with_noise16k.wav"
    # block_bytes_size = 160 * 2
    # block_bytes_size = 16 * 2
    block_bytes_size = 512 * 2  # silero vad frame size * 2

    start = time.time()
    waveform_48k_int16 = resample_bytes2numpy(
        b" " * block_bytes_size,
        orig_freq=16000,
        new_freq=48000,
    )
    denoiser.denoise_chunk(waveform_48k_int16, False)
    print(f"warmup cost: {time.time() - start} s")

    for i in range(10):
        denoiser.reset()
        audio_pcm = b""
        begin = time.time()
        with open(os.path.join(os.getcwd(), in_audio_path), "rb") as f:
            # 跳过 WAV 头
            f.read(44)

            audio_bytes = f.read(block_bytes_size)
            while len(audio_bytes) > 0:
                start = time.time()
                print("before", len(audio_bytes))
                waveform_48k_int16 = resample_bytes2numpy(
                    audio_bytes,
                    orig_freq=16000,
                    new_freq=48000,
                )
                pcm = b""
                for speech_prob, frame in denoiser.denoise_chunk(
                    waveform_48k_int16, len(audio_bytes) != block_bytes_size
                ):
                    # print(speech_prob)
                    pcm_48k = frame.tobytes()
                    pcm += pcm_48k
                pcm = resample_bytes2bytes(
                    pcm,
                    orig_freq=48000,
                    new_freq=16000,
                )
                print("after", len(pcm))
                audio_pcm += pcm
                print(f"cost: {time.time() - start} s")
                audio_bytes = f.read(block_bytes_size)

    print(f"total cost: {time.time() - begin} s")
    out_path = "./records/output_rnnoise_enhancer_stream.wav"
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_pcm)
        print(f"{in_audio_path=} save to {out_path}")


@app.command()
def online_stream_48k():
    denoiser = RNNoise(sample_rate=48000)

    # 模型每次处理的音频片段不能小于block_size
    # NOTE:
    # - rnnoise min support 10ms for 48000 sample rate
    # - if not 10ms frame size, AudioGraph to buffering until 10ms frame size
    # SEE:
    # - https://av.basswood-io.com/docs/stable/api/filter.html#bv.filter.graph.Graph
    # https://ffmpeg.org/ffmpeg-filters.html#abuffer

    in_audio_path = "./records/speech_with_noise_48k.wav"
    block_bytes_size = 480 * 2  # 10ms
    # block_bytes_size = 48 * 2  # 1ms

    start = time.time()
    denoiser.denoise_chunk(b" " * block_bytes_size, False)
    print(f"warmup cost: {time.time() - start} s")

    for i in range(10):
        denoiser.reset()
        try:
            audio_pcm = b""
            begin = time.time()
            with open(os.path.join(os.getcwd(), in_audio_path), "rb") as f:
                # 跳过 WAV 头
                f.read(44)

                audio_bytes = f.read(block_bytes_size)
                while len(audio_bytes) > 0:
                    start = time.time()
                    print("before", len(audio_bytes))
                    pcm = b""
                    # 1. 转换为 int16 数组
                    waveform_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    # 2. 转换为归一化浮点数
                    # waveform_float = waveform_int16.astype(np.float32) / 32768.0
                    for speech_prob, frame in denoiser.denoise_chunk(
                        waveform_int16, len(audio_bytes) != block_bytes_size
                    ):
                        # print(speech_prob)
                        pcm_48k = frame.tobytes()
                        pcm += pcm_48k
                    print("after", len(pcm))
                    audio_pcm += pcm
                    print(f"cost: {time.time() - start} s")
                    audio_bytes = f.read(block_bytes_size)
        except Exception as e:
            print(i, e, traceback.format_exc())
            break

    print(f"total cost: {time.time() - begin} s")

    if len(audio_pcm) > 0:
        out_path = "./records/output_rnnoise_enhancer_stream_48k.wav"
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_pcm)
            print(f"{in_audio_path=} save to {out_path}")


"""
python -m demo.ans_rnnoise_enhancer random-np-test
python -m demo.ans_rnnoise_enhancer file-test-16k
python -m demo.ans_rnnoise_enhancer file-test-48k
python -m demo.ans_rnnoise_enhancer online-stream
python -m demo.ans_rnnoise_enhancer online-stream-48k
"""
if __name__ == "__main__":
    app()
