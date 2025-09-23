import os
import time
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
    # Create denoiser instance
    denoiser = RNNoise(sample_rate=48000)

    # Generate or load some audio data (stereo in this example)
    audio_data = np.random.randint(-32768, 32767, (2, 48000), dtype=np.int16)

    # Process audio chunk
    for speech_prob, denoised_audio in denoiser.denoise_chunk(audio_data):
        print(f"{denoised_audio.shape}")
        print(f"Speech probability: {speech_prob}")
        # Process denoised_audio as needed


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
    block_bytes_size = 160 * 2
    # block_bytes_size = 16 * 2

    start = time.time()
    waveform_48k_int16 = resample_bytes2numpy(
        b" " * block_bytes_size,
        orig_freq=16000,
        new_freq=48000,
    )
    denoiser.denoise_chunk(waveform_48k_int16, False)
    print(f"warmup cost: {time.time() - start} s")

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


"""
python -m demo.ans_rnnoise_enhancer file-test-16k
python -m demo.ans_rnnoise_enhancer file-test-48k
python -m demo.ans_rnnoise_enhancer online-stream
"""
if __name__ == "__main__":
    app()
