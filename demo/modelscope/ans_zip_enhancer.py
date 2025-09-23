import time
import wave

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.fileio import File
from modelscope.outputs import OutputKeys
import typer

app = typer.Typer()


"""
https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/summary
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
def offline_stream():
    ans = pipeline(
        Tasks.acoustic_noise_suppression, model="iic/speech_zipenhancer_ans_multiloss_16k_base"
    )

    # audio_path = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise16k.wav"
    audio_path = "./records/speech_with_noise16k.wav"

    if audio_path.startswith("http"):
        import io

        file_bytes = File.read(audio_path)
        audiostream = io.BytesIO(file_bytes)
    else:
        audiostream = open(audio_path, "rb")

    window = 2 * 16000 * 2  #  2s的窗口大小，以字节为单位
    outputs = b""
    total_bytes_len = 0
    audiostream.read(44)
    for dataflow in iter(lambda: audiostream.read(window), ""):
        print(len(dataflow))
        total_bytes_len += len(dataflow)
        if len(dataflow) == 0:
            break
        start = time.time()
        result = ans(
            create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16)
        )
        print(f"cost: {time.time() - start} s")
        output = result["output_pcm"]
        outputs = outputs + output
    audiostream.close()

    outputs = outputs[:total_bytes_len]
    output_path = "zip_enhancer_output_stream.wav"
    with open(output_path, "wb") as out_wave:
        out_wave.write(
            create_wav_header(outputs, sample_rate=16000, num_channels=1, bits_per_sample=16)
        )


@app.command()
def offline():
    ans = pipeline(
        Tasks.acoustic_noise_suppression, model="iic/speech_zipenhancer_ans_multiloss_16k_base"
    )

    start = time.time()
    result = ans(
        # "https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise16k.wav",
        "./records/speech_with_noise16k.wav",
        # output_path="zip_enhancer_output.wav",
    )
    print(f"cost: {time.time() - start} s")
    print(len(result["output_pcm"]))


"""
python demo/modelscope/ans_zip_enhancer.py offline
python demo/modelscope/ans_zip_enhancer.py offline-stream
"""
if __name__ == "__main__":
    import torch
    import os

    thread_count = os.cpu_count()
    print(f"thread_count: {thread_count}")

    torch.set_num_threads(min(8, thread_count))
    torch.set_num_interop_threads(min(8, thread_count))

    app()
