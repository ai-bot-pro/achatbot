import json
import os

from scipy.signal import resample_poly
from scipy.io.wavfile import read, write
import pyloudnorm as pyln
import numpy as np
import torch

from src.common.types import INT16_MAX_ABS_VALUE, RECORDS_DIR


async def save_audio_to_file(
        audio_data,
        file_name,
        audio_dir=RECORDS_DIR,
        channles=1,
        sample_width=2,
        sample_rate=16000):
    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    import wave
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(channles)  # Assuming mono audio
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return file_path


async def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)


async def get_audio_segment(file_path, start=None, end=None):
    from pydub import AudioSegment
    with open(file_path, 'rb') as file:
        audio = AudioSegment.from_file(file, format="wav")
    if start is not None and end is not None:
        # pydub works in milliseconds
        return audio[start * 1000:end * 1000]
    return audio


async def read_audio_file(file_path):
    import wave
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return frames


def bytes2NpArrayWith16(frames: bytes | bytearray):
    # Convert the buffer frames to a NumPy array
    audio_array = np.frombuffer(frames, dtype=np.int16)
    # Normalize the array to a [-1, 1] range
    float_data = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
    return float_data


def bytes2TorchTensorWith16(frames: bytes | bytearray):
    float_data = bytes2NpArrayWith16(frames)
    waveform_tensor = torch.tensor(float_data, dtype=torch.float32)
    # don't Stereo, just Mono, reshape(1,-1) (1(channel),size(time))
    if waveform_tensor.ndim == 1:
        # float_data= float_data.reshape(1, -1)
        waveform_tensor = waveform_tensor.reshape(1, -1)
    return waveform_tensor


def npArray2bytes(np_arr: np.ndarray) -> bytearray:
    # Convert a NumPy array to bytes
    bytes_obj = np_arr.tobytes()
    # bytes -> bytearray
    byte_arr = bytearray(bytes_obj)
    return byte_arr


def torchTensor2bytes(tensor: torch.Tensor) -> bytearray:
    # Convert a NumPy array to bytes
    np_arr = tensor.numpy()

    return npArray2bytes(np_arr)


def postprocess_tts_wave_int16(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform with numpy.int16 to bytes
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk * (2 ** 15)
    chunk = chunk.astype(np.int16)
    return chunk.tobytes()


def postprocess_tts_wave(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform with numpy.float32 to bytes
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    chunk = chunk.astype(np.float32)
    return chunk.tobytes()


def convertSampleRateTo16khz(audio_data: bytes | bytearray, sample_rate):
    if sample_rate == 16000:
        return audio_data

    pcm_data = np.frombuffer(audio_data, dtype=np.int16)
    data_16000 = resample_poly(
        pcm_data, 16000, sample_rate)
    audio_data = data_16000.astype(np.int16).tobytes()

    return audio_data


def convert_sampling_rate_to_16k(input_file, output_file):
    original_rate, data = read(input_file)
    if original_rate == 16000:
        return
    up = 16000
    down = original_rate
    resampled_data = resample_poly(data, up, down)
    write(output_file, 16000, resampled_data.astype(np.int16))


def normalize_value(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)

    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)

    # Loudness goes from -20 to 80 (more or less), where -20 is quiet and 80 is
    # loud.
    loudness = normalize_value(loudness, -20, 80)

    return loudness


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    return prev_value + factor * (value - prev_value)
