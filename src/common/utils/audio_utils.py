import json
import os

import numpy as np
import torch

from src.common.types import INT16_MAX_ABS_VALUE


async def save_audio_to_file(
        audio_data,
        file_name,
        audio_dir="records",
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


def bytes2NpArrayWith16(frames: bytearray):
    # Convert the buffer frames to a NumPy array
    audio_array = np.frombuffer(frames, dtype=np.int16)
    # Normalize the array to a [-1, 1] range
    float_data = audio_array.astype(
        np.float32) / INT16_MAX_ABS_VALUE
    return float_data


def bytes2TorchTensorWith16(frames: bytearray):
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


def postprocess_tts_wave(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    chunk = chunk.astype(np.float32)
    return chunk.tobytes()
