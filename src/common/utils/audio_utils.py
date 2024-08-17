from scipy.signal import resample_poly
from scipy.io.wavfile import read, write
import pyloudnorm as pyln
import numpy as np
import torch

from src.common.types import INT16_MAX_ABS_VALUE


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


