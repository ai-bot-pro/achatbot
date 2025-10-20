import math
from typing import List
from scipy import signal
from scipy.io.wavfile import read, write

# import pyloudnorm as pyln
import numpy as np
import torch
import wave

from src.common.types import INT16_MAX_ABS_VALUE

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
}


def bytes2NpArrayWith16(data: bytes | bytearray):
    """Convert PCM buffer in s16le format to normalized NumPy array."""
    # Convert the buffer frames to a NumPy array
    audio_array = np.frombuffer(data, dtype=np.int16)
    # Normalize the array to a [-1, 1] range
    float_data = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
    return float_data


def bytes2TorchTensorWith16(data: bytes | bytearray):
    float_data = bytes2NpArrayWith16(data)
    waveform_tensor = torch.tensor(float_data, dtype=torch.float32)
    # don't Stereo, just Mono, reshape(1,-1) (1(channel),size(time))
    if waveform_tensor.ndim == 1:
        # float_data= float_data.reshape(1, -1)
        waveform_tensor = waveform_tensor.reshape(1, -1)
    return waveform_tensor  # (1, size(time))


def npArray2bytes(np_arr: np.ndarray) -> bytearray:
    # Convert a NumPy array to bytes
    bytes_obj = np_arr.tobytes()
    # bytes -> bytearray
    byte_arr = bytearray(bytes_obj)
    return byte_arr


def torchTensor2bytes(tensor: torch.Tensor) -> bytearray:
    # Convert a torch tensor to bytes
    np_arr = tensor.numpy()

    return npArray2bytes(np_arr)


def postprocess_tts_wave_int16(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform with numpy.int16 to bytes
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk * (2**15)
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


def convertSampleRateTo16khz(audio_data: bytes | bytearray, original_sample_rate):
    if original_sample_rate == 16000:
        return audio_data

    pcm_data = np.frombuffer(audio_data, dtype=np.int16)
    pcm_data_16K = resample_audio(pcm_data, original_sample_rate, 16000)
    audio_data = pcm_data_16K.tobytes()

    return audio_data


def resample_audio(pcm_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    num_samples = math.ceil(len(pcm_data) * target_rate / original_rate)
    resampled_audio = signal.resample(pcm_data, num_samples)
    # resampled_audio = signal.resample_poly(pcm_data, target_rate, original_rate)
    return resampled_audio.astype(np.int16)


def convert_sampling_rate_to_16k(input_file, output_file):
    original_rate, data = read(input_file)
    if original_rate == 16000:
        return
    up = 16000
    down = original_rate
    resampled_data = signal.resample_poly(data, up, down)
    write(output_file, 16000, resampled_data.astype(np.int16))


def combine_audio_segments(
    segments: List[np.ndarray], crossfade_duration=0.04, sr=24000
) -> np.ndarray:
    """Smoothly combine audio segments using crossfade transitions." """
    window_length = int(sr * crossfade_duration)
    hanning_window = np.hanning(2 * window_length)
    # Combine
    for i, segment in enumerate(segments):
        if i == 0:
            combined_audio = segment
        else:
            overlap = (
                combined_audio[-window_length:] * hanning_window[window_length:]
                + segment[:window_length] * hanning_window[:window_length]
            )
            combined_audio = np.concatenate(
                [combined_audio[:-window_length], overlap, segment[window_length:]]
            )
    return combined_audio


def read_wav_to_bytes(file_path) -> tuple[bytes, int]:
    with wave.open(file_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if num_channels not in (1, 2):
            raise Exception(f"WAV file must be mono or stereo")

        if sample_width != 2:
            raise Exception(f"WAV file must be 16-bit")

        raw = wf.readframes(num_frames)
    return raw, sample_rate


def read_wav_to_np(file_path) -> tuple[np.ndarray, int]:
    with wave.open(file_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if num_channels not in (1, 2):
            raise Exception(f"WAV file must be mono or stereo")

        if sample_width != 2:
            raise Exception(f"WAV file must be 16-bit")

        raw = wf.readframes(num_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if num_channels == 1:
        pcmf32 = audio / 32768.0
    else:
        audio = audio.reshape(-1, 2)
        # Averaging the two channels
        pcmf32 = (audio[:, 0] + audio[:, 1]) / 65536.0
    return pcmf32, sample_rate


"""
python -m src.common.utils.audio_utils
"""
if __name__ == "__main__":
    import time
    import librosa

    audio_file = "./test/audio_files/asr_example_zh.wav"
    start = time.perf_counter()
    audio_np, sr = read_wav_to_np(audio_file)
    end = time.perf_counter()
    print(audio_np, "wave cost--->", end - start)

    start = time.perf_counter()
    audio_np, _ = librosa.load(audio_file, sr=16000)
    end = time.perf_counter()
    print(audio_np, "librosa cost--->", end - start)
