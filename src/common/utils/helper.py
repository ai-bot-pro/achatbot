import hashlib
import json
import logging
import platform
import threading

import pyloudnorm as pyln
import numpy as np
import torch


async def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


async def get_audio_segment(file_path, start=None, end=None):
    from pydub import AudioSegment

    with open(file_path, "rb") as file:
        audio = AudioSegment.from_file(file, format="wav")
    if start is not None and end is not None:
        # pydub works in milliseconds
        return audio[start * 1000 : end * 1000]
    return audio


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    return prev_value + factor * (value - prev_value)


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


def normalize_value(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.debug(model)
    logging.info(f"{extra_info} {model_million_params} M parameters")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def file_md5_hash(file_path: str):
    # Compute a hash of the file
    with open(file_path, "rb") as _file:
        data = _file.read()
        data_hash = hashlib.md5(data).hexdigest()
        return data_hash


class ThreadSafeDict:
    def __init__(self):
        self._map = {}
        self._lock = threading.RLock()

    def set(self, key, value):
        with self._lock:
            self._map[key] = value

    def get(self, key):
        with self._lock:
            return self._map.get(key)

    def pop(self, key):
        with self._lock:
            return self._map.pop(key, None)

    def delete(self, key):
        with self._lock:
            if key in self._map:
                del self._map[key]

    def contains(self, key):
        with self._lock:
            return key in self._map

    def size(self):
        with self._lock:
            return len(self._map)

    def keys(self):
        with self._lock:
            return list(self._map.keys())

    def values(self):
        with self._lock:
            return list(self._map.values())

    def items(self):
        with self._lock:
            return list(self._map.items())
