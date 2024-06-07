import os
from typing import Union
from dataclasses import dataclass

import numpy as np
from pyannote.audio.core.io import AudioFile
import pyaudio
import torch

from .interface import IBuffering, IDetector, IAsr, ILlm, ITts


SRC_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
LOG_DIR = os.path.normpath(
    os.path.join(SRC_PATH, os.pardir, "log")
)
CONFIG_DIR = os.path.normpath(
    os.path.join(SRC_PATH, os.pardir, "config")
)
MODELS_DIR = os.path.normpath(
    os.path.join(SRC_PATH, os.pardir, "models")
)
RECORDS_DIR = os.path.normpath(
    os.path.join(SRC_PATH, os.pardir, "records")
)
TEST_DIR = os.path.normpath(
    os.path.join(SRC_PATH, os.pardir, "test")
)


@dataclass
class SessionCtx:
    client_id: str
    sampling_rate: int = 16000
    samples_width: int = 2
    frames = bytearray()
    buffering_stragegy: IBuffering = None
    waker: IDetector = None
    vad: IDetector = None
    asr: IAsr = None
    llm: ILlm = None
    tts: ITts = None
    on_session_start: callable = None
    on_session_end: callable = None
    vad_pyannote_audio: AudioFile = None
    # NOTE:
    # - openai-whisper or whispertimestamped use str(file_path)/np.ndarray/torch tensor
    # - transformers whisper use torch tensor/tf tensor
    # - faster whisper don't use torch tensor, use np.ndarray or str(file_path)/~BinaryIO~
    # - mlx whisper don't use torch tensor, use str(file_path)/np.ndarray/~mlx.array~
    asr_audio: Union[str, np.ndarray, torch.Tensor] = None
    language: str = "zh"


# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000
SILENCE_THRESHOLD = 500
# two seconds of silence marks the end of user voice input
SILENT_CHUNKS = 2 * RATE / CHUNK
# Set microphone id. Use list_microphones.py to see a device list.
MIC_IDX = 1

INT16_MAX_ABS_VALUE = 32768.0


@dataclass
class AudioRecoderArgs:
    format_: str = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    input_device_index: int = MIC_IDX
    frames_per_buffer: int = CHUNK


@dataclass
class SilenceAtEndOfChunkArgs:
    chunk_length_seconds: float
    chunk_offset_seconds: float


@dataclass
class PyannoteDetectorArgs:
    hf_auth_token: str = ""  # defualt use env HF_TOKEN
    path_or_hf_repo: str = "pyannote/segmentation-3.0"
    model_type: str = "segmentation-3.0"
    # remove speech regions shorter than that many seconds.
    min_duration_on: float = 0.3
    # fill non-speech regions shorter than that many seconds.
    min_duration_off: float = 0.3
    # if use pyannote/segmentation open onset/offset activation thresholds
    onset: float = 0.5
    offset: float = 0.5


INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0


@dataclass
class PorcupineDetectorArgs:
    wake_words: str = ""
    wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY
    wake_word_activation_delay: float = INIT_WAKE_WORD_ACTIVATION_DELAY
    wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT
    on_wakeword_detected: callable = None
    on_wakeword_timeout: callable = None
    on_wakeword_detection_start: callable = None
    on_wakeword_detection_end: callable = None


@dataclass
class WhisperASRArgs:
    download_path: str = ""
    model_name_or_path: str = "base"


@dataclass
class WhisperTimestampedASRArgs(WhisperASRArgs):
    pass


@dataclass
class WhisperFasterASRArgs(WhisperASRArgs):
    pass


@dataclass
class WhisperMLXASRArgs(WhisperASRArgs):
    pass


@dataclass
class WhisperTransformersASRArgs(WhisperASRArgs):
    pass
