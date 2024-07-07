r"""
use SOTA LLM like chatGPT to generate config file(json,yaml,toml) from dataclass type
"""
from dataclasses import dataclass
from typing import (
    IO,
    Optional,
    Sequence,
    Union,
    List
)
import os

import numpy as np
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

# audio stream default configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = 2


@dataclass
class SessionCtx:
    client_id: str
    sampling_rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    read_audio_frames = bytes()
    state = dict()
    buffering_strategy: IBuffering = None
    waker: IDetector = None
    vad: IDetector = None
    asr: IAsr = None
    llm: ILlm = None
    tts: ITts = None
    on_session_start: callable = None
    on_session_end: callable = None

    def __repr__(self) -> str:
        d = {
            "client_id": self.client_id,
            "sampling_rate": self.sampling_rate,
            "sample_width": self.sample_width,
            "read_audio_frames_len": len(self.read_audio_frames),
            "state": self.state,
        }
        if "tts_chunk" in self.state:
            d['state']["tts_chunk_len"] = len(self.state['tts_chunk'])
            d['state'].pop("tts_chunk")

        res = f"session ctx: {d}"
        return res

    def __getstate__(self):
        return {
            "client_id": self.client_id,
            "sampling_rate": self.sampling_rate,
            "sample_width": self.sample_width,
            "read_audio_frames": self.read_audio_frames,
            "state": self.state,
        }

    def __setstate__(self, state):
        self.client_id = state["client_id"]
        self.sampling_rate = state["sampling_rate"]
        self.sample_width = state["sample_width"]
        self.read_audio_frames = state["read_audio_frames"]
        self.state = state["state"]


# audio recorder default configuration
SILENCE_THRESHOLD = 500
# two seconds of silence marks the end of user voice input
SILENT_CHUNKS = 2 * RATE / CHUNK
# Set microphone id. Use list_microphones.py to see a device list.
MIC_IDX = 1
# 2^(16-1)
INT16_MAX_ABS_VALUE = 32768.0


@dataclass
class AudioStreamArgs:
    format: str = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    frames_per_buffer: int = CHUNK
    input_device_index: int = None
    output_device_index: int = None
    input: bool = False
    output: bool = False


@dataclass
class AudioRecoderArgs:
    format: str = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    input_device_index: int = None
    frames_per_buffer: int = CHUNK
    silence_timeout_s: int = 10


@dataclass
class AudioRecoderArgs:
    format: str = FORMAT


@dataclass
class AudioPlayerArgs:
    format: str = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    output_device_index: int = None
    frames_per_buffer: int = CHUNK
    on_play_start: Optional[str] = None
    on_play_end: Optional[str] = None
    on_play_chunk: Optional[str] = None
    is_immediate_stop: bool = False


@dataclass
class SilenceAtEndOfChunkArgs:
    chunk_length_seconds: float
    chunk_offset_seconds: float


VAD_CHECK_PER_FRAMES = 1
VAD_CHECK_ALL_FRAMES = 2


@dataclass
class WebRTCVADArgs:
    aggressiveness: int = 1  # 0,1,2,3
    sample_rate: int = RATE
    check_frames_mode: int = VAD_CHECK_PER_FRAMES
    frame_duration_ms: int = 10  # ms


INIT_SILERO_SENSITIVITY = 0.4


@dataclass
class SileroVADArgs:
    sample_rate: int = RATE
    repo_or_dir: str = "snakers4/silero-vad"
    model: str = "silero_vad"
    source: str = "github"  # github | local
    force_reload: bool = False
    verbose: bool = True
    onnx: bool = False
    silero_sensitivity: float = INIT_SILERO_SENSITIVITY
    is_pad_tensor: bool = False
    check_frames_mode: int = VAD_CHECK_PER_FRAMES


@dataclass
class PyannoteDetectorArgs:
    from pyannote.audio.core.io import AudioFile
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
    # vad
    vad_pyannote_audio: AudioFile = None


INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0


@dataclass
class PorcupineDetectorArgs:
    access_key: str = ""
    keyword_paths: Optional[Sequence[str]] = None
    model_path: Optional[str] = None
    library_path: Optional[str] = None
    wake_words: str = ""
    wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY
    # wake_word_activation_delay: float = INIT_WAKE_WORD_ACTIVATION_DELAY
    # wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT
    on_wakeword_detected: Optional[str] = None
    # on_wakeword_timeout: Optional[str] = None
    # on_wakeword_detection_start: Optional[str] = None
    # on_wakeword_detection_end: Optional[str] = None


@dataclass
class ASRArgs:
    download_path: str = ""
    model_name_or_path: str = "base"
    # asr
    # NOTE:
    # - openai-whisper or whispertimestamped use str(file_path)/np.ndarray/torch tensor
    # - transformers whisper use torch tensor/tf tensor
    # - faster whisper don't use torch tensor, use np.ndarray or str(file_path)/~BinaryIO~
    # - mlx whisper don't use torch tensor, use str(file_path)/np.ndarray/~mlx.array~
    # - funasr whisper, SenseVoiceSmall use str(file_path)/torch tensor
    asr_audio: str | bytes | IO[bytes] | np.ndarray | torch.Tensor = None
    # https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
    language: str = "zh"
    verbose: bool = True
    prompt: str = ""
    sample_rate: int = RATE


@dataclass
class WhisperTimestampedASRArgs(ASRArgs):
    pass


@dataclass
class WhisperFasterASRArgs(ASRArgs):
    from faster_whisper.vad import VadOptions
    vad_filter: bool = False
    vad_parameters: Optional[Union[dict, VadOptions]] = None


@dataclass
class WhisperMLXASRArgs(ASRArgs):
    pass


@dataclass
class WhisperTransformersASRArgs(ASRArgs):
    pass


@dataclass
class WhisperGroqASRArgs(ASRArgs):
    timeout_s: float = None
    """
    temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the
    output more random, while lower values like 0.2 will make it more focused and
    deterministic. If set to 0, the model will use
    [log probability](https://en.wikipedia.org/wiki/Log_probability) to
    automatically increase the temperature until certain thresholds are hit.
    """
    temperature: float = 0.0
    # timestamp_granularities: list = []
    # response_format: str = "json"  # json, verbose_json, text


@dataclass
class LLamcppLLMArgs:
    model_name: str = ""
    model_type: str = ""
    model_path: str = ""
    n_threads: int = 1
    n_batch: int = 8
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    chat_format: str = "chatml"
    verbose: bool = True
    flash_attn: bool = False
    # llm
    llm_prompt_tpl: str = "<|user|>\n{%s}<|end|>\n<|assistant|>"
    llm_stop: Optional[List[str]] = None
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.8
    llm_top_p: float = 0.95
    llm_top_k: int = 40
    llm_stream: bool = False
    llm_chat_system: str = ""


@dataclass
class CoquiTTSArgs:
    conf_file: str = ""
    model_path: str = ""
    reference_audio_path: str = ""
    tts_temperature: float = 0.75
    tts_top_p: float = 0.85
    tts_stream: bool = False
    tts_length_penalty: float = 1.0
    tts_repetition_penalty: float = 10.0
    tts_num_beams: int = 1
    tts_speed: float = 1.0
    tts_stream_chunk_size: int = 20
    tts_overlap_wav_len: int = 1024
    tts_enable_text_splitting: bool = False
    tts_language: str = "zh"
    tts_default_silence_duration: float = 0.3
    tts_comma_silence_duration: float = 0.3
    tts_sentence_silence_duration: float = 0.6
    tts_use_deepspeed: bool = False


@dataclass
class ChatTTSArgs:
    source: str = 'huggingface'  # local | huggingface
    force_redownload: bool = False
    local_path: str = ''
    compile: bool = True
    device: str = None
    skip_refine_text: bool = False
    refine_text_only: bool = False
    params_refine_text: dict = None
    params_infer_code: dict = None
    use_decoder: bool = True
    do_text_normalization: bool = False
    lang: str = None
    tts_stream: bool = False


PYTTSX3_SYNTHESIS_FILE = 'pyttsx3_synthesis.wav'


@dataclass
class Pyttsx3TTSArgs:
    voice_name: str = "Tingting"


GTTS_SYNTHESIS_FILE = 'gtts_synthesis.wav'


@dataclass
class GTTSArgs:
    language: str = "en"
    tld: str = "com"
    slow: bool = False
    speed_increase: float = 1.0
    chunk_size: int = 100
    crossfade_lenght: int = 10


EDGE_TTS_SYNTHESIS_FILE = 'edge_tts_synthesis.wav'


@dataclass
class EdgeTTSArgs:
    language: str = "en"
    gender: str = "Female"
    voice_name: str = "en-GB-SoniaNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"
