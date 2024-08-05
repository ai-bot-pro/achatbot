r"""
use SOTA LLM like chatGPT to generate config file(json,yaml,toml) from dataclass type
"""
from dataclasses import dataclass
from enum import Enum
from typing import (
    IO,
    Optional,
    Sequence,
    Union,
    List,
    Any,
    Mapping,
)
import os

import numpy as np
import pyaudio
import torch
from pydantic.main import BaseModel
from pydantic import ConfigDict

from .interface import (
    IAudioStream,
    IBuffering, IDetector, IAsr,
    ILlm, ITts, IVADAnalyzer,
)
from .factory import EngineClass


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
CHUNK = 1600  # 100ms 16k rate num_frames
# pyaudio format
#   >>> print(int(pyaudio.paInt16))
#   8
#   >>> print(int(pyaudio.paInt24))
#   4
#   >>> print(int(pyaudio.paInt32))
#   2
#   >>> print(int(pyaudio.paFloat32))
#   1
#   >>> print(int(pyaudio.paInt8))
#   16
#   >>> print(int(pyaudio.paUInt8))
#   32
#   >>> print(int(pyaudio.paCustomFormat))
#   65536
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
    waker: IDetector | EngineClass = None
    vad: IDetector | EngineClass = None
    asr: IAsr | EngineClass = None
    llm: ILlm | EngineClass = None
    tts: ITts | EngineClass = None
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
# 1 seconds of silence marks the end of user voice input
SILENT_CHUNKS = 1 * RATE / CHUNK
# record silence timeout (10s)
SILENCE_TIMEOUT_S = 10
# Set microphone id. Use list_microphones.py to see a device list.
MIC_IDX = 1
# 2^(16-1)
INT16_MAX_ABS_VALUE = 32768.0


@dataclass
class AudioStreamInfo:
    in_channels: int = CHANNELS
    in_sample_rate: int = RATE
    in_sample_width: int = SAMPLE_WIDTH
    in_frames_per_buffer: int = CHUNK
    out_channels: int = CHANNELS
    out_sample_rate: int = RATE
    out_sample_width: int = SAMPLE_WIDTH
    out_frames_per_buffer: int = CHUNK
    pyaudio_out_format: int | None = FORMAT


@dataclass
class AudioStreamArgs:
    stream_callback: Optional[str] = None
    input: bool = False
    output: bool = False
    frames_per_buffer: int = CHUNK


@dataclass
class PyAudioStreamArgs(AudioStreamArgs):
    format: int = FORMAT
    channels: int = CHANNELS
    rate: int = RATE
    sample_width: int = SAMPLE_WIDTH
    input_device_index: int = None
    output_device_index: int = None


@dataclass
class DailyAudioStreamArgs(AudioStreamArgs):
    """
    in live room,one joined client instance need diff in/out stream sample rate
    """
    meeting_room_url: str = ""
    bot_name: str = "chat-bot"
    meeting_room_token: str = ""
    in_channels: int = CHANNELS
    in_sample_rate: int = RATE
    in_sample_width: int = SAMPLE_WIDTH
    out_channels: int = CHANNELS
    out_sample_rate: int = RATE
    out_sample_width: int = SAMPLE_WIDTH
    out_queue_timeout_s: float = 0.1
    is_async_output: bool = False


@dataclass
class AudioRecoderArgs:
    silent_chunks: int = SILENT_CHUNKS
    silence_timeout_s: int = SILENCE_TIMEOUT_S
    num_frames: int = CHUNK
    is_stream_callback: bool = False
    no_stream_sleep_time_s: float = 0.03  # for dequeue get


@dataclass
class VADRecoderArgs(AudioRecoderArgs):
    padding_ms: int = 300
    active_ratio: float = 0.75
    silent_ratio: float = 0.75


@dataclass
class AudioPlayerArgs:
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
    aggressiveness: int = 3  # 0,1,2,3
    sample_rate: int = RATE
    check_frames_mode: int = VAD_CHECK_PER_FRAMES
    frame_duration_ms: int = 20  # ms


INIT_SILERO_SENSITIVITY = 0.4
# How often should we reset internal model state
SILERO_MODEL_RESET_STATES_TIME = 5.0


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
class WebRTCSileroVADArgs(WebRTCVADArgs, SileroVADArgs):
    pass


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
    model_type: str = "generate"  # generate < chat | chat-func
    model_path: str = ""
    n_threads: int = 1
    n_batch: int = 8
    n_gpu_layers: int = 0
    n_ctx: int = 2048
    chat_format: Optional[str] = None  # chatml | chatml-function-calling | functionary-v2 ..
    tokenizer_path: Optional[str] = None
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
    # "none" | "auto" | dict function like this: { "type": "function", "function": { "name": "get_current_weather" } },
    llm_tool_choice: str = None


@dataclass
class PersonalAIProxyArgs:
    max_retry_cn: int = 2
    api_url: str = "http://localhost:8787/"
    chat_bot: str = "openai"  # openai | qianfan
    openai_api_base_url: str = "https://api.groq.com/openai/v1/"
    model_type: str = "chat_only"  # chat_only | chat_with_functions
    model_name: str = ""
    llm_chat_system: str = ""
    llm_max_tokens: int = 1024
    llm_stream: bool = False
    llm_stop: Optional[List[str]] = None
    llm_temperature: float = 0.8
    llm_top_p: float = 0.95
    func_search_name: str = "search_api"
    func_weather_name: str = "openweathermap_api"


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
    use_flash_attn: bool = False
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


@dataclass
class CosyVoiceTTSArgs:
    r"""
    For zero_shot/cross_lingual inference, please use CosyVoice-300M model.
    For sft inference, please use CosyVoice-300M-SFT model.
    For instruct inference, please use CosyVoice-300M-Instruct model.
    """
    model_dir: str = os.path.join(MODELS_DIR, "CosyVoice-300M-SFT")  # sft model
    reference_audio_path: str = ""  # 16k sample rate audio file for base pt model
    instruct_text: str = ""  # use with instruct model
    spk_id: str = ""  # use with sft and instruct model
    language: str = "zh"
    tts_default_silence_duration: float = 0.3
    tts_comma_silence_duration: float = 0.3
    tts_sentence_silence_duration: float = 0.6


# ------------- llm function calling -----------------
@dataclass
class OpenWeatherMapArgs:
    lang: str = "zh_cn"
    units: str = "metric"


@dataclass
class SearchApiArgs:
    engine: str = "google"
    gl: str = "cn"
    hl: str = "zh-cn"
    page: int = 1
    num: int = 5


@dataclass
class Search1ApiArgs:
    search_service: str = "google"
    image: bool = False
    crawl_results: int = 0
    max_results: int = 5


@dataclass
class SerperApiArgs:
    gl: str = "cn"
    hl: str = "zh-cn"
    page: int = 1
    num: int = 5


# --------------- vad analyzer------------------
DAILY_WEBRTC_VAD_RESET_PERIOD_MS = 2000


class VADState(Enum):
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


@dataclass
class VADAnalyzerArgs:
    sample_rate: int = RATE
    num_channels: int = CHANNELS
    confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_volume: float = 0.6


@dataclass
class SileroVADAnalyzerArgs(SileroVADArgs, VADAnalyzerArgs):
    pass

# --------------- daily -------------------------------


class DailyDialinSettings(BaseModel):
    call_id: str = ""
    call_domain: str = ""


# https://reference-python.daily.co/types.html#transcriptionsettings
class DailyTranscriptionSettings(BaseModel):
    language: str = "en"
    tier: str = "nova"
    model: str = "2-conversationalai"
    profanity_filter: bool = True
    redact: bool = False
    endpointing: bool = True
    punctuate: bool = True
    includeRawResponse: bool = True
    extra: Mapping[str, Any] = {
        "interim_results": True
    }


class AudioParams(BaseModel):
    audio_out_enabled: bool = False
    audio_out_sample_rate: int = RATE
    audio_out_channels: int = CHANNELS
    audio_in_enabled: bool = False
    audio_in_sample_rate: int = RATE
    audio_in_channels: int = CHANNELS


class AudioVADParams(AudioParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: IVADAnalyzer | None = None


class CameraParams(BaseModel):
    camera_out_enabled: bool = False
    camera_out_is_live: bool = False
    camera_out_width: int = 1024
    camera_out_height: int = 768
    camera_out_bitrate: int = 800000
    camera_out_framerate: int = 30
    camera_out_color_format: str = "RGB"


class AudioCameraParams(CameraParams, AudioVADParams):
    pass


class DailyParams(AudioCameraParams):
    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    dialin_settings: DailyDialinSettings | None = None
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


# ---------------- Bots -------------

@dataclass
class DailyRoomBotArgs:
    room_url: str = ""
    token: str = ""
    bot_config: dict = None
    bot_name: str | None = None
