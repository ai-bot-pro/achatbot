r"""
use SOTA LLM like chatGPT to generate config file(json,yaml,toml) from dataclass type
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Literal,
    Optional,
    Sequence,
    List,
    Any,
    Mapping,
    Union,
)
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic.main import BaseModel
from pydantic import ConfigDict

from .interface import (
    IBuffering,
    IDetector,
    IAsr,
    ILlm,
    ITts,
    IVADAnalyzer,
)
from .factory import EngineClass

load_dotenv(override=True)


SRC_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DIR_PATH = os.path.join(SRC_PATH, os.pardir)
if bool(os.getenv("ACHATBOT_PKG", "")):
    home_dir = Path.home()
    DIR_PATH = os.path.join(home_dir, ".achatbot")

LOG_DIR = os.path.normpath(os.path.join(DIR_PATH, "log"))
CONFIG_DIR = os.path.normpath(os.path.join(DIR_PATH, "config"))
MODELS_DIR = os.path.normpath(os.path.join(DIR_PATH, "models"))
RECORDS_DIR = os.path.normpath(os.path.join(DIR_PATH, "records"))
VIDEOS_DIR = os.path.normpath(os.path.join(DIR_PATH, "videos"))
ASSETS_DIR = os.path.normpath(os.path.join(DIR_PATH, "assets"))

TEST_DIR = os.path.normpath(os.path.join(SRC_PATH, os.pardir, "test"))

# audio stream default configuration
CHUNK = 1600  # 100ms 16k rate num_frames
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = 2

# llm sys default prompt
DEFAULT_SYSTEM_PROMPT = "你是一个中国人,一名中文助理，请用中文简短回答，回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"


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
            d["state"]["tts_chunk_len"] = len(self.state["tts_chunk"])
            d["state"].pop("tts_chunk")

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

# pyaudio format
#   >>> print(int(pyaudio.paInt16))
#   8
PYAUDIO_PAINT16 = 8
#   >>> print(int(pyaudio.paInt24))
#   4
PYAUDIO_PAINT24 = 4
#   >>> print(int(pyaudio.paInt32))
#   2
PYAUDIO_PAINT32 = 2
#   >>> print(int(pyaudio.paFloat32))
#   1
PYAUDIO_PAFLOAT32 = 1
#   >>> print(int(pyaudio.paInt8))
#   16
PYAUDIO_PAINT8 = 16
#   >>> print(int(pyaudio.paUInt8))
#   32
PYAUDIO_PAUINT8 = 32
#   >>> print(int(pyaudio.paCustomFormat))
#   65536
PYAUDIO_PACUSTOMFORMAT = 65536

# PortAudio Callback Return Codes

PYAUDIO_PACONTINUE = 0  #: There is more audio data to come
PYAUDIO_PACOMPLETE = 1  #: This was the last block of audio data
PYAUDIO_PAABORT = 2  #: An error ocurred, stop playback/recording


@dataclass
class AudioStreamArgs:
    stream_callback: Optional[str] = None
    input: bool = False
    output: bool = False
    frames_per_buffer: int = CHUNK


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


@dataclass
class FSMNVADArgs:
    sample_rate: int = RATE
    model: str = "fsmn-vad"
    model_version: str = "v2.0.4"
    check_frames_mode: int = VAD_CHECK_PER_FRAMES
    # frame_duration_ms: int = 64  # ms,  frame_length: 1024 = 16000*64 / 1000


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
    trust_repo: bool = True
    verbose: bool = True
    onnx: bool = False
    silero_sensitivity: float = INIT_SILERO_SENSITIVITY
    is_pad_tensor: bool = False
    check_frames_mode: int = VAD_CHECK_PER_FRAMES


@dataclass
class WebRTCSileroVADArgs(WebRTCVADArgs, SileroVADArgs):
    pass


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
class LLamcppLLMArgs:
    save_chat_history: bool = True
    model_name: str = ""
    model_type: str = "generate"  # generate < chat | chat-func
    model_path: str = ""
    n_threads: int = 1
    n_batch: int = 8
    n_gpu_layers: int = 0
    n_ctx: int = 4096
    # chatml | chatml-function-calling | functionary-v2 | minicpm-v-2.6 ..
    chat_format: Optional[str] = None
    tokenizer_path: Optional[str] = None
    clip_model_path: Optional[str] = None
    verbose: bool = True
    flash_attn: bool = False
    # llm
    llm_prompt_tpl: str = "<|user|>\n{%s}<|end|>\n<|assistant|>"
    llm_stop: Optional[Union[int, List[str]]] = field(default_factory=list)
    llm_stop_ids: Optional[Union[int, List[int]]] = field(default_factory=list)
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.8
    llm_top_p: float = 0.95
    llm_min_p: float = 0.05
    llm_top_k: int = 40
    llm_seed: Optional[int] = None
    # repeat penalty
    llm_repeat_penalty: float = 1.1
    llm_repeat_last_n: int = 64
    # chat stream
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


# ---------------- TTS -----------------
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


PYTTSX3_SYNTHESIS_FILE = "pyttsx3_synthesis.wav"


@dataclass
class Pyttsx3TTSArgs:
    voice_name: str = "Tingting"


GTTS_SYNTHESIS_FILE = "gtts_synthesis.wav"


@dataclass
class GTTSArgs:
    language: str = "en"
    tld: str = "com"
    slow: bool = False
    speed_increase: float = 1.0
    chunk_size: int = 100
    crossfade_lenght: int = 10


EDGE_TTS_SYNTHESIS_FILE = "edge_tts_synthesis.wav"


@dataclass
class EdgeTTSArgs:
    language: str = "en"
    gender: str = "Female"
    voice_name: str = "en-GB-SoniaNeural"
    rate: str = "+15%"
    volume: str = "+0%"
    pitch: str = "+0Hz"


@dataclass
class CosyVoiceTTSArgs:
    r"""
    CosyVoice:
    For zero_shot/cross_lingual/vc(voice convert) inference, please use CosyVoice-300M model.
    For sft inference, please use CosyVoice-300M-SFT model.
    For instruct inference, please use CosyVoice-300M-Instruct model.

    CosyVoice2:
    - if no reference_text and no instruct_text, default use cross lingual infer;
        fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    - if have reference_text and no instruct_text, use zero shot infer;
    - if have no reference_text and have instruct_text, use instruct2 infer;
    """

    model_dir: str = os.path.join(MODELS_DIR, "CosyVoice-300M-SFT")  # sft model

    reference_text: str = ""  # reference audio text
    reference_audio_path: str = ""  # 16k sample rate audio file for base pt model

    src_audio_path: str = ""  # for voice convert from src to reference audio

    instruct_text: str = ""  # use with instruct model
    spk_id: str = ""  # use with sft and instruct model when use CosyVoice, CosyVoice2 instruct2 method remove spk_id

    language: str = "zh"
    tts_default_silence_duration: float = 0.3
    tts_comma_silence_duration: float = 0.3
    tts_sentence_silence_duration: float = 0.6
    tts_speed: float = 1.0

    # stream
    tts_stream: bool = False
    chunk_length_seconds: float = 1.0


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


# --------------- in/out audio camera(video) params -------------------


class AudioParams(BaseModel):
    audio_out_enabled: bool = False
    audio_out_sample_rate: int = RATE
    audio_out_channels: int = CHANNELS
    audio_in_enabled: bool = False
    audio_in_participant_enabled: bool = False
    audio_in_sample_rate: int = RATE
    audio_in_channels: int = CHANNELS


class AudioVADParams(AudioParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: IVADAnalyzer | EngineClass | None = None


class CameraParams(BaseModel):
    camera_in_enabled: bool = False
    camera_in_color_format: str = "RGB"
    camera_out_enabled: bool = False
    camera_out_is_live: bool = False
    camera_out_width: int = 1024
    camera_out_height: int = 768
    camera_out_bitrate: int = 800000
    camera_out_framerate: int = 30
    camera_out_color_format: str = "RGB"


class AudioCameraParams(CameraParams, AudioVADParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    extra: Mapping[str, Any] = {"interim_results": True}


class DailyParams(AudioCameraParams):
    api_url: str = "https://api.daily.co/v1"
    api_key: str = ""
    dialin_settings: DailyDialinSettings | None = None
    transcription_enabled: bool = False
    transcription_settings: DailyTranscriptionSettings = DailyTranscriptionSettings()


class DailyRoomArgs(BaseModel):
    privacy: Literal["private", "public"] = "public"


# --------------- livekit -------------------------------


class LivekitParams(AudioCameraParams):
    # audio_in_sample_rate: int = 48000  # livekit audio in stream default sample rate 48000
    websocket_url: str = ""  # project url
    api_key: str = ""
    api_secret: str = ""
    e2ee_shared_key: Optional[bytes] = None
    sandbox_room_url: str = "https://ultra-terminal-re8nmd.sandbox.livekit.io"


class LivekitRoomArgs(BaseModel):
    bot_name: str = "chat-bot"
    # if use session, need manual close async http session
    is_common_session: bool = False


# --------------- agora -------------------------------


class AgoraParams(AudioCameraParams):
    app_id: str = ""
    app_cert: str = ""
    enable_pcm_dump: bool = False
    demo_voice_url: str = "https://webdemo.agora.io/basicVoiceCall/index.html"
    demo_video_url: str = "https://webdemo.agora.io/basicVideoCall/index.html"


class AgoraChannelArgs(BaseModel):
    pass


# ---------------- Room Bots -------------
class GeneralRoomInfo(BaseModel):
    """general room info for diff webRTC room info to align"""

    sid: str = ""
    name: str = ""
    url: str = ""
    ttl_s: int | None = None
    creation_time: str | None = None
    extra_data: dict = field(default_factory=dict)


@dataclass
class BotRunArgs:
    room_name: str = ""
    room_url: str = ""
    token: str = ""
    bot_config: dict | None = None
    bot_name: str | None = None
    bot_config_list: list | None = None
    services: dict | None = None
    websocket_server_host: str = "localhost"
    websocket_server_port: int = 8765
    handle_sigint: bool = True
