
from dataclasses import dataclass
from _typeshed import ReadableBuffer

from pyannote.audio.core.io import AudioFile

from src.common.interface import IBuffering, IDetector, IAsr, ILlm, ITts


@dataclass
class SessionCtx:
    client_id: str
    sampling_rate: int = 16000
    samples_width: int = 2
    buffering: IBuffering = None
    vad: IDetector = None
    asr: IAsr = None
    llm: ILlm = None
    tts: ITts = None
    on_session_start: callable = None
    on_session_end: callable = None
    vad_pyannote_audio: AudioFile = None
    wake_word_buffer: ReadableBuffer = None


@dataclass
class SilenceAtEndOfChunkArgs:
    chunk_length_seconds: float
    chunk_offset_seconds: float


@dataclass
class PyannoteDetectorArgs:
    hf_auth_token: str
    path_or_hf_repo: str = "pyannote/segmentation-3.0"
    model_type: str = "segmentation-3.0"
    # remove speech regions shorter than that many seconds.
    min_duration_on: float = 0.3,
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
