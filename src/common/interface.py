from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncGenerator, Generator, List, Dict, Optional, Tuple

from src.types.speech.turn_analyzer import EndOfTurnState

import numpy as np


class IPoolInstance(ABC):
    """池化对象的接口，需要实现 Reset 和 Release 方法。"""

    @abstractmethod
    def reset(self) -> None:
        """重置对象状态，以便下次使用。"""
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def release(self) -> None:
        """释放对象资源。"""
        raise NotImplementedError("must be implemented in the child class")


class IModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        raise NotImplementedError("must be implemented in the child class")


class IAudioStream(ABC):
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    @abstractmethod
    def open_stream(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def start_stream(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def stop_stream(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close_stream(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def is_stream_active(self) -> bool:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def write_stream(self, data):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def get_stream_info(self):
        """
        return AudioStreamInfo
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def read_stream(self, num_frames) -> bytes:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError("must be implemented in the child class")


class IRecorder(ABC):
    @abstractmethod
    def open(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def record_audio(self, session) -> list[bytes]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def record_audio_generator(self, session) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_in_stream(self, audio_stream):
        raise NotImplementedError("must be implemented in the child class")


class IBuffering(ABC):
    @abstractmethod
    def process_audio(self, session):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def is_voice_active(self, session):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def insert(self, audio_data):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def clear(self):
        raise NotImplementedError("must be implemented in the child class")


class IDetector(ABC):
    @abstractmethod
    async def detect(self, session) -> list[dict | bytes | None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_audio_data(self, audio_data):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def get_sample_info(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplementedError("must be implemented in the child class")


class IVADAnalyzer(ABC):
    @abstractmethod
    def voice_confidence(self, buffer) -> float:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def analyze_audio(self, buffer):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def reset(self):
        """reset vad stats and model stats"""
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def num_frames_required(self) -> int:
        """required audio frames num to analyze"""
        raise NotImplementedError("must be implemented in the child class")


class ITurnAnalyzer(ABC):
    """Abstract base class for analyzing user end of turn.

    This class defines the abstract interface for turn analyzers, which are
    responsible for determining when a user has finished speaking.
    """

    @property
    @abstractmethod
    def speech_triggered(self) -> bool:
        """Determines if speech has been detected.

        Returns:
            bool: True if speech is triggered, otherwise False.
        """
        pass

    @abstractmethod
    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Appends audio data for analysis.

        Args:
            buffer (bytes): The audio data to append.
            is_speech (bool): Indicates whether the appended audio is speech or not.

        Returns:
            EndOfTurnState: The resulting state after appending the audio.
        """
        pass

    @abstractmethod
    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[Dict[str, Any]]]:
        """Analyzes if an end of turn has occurred based on the audio input.

        Returns:
            EndOfTurnState: The result of the end of turn analysis.
        """
        pass

    @abstractmethod
    def clear(self):
        """Reset the turn analyzer to its initial state."""
        pass


class ISpeechEnhancer(ABC):
    @abstractmethod
    def enhance(self, session, **kwargs) -> bytes:
        """
        session
            - session.ctx.state["audio_chunk"] bytes
            - session.ctx.state["sample_rate"] int
            - session.ctx.state["is_last"] bool
        return
        - bytes
        """
        raise NotImplementedError("must be implemented in the child class")

    def warmup(self, session, **kwargs):
        raise NotImplementedError("must be implemented in the child class")

    def reset(self):
        raise NotImplementedError("must be implemented in the child class")


class IAsr(ABC):
    @abstractmethod
    def transcribe_stream_sync(self, session) -> Generator[str, None, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def transcribe_stream(self, session) -> AsyncGenerator[str, None]:
        """decode stream (text token step by step)"""
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def transcribe(self, session) -> dict:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_audio_data(self, audio_data):
        raise NotImplementedError("must be implemented in the child class")


class IAsrLive(ABC):
    @abstractmethod
    async def streaming_transcribe(self, session, **kwargs) -> AsyncGenerator[dict, None]:
        """
        session
            - session.ctx.state["audio_chunk"]
            - session.ctx.state["is_last"]
        return
        - {"timestamps":[],"text":""}
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("must be implemented in the child class")


class IPunc(ABC):
    @abstractmethod
    def generate(self, session, **kwargs) -> Generator[str, None, None]:
        """
        - generate text with punc
        """
        raise NotImplementedError("must be implemented in the child class")


class ITextProcessing(ABC):
    @abstractmethod
    def normalize(self, session, **kwargs) -> str:
        """
        - Text Normalize
        - Inverse Text Normalize
        """
        raise NotImplementedError("must be implemented in the child class")


class IHallucination(ABC):
    @abstractmethod
    def check(self, session) -> bool:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def filter(self, session) -> str:
        raise NotImplementedError("must be implemented in the child class")


class ILlmGenerator(ABC):
    @abstractmethod
    async def generate(self, session, **kwargs) -> AsyncGenerator[int, None]:
        """return token ids async generator"""
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self):
        """close generator"""
        raise NotImplementedError("must be implemented in the child class")


class ILlm(ABC):
    @abstractmethod
    def generate(self, session, **kwargs) -> Iterator[str | dict | np.ndarray]:
        """
        generate text or tokens with stream iterator
        - local llm cpu/gpu bind
        - api llm io bind
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def async_generate(
        self, session, **kwargs
    ) -> AsyncGenerator[str | dict | np.ndarray, None]:
        """
        async generate text or tokens with stream iterator
        - local llm cpu/gpu bind
        - api llm io bind
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def chat_completion(self, session, **kwargs) -> Iterator[str]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def async_chat_completion(self, session, **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def count_tokens(self, text: str | bytes):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def model_name(self):
        raise NotImplementedError("must be implemented in the child class")


class IFunction(ABC):
    @abstractmethod
    def execute(self, session, **args):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def get_tool_call(self):
        raise NotImplementedError("must be implemented in the child class")


class ITts(ABC):
    @abstractmethod
    def synthesize_sync(self, session) -> Generator[bytes, None, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def synthesize(self, session) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def get_stream_info(self) -> dict:
        """
        e.g.
        return {
            "format": PYAUDIO_PAINT16, # int
            "channels": 1, # int
            "rate": 16000, # int
            "sample_width": 2, # int
            "np_dtype": np.int16, # for numpy data type
        }
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_voice(self, voice: str, **kwargs):
        """
        Note:
        - just simple voice set, don't support set voice with user id
        - if set user voice ,need external store set api
        - u can impl set_user_voice method
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def get_voices(self) -> list:
        """
        Note:
        - just simple voice get, don't support get voice with user id
        - if get user voice ,need external store get api
        - u can impl get_user_voices method
        """
        raise NotImplementedError("must be implemented in the child class")


class IPlayer(ABC):
    @abstractmethod
    def open(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def start(self, session):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def play_audio(self, session):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def pause(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def resume(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def stop(self, session):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_out_stream(self, audio_stream):
        raise NotImplementedError("must be implemented in the child class")


class IConnector(ABC):
    @abstractmethod
    def send(self, data, at: str):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def recv(self, at: str, timeout: float | None = None):
        """
        just simple recv, block with timeout default no timeout
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplementedError("must be implemented in the child class")


class IBot(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def run(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def async_run(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def bot_config(self) -> dict:
        raise NotImplementedError("must be implemented in the child class")


class IVisionDetector(ABC):
    @abstractmethod
    def detect(self, session) -> bool:
        """
        cpu/gpu binding, need optimize to 10ms<
        input: session.ctx.state["detect_img"]
        detect object with confidence should be above threshold
        return bool
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def annotate(self, session) -> Generator[Any, None, None]:
        """
        cpu/gpu binding, need optimize to 10ms<
        input: session.ctx.state["detect_img"]
        annotate object from dectections,
        return Generator
        """
        raise NotImplementedError("must be implemented in the child class")


class IVisionOCR(ABC):
    @abstractmethod
    async def async_generate(
        self, session, **kwargs
    ) -> AsyncGenerator[str | dict | np.ndarray, None]:
        """
        input: session.ctx.state["ocr_img"]
        detect object and generate text
        return iterator a sentence (str)
        """
        raise NotImplementedError("must be implemented in the child class")


class IRoomManager(ABC):
    @abstractmethod
    async def create_room(self, room_name: str, exp_time_s: int):
        """
        create room by room name with expire time(s)
        if the room has been created, return
        if room_name is None or empty, create random name room
        return general room info
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def gen_token(self, room_name: str, exp_time_s: int) -> str:
        """
        generate a token to join room
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def get_room(self, room_name: str):
        """
        get general room info by room name
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def check_valid_room(self, room_name: str, token: str) -> bool:
        """
        check valid token and room status
        return bool is vaild room
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def close_session(self):
        """
        if api session(http or ws) is common long session, need to close
        """
        raise NotImplementedError("must be implemented in the child class")
