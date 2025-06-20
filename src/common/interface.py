from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncGenerator, Generator, List

import numpy as np


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
    async def detect(self, session) -> list[bytes | None]:
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


class IAsr(ABC):
    @abstractmethod
    def transcribe_stream_sync(self, session) -> Generator[str, None, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def transcribe_stream(self, session) -> AsyncGenerator[str, None]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    async def transcribe(self, session) -> dict:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def set_audio_data(self, audio_data):
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
    def chat_completion(self, session, **kwargs) -> Iterator[str]:
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
    def generate(self, session) -> Iterator[str]:
        """
        input: session.ctx.state["ocr_img"]
        detect object and generate text
        return iterator a sentence (str)
        """
        raise NotImplementedError("must be implemented in the child class")

    def stream_infer(self, session) -> Iterator[str]:
        """
        input: session.ctx.state["ocr_img"]
        detect object and generate text
        return iterator next token (str)
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
