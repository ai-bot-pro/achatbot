
from abc import ABC, abstractmethod
from typing import Iterator, AsyncGenerator, Generator


class IModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")


class IAudioStream(ABC):
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    @abstractmethod
    def open_stream(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def start_stream(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def stop_stream(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close_stream(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def is_stream_active(self) -> bool:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def write_stream(self, data):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def get_stream_info(self):
        """
        return AudioStreamInfo
        """
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def read_stream(self, num_frames) -> bytes:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close(self) -> None:
        raise NotImplemented("must be implemented in the child class")


class IRecorder(ABC):
    @abstractmethod
    def open(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def record_audio(self, session) -> list[bytes]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def record_audio_generator(self, session) -> AsyncGenerator[bytes, None]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def set_in_stream(self, audio_stream):
        raise NotImplemented("must be implemented in the child class")


class IBuffering(ABC):
    @abstractmethod
    def process_audio(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def is_voice_active(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def insert(self, audio_data):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def clear(self):
        raise NotImplemented("must be implemented in the child class")


class IDetector(ABC):
    @abstractmethod
    async def detect(self, session) -> list[bytes | None]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def set_audio_data(self, audio_data):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def get_sample_info(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplemented("must be implemented in the child class")


class IVADAnalyzer(ABC):
    @abstractmethod
    def voice_confidence(self, buffer) -> float:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def analyze_audio(self, buffer):
        raise NotImplemented("must be implemented in the child class")


class IAsr(ABC):
    @abstractmethod
    def transcribe_stream_sync(self, session) -> Generator[str, None, None]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def transcribe_stream(self, session) -> AsyncGenerator[str, None]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def transcribe(self, session) -> dict:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def set_audio_data(self, audio_data):
        raise NotImplemented("must be implemented in the child class")


class IHallucination(ABC):
    @abstractmethod
    def check(self, session) -> bool:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def filter(self, session) -> str:
        raise NotImplemented("must be implemented in the child class")


class ILlm(ABC):
    @abstractmethod
    def generate(self, session) -> Iterator[str]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def chat_completion(self, session) -> Iterator[str]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def count_tokens(self, text: str | bytes):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def model_name(self):
        raise NotImplemented("must be implemented in the child class")


class IFunction(ABC):
    @abstractmethod
    def execute(self, session, **args):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def get_tool_call(self):
        raise NotImplemented("must be implemented in the child class")


class ITts(ABC):
    @abstractmethod
    def synthesize_sync(self, session) -> Generator[bytes, None, None]:
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def synthesize(self, session) -> AsyncGenerator[bytes, None]:
        raise NotImplemented("must be implemented in the child class")

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
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def set_voice(self, voice: str):
        raise NotImplemented("must be implemented in the child class")


class IPlayer(ABC):
    @abstractmethod
    def open(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def start(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def play_audio(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def pause(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def resume(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def stop(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def set_out_stream(self, audio_stream):
        raise NotImplemented("must be implemented in the child class")


class IConnector(ABC):
    @abstractmethod
    def send(self, data, at: str):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def recv(self, at: str):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def close(self):
        raise NotImplemented("must be implemented in the child class")


class IBot(ABC):
    @abstractmethod
    def run(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    async def arun(self):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def bot_config(self) -> dict:
        raise NotImplemented("must be implemented in the child class")
