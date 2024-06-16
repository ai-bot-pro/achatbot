
from src.common.audio_stream import AudioStream, AudioStreamArgs
from src.common.factory import EngineClass
from src.common.interface import IPlayer
from src.common.session import Session
from src.common.types import AudioPlayerArgs


class PyAudioPlayer(EngineClass):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioPlayerArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = AudioPlayerArgs(**args)
        self.audio = AudioStream(AudioStreamArgs(
            format_=self.args.format_,
            channels=self.args.channels,
            rate=self.args.rate,
            output_device_index=self.args.output_device_index,
            output=True,
            frames_per_buffer=self.args.chunk_size,
        ))
        self.audio.open_stream()

    def close(self):
        self.audio.close()


class StreamPlayer(PyAudioPlayer, IPlayer):
    TAG = "stream_player"

    def play_audio(self, session: Session):
        if session.ctx.state.get("tts_chunk") is None:
            return
        chunk = session.ctx.state["tts_chunk"]
        for i in range(0, len(chunk), self.args.chunk_size):
            sub_chunk = chunk[i:i + self.args.chunk_size]
            self.audio.stream.write(sub_chunk)
