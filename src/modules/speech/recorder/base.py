from src.common.audio_stream import AudioStream, AudioStreamArgs
from src.common.factory import EngineClass
from src.common.interface import IRecorder
from src.common.types import AudioRecoderArgs


class PyAudioRecorder(EngineClass, IRecorder):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioRecoderArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = AudioRecoderArgs(**args)
        self.audio = AudioStream(AudioStreamArgs(
            format=self.args.format,
            channels=self.args.channels,
            rate=self.args.rate,
            input_device_index=self.args.input_device_index,
            input=True,
            frames_per_buffer=self.args.frames_per_buffer,
        ))
        self.audio.open_stream()

    def record_audio(self, session) -> list[bytes]:
        return []

    def close(self):
        self.audio.close()
