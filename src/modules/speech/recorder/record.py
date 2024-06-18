import logging
import struct


from src.common.audio_stream import AudioStream, AudioStreamArgs
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.interface import IRecorder
from src.common.types import AudioRecoderArgs, SILENCE_THRESHOLD, SILENT_CHUNKS, RATE


class PyAudioRecorder(EngineClass):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**AudioRecoderArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = AudioRecoderArgs(**args)
        self.audio = AudioStream(AudioStreamArgs(
            format_=self.args.format_,
            channels=self.args.channels,
            rate=self.args.rate,
            input_device_index=self.args.input_device_index,
            input=True,
            frames_per_buffer=self.args.frames_per_buffer,
        ))
        self.audio.open_stream()

    def close(self):
        self.audio.close()


class RMSRecorder(PyAudioRecorder, IRecorder):
    TAG = "rms_recorder"

    def compute_rms(self, data):
        # Assuming data is in 16-bit samples
        format = "<{}h".format(len(data) // 2)
        ints = struct.unpack(format, data)

        # Calculate RMS
        sum_squares = sum(i ** 2 for i in ints)
        rms = (sum_squares / len(ints)) ** 0.5
        return rms

    def record_audio(self, session: Session) -> list[bytes]:
        silent_chunks = 0
        audio_started = False
        frames = []

        if self.args.rate != RATE:
            logging.warning(
                "Sampling rate of the audio just support 16000Hz at now"
            )
            return frames

        self.audio.stream.start_stream()
        logging.debug("start recording")
        while True:
            data = self.audio.stream.read(self.args.frames_per_buffer)
            rms = self.compute_rms(data)
            if audio_started:
                frames.append(data)
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks > SILENT_CHUNKS:
                        break
                else:
                    silent_chunks = 0
            elif rms >= SILENCE_THRESHOLD:
                frames.append(data)
                audio_started = True

        self.audio.stream.stop_stream()
        logging.debug("end recording")

        return frames


class VADRecorder(PyAudioRecorder, IRecorder):
    pass
