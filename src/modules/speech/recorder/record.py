import logging
import struct

import pyaudio

from src.common.session import Session
from src.common.interface import IRecorder
from src.common.types import AudioRecoderArgs, SILENCE_THRESHOLD, SILENT_CHUNKS


class RMSRecorder(IRecorder):
    def __init__(self, args: AudioRecoderArgs) -> None:
        self.args = args
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=args.format_, channels=args.channels,
            rate=args.rate, input=True,
            input_device_index=args.input_device_index,
            s_per_buffer=args.frames_per_buffer)

    def compute_rms(self, data):
        # Assuming data is in 16-bit samples
        format = "<{}h".format(len(data) // 2)
        ints = struct.unpack(format, data)

        # Calculate RMS
        sum_squares = sum(i ** 2 for i in ints)
        rms = (sum_squares / len(ints)) ** 0.5
        return rms

    def record_audio(self, session: Session):
        silent_chunks = 0
        audio_started = False
        frames = []

        logging.debug("start recording")
        while True:
            data = self.stream.read(self.args.frames_per_buffer)
            frames.append(data)
            rms = self.compute_rms(data)
            if audio_started:
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks > SILENT_CHUNKS:
                        break
                else:
                    silent_chunks = 0
            elif rms >= SILENCE_THRESHOLD:
                audio_started = True
        return frames

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
