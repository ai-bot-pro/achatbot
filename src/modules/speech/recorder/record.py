import logging
import struct
import asyncio
import time


from src.common.audio_stream import AudioStream, AudioStreamArgs, RingBuffer
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

    def __init__(self, **args) -> None:
        super().__init__(**args)
        if self.args.rate != RATE:
            raise Exception(
                f"Sampling rate of the audio just support 16000Hz at now")

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
        silence_timeout = 0
        if "silence_timeout_s" in session.ctx.state:
            logging.info(
                f"rms recording with silence_timeout {session.ctx.state['silence_timeout_s']} s")
            silence_timeout = int(session.ctx.state['silence_timeout_s'])

        self.audio.stream.start_stream()
        logging.debug("start rms recording")
        start_time = time.time()
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
            else:
                if silence_timeout > 0 \
                        and time.time() - start_time > silence_timeout:
                    logging.warning(f"rms recording silence timeout")
                    break

        self.audio.stream.stop_stream()
        logging.debug("end rms recording")

        return frames


class WakeWordsRMSRecorder(RMSRecorder, IRecorder):
    TAG = "wakeword_rms_recorder"

    def __init__(self, **args) -> None:
        super().__init__(**args)

    def record_audio(self, session: Session) -> list[bytes]:
        if session.ctx.waker is None:
            logging.warning(
                f"WakeWordsRMSRecorder no waker instance in session ctx, use RMSRecorder")
            return super().record_audio(session)

        sample_rate, frame_length = session.ctx.waker.get_sample_info()
        self.sample_rate, self.frame_length = sample_rate, frame_length

        # ring buffer
        pre_recording_buffer_duration = 10.0
        maxlen = int((sample_rate // frame_length) *
                     pre_recording_buffer_duration)
        self.audio_buffer = RingBuffer(maxlen)
        logging.debug(f"audio ring buffer maxlen: {maxlen}")

        self.audio.stream.start_stream()
        logging.debug("start wake words detector rms recording")

        while True:
            data = self.audio.stream.read(self.frame_length)
            session.ctx.read_audio_frames = data
            session.ctx.waker.set_audio_data(self.audio_buffer.get_buf())
            res = asyncio.run(
                session.ctx.waker.detect(session))
            if res is True:
                break
            self.audio_buffer.extend(data)

        self.audio.stream.stop_stream()
        logging.debug("end wake words detector rms recording")

        if self.args.silence_timeout_s > 0:
            session.ctx.state["silence_timeout_s"] = self.args.silence_timeout_s
        return super().record_audio(session)


class VADRecorder(PyAudioRecorder, IRecorder):
    TAG = ""

    def record_audio(self, session: Session) -> list[bytes]:
        pass


class WakeWordsVADRecorder(PyAudioRecorder, IRecorder):
    TAG = ""

    def record_audio(self, session: Session) -> list[bytes]:
        pass
