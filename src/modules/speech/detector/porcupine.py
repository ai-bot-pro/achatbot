import collections
import itertools
import logging
import struct
import time
import os


from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import PorcupineDetectorArgs
from src.common.factory import EngineClass


class PorcupineDetector(EngineClass):
    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**PorcupineDetectorArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        import pvporcupine

        self.args = PorcupineDetectorArgs(**args)
        if self.args.wake_words:
            self.wake_words_list = [
                word.strip() for word in self.args.wake_words.lower().split(",")
            ]
            sensitivity_list = [
                float(self.args.wake_words_sensitivity) for _ in range(len(self.wake_words_list))
            ]

            self.porcupine: pvporcupine.Porcupine = pvporcupine.create(
                access_key=os.getenv("PORCUPINE_ACCESS_KEY", ""),
                library_path=self.args.library_path,
                model_path=self.args.model_path,
                keyword_paths=self.args.keyword_paths,
                keywords=self.wake_words_list,
                sensitivities=sensitivity_list,
            )

            logging.debug(
                f"sample_rate: {self.porcupine.sample_rate},frame_length: {self.porcupine.frame_length}"
            )

    def get_sample_info(self):
        return self.porcupine.sample_rate, self.porcupine.frame_length

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, collections.deque):
            self.audio_buffer = audio_data

    def close(self):
        self.porcupine and self.porcupine.delete()


class PorcupineWakeWordDetector(PorcupineDetector, IDetector):
    TAG = "porcupine_wakeword"

    async def detect(self, session: Session):
        if len(self.args.wake_words) == 0:
            return
        logging.debug(
            f"{self.TAG} detect porcupine frame_len:{self.porcupine.frame_length}, read_audio_frames_len:{len(session.ctx.read_audio_frames)}"
        )
        pcm = struct.unpack_from(
            "h" * self.porcupine.frame_length,
            session.ctx.read_audio_frames,
        )
        wakeword_index = self.porcupine.process(pcm)

        # If a wake word is detected
        if wakeword_index >= 0:
            logging.info(
                f"index {wakeword_index} {self.wake_words_list[wakeword_index]} hotword detected, audio_buffer length {len(self.audio_buffer)}"
            )
            session.ctx.state["bot_name"] = self.wake_words_list[wakeword_index]
            # Removing the wake word from the recording
            samples_for_0_1_sec = int(self.porcupine.sample_rate * 0.1)
            start_index = max(0, len(self.audio_buffer) - samples_for_0_1_sec)
            temp_samples = collections.deque(itertools.islice(self.audio_buffer, start_index, None))
            self.audio_buffer.clear()
            self.audio_buffer.extend(temp_samples)

            if self.args.on_wakeword_detected:
                self.args.on_wakeword_detected(session, self.audio_buffer)
            return True
        return False
