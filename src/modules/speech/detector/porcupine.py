import collections
import itertools
import logging
import struct
import time

import pvporcupine

from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import PorcupineDetectorArgs


class PorcupineDetector:
    def __init__(self, args: PorcupineDetectorArgs) -> None:
        self.args = args
        if self.args.wake_words:
            self.wake_words_list = [
                word.strip() for word in self.args.wake_words.lower().split(',')
            ]
            sensitivity_list = [
                float(self.args.wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            self.porcupine = pvporcupine.create(
                keywords=self.wake_words_list,
                sensitivities=sensitivity_list
            )
            self.buffer_size = self.porcupine.frame_length
            self.sample_rate = self.porcupine.sample_rate


class PorcupineWakeWordDetector(IDetector, PorcupineDetector):
    def __init__(self, args: PorcupineDetectorArgs) -> None:
        super.__init__(args)
        logging.debug(
            "Porcupine wake word detection engine initialized successfully")

    async def detect(self, session: Session):
        if len(self.args.wake_words) == 0:
            return
        pcm = struct.unpack_from(
            "h" * self.buffer_size,
            session.args.wake_word_buffer
        )
        wakeword_index = self.porcupine.process(pcm)

        # If a wake word is detected
        if wakeword_index >= 0:
            # Removing the wake word from the recording
            samples_for_0_1_sec = int(self.sample_rate * 0.1)
            start_index = max(0, len(self.audio_buffer) - samples_for_0_1_sec)
            temp_samples = collections.deque(
                itertools.islice( self.audio_buffer, start_index, None)
            )
            self.audio_buffer.clear()
            self.audio_buffer.extend(temp_samples)

            self.wake_word_detect_time = time.time()
            self.wakeword_detected = True
            if self.args.on_wakeword_detected:
                self.on_wakeword_detected()
