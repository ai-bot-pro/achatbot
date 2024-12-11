import logging
import math
import struct
import time


from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.pipeline.pipeline import FrameDirection
from apipeline.processors.frame_processor import FrameProcessor


class AudioVolumeTimeProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_transition_ts = 0
        self._prev_volume = -80
        self._speech_volume_threshold = -50

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            volume = self.calculate_volume(frame)
            logging.debug(f"Audio volume: {volume:.2f} dB")
            if (
                volume >= self._speech_volume_threshold
                and self._prev_volume < self._speech_volume_threshold
            ):
                logging.debug("transition above speech volume threshold")
                self.last_transition_ts = time.time()
            elif (
                volume < self._speech_volume_threshold
                and self._prev_volume >= self._speech_volume_threshold
            ):
                logging.debug("transition below non-speech volume threshold")
                self.last_transition_ts = time.time()
            self._prev_volume = volume

        await self.push_frame(frame, direction)

    def calculate_volume(self, frame: AudioRawFrame) -> float:
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # Unpack audio data into 16-bit integers
        fmt = f"{len(frame.audio) // 2}h"
        audio_samples = struct.unpack(fmt, frame.audio)

        # Calculate RMS
        sum_squares = sum(sample**2 for sample in audio_samples)
        rms = math.sqrt(sum_squares / len(audio_samples))

        # Convert RMS to decibels (dB)
        # Reference: maximum value for 16-bit audio is 32767
        if rms > 0:
            db = 20 * math.log10(rms / 32767)
        else:
            db = -96  # Minimum value (almost silent)

        return db
