import numpy as np
import torch

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.utils.helper import exp_smoothing, calculate_audio_volume
from src.common.types import VADAnalyzerArgs, VADState
from src.types.frames.data_frames import VADStateAudioRawFrame
from src.common.interface import IVADAnalyzer
from src.common.factory import EngineClass


class BaseVADAnalyzer(IVADAnalyzer, EngineClass):
    def __init__(self, **args):
        self._args = VADAnalyzerArgs(**args)
        self._vad_frames = self.num_frames_required()
        # sample width: 2
        self._vad_frames_num_bytes = (
            self._vad_frames * self._args.num_channels * self._args.sample_width
        )
        self._sample_num_bytes = (
            self._args.sample_rate * self._args.num_channels * self._args.sample_width
        )

        vad_frames_per_sec = self._vad_frames / self._args.sample_rate

        self._vad_start_frames = round(self._args.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._args.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

        # active speech segment
        self._speech_id = 0
        self._accumulate_speech_bytes_len = 0
        self._is_final = False
        self._start_at_s = 0.0
        self._cur_at_s = 0.0
        self._end_at_s = 0.0
        self.reset()

    def reset(self):
        pass

    @property
    def sample_rate(self):
        return self._args.sample_rate

    def num_frames_required(self) -> int:
        return int(self.sample_rate / 100.0)  # 10 ms frame size

    def process_audio_buffer(self, buffer):
        audio_chunk = buffer
        if isinstance(buffer, (bytes, bytearray)):
            audio_chunk = bytes2NpArrayWith16(buffer)
        if not isinstance(audio_chunk, np.ndarray):
            raise Exception(f"buffer type error, expect bytes or np.ndarray, got {type(buffer)}")

        assert len(audio_chunk) == self.num_frames_required()

        return audio_chunk

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = calculate_audio_volume(audio, self._args.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    def analyze_audio(self, buffer) -> VADStateAudioRawFrame:
        """
        Starting -> {Start} -> Speaking -> Stopping -> {End} -> Quiet(no active) -> Starting ....
        """
        assert len(buffer) == self._vad_frames_num_bytes

        self._cur_at_s = round(self._accumulate_speech_bytes_len / self._sample_num_bytes, 3)
        self._accumulate_speech_bytes_len += len(buffer)

        confidence = self.voice_confidence(buffer)

        volume = self._get_smoothed_volume(buffer)
        self._prev_volume = volume

        # @weedge maybe add praat energy threshold
        speaking = confidence >= self._args.confidence and volume >= self._args.min_volume

        if speaking:
            match self._vad_state:
                case VADState.QUIET:
                    self._vad_state = VADState.STARTING
                    self._vad_starting_count = 1
                case VADState.STARTING:
                    self._vad_starting_count += 1
                case VADState.STOPPING:
                    self._vad_state = VADState.SPEAKING
                    self._vad_stopping_count = 0
        else:
            match self._vad_state:
                case VADState.STARTING:
                    self._vad_state = VADState.QUIET
                    self._vad_starting_count = 0
                    self.reset()  # reset model stats to release memory
                case VADState.SPEAKING:
                    self._vad_state = VADState.STOPPING
                    self._vad_stopping_count = 1
                case VADState.STOPPING:
                    self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):  # Start
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

            self._speech_id += 1  # new active speech id
            self._is_final = False
            self._start_at_s = round(
                (self._accumulate_speech_bytes_len - len(buffer)) / self._sample_num_bytes, 3
            )
            self._end_at_s = 0.0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):  # End
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

            self._is_final = True
            self._end_at_s = round(self._accumulate_speech_bytes_len / self._sample_num_bytes, 3)

        # Starting / Speaking / Stopping / Quiet(no active)
        return VADStateAudioRawFrame(
            audio=buffer,
            sample_rate=self._args.sample_rate,
            num_channels=self._args.num_channels,
            sample_width=self._args.sample_width,
            state=self._vad_state,
            speech_id=self._speech_id,
            is_final=self._is_final,
            start_at_s=self._start_at_s,
            cur_at_s=self._cur_at_s,
            end_at_s=self._end_at_s,
        )
