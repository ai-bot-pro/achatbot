import logging
import time


from .base import BaseVADAnalyzer
from src.modules.speech.detector.silero_vad import SileroVAD
from src.common.types import (
    SILERO_MODEL_RESET_STATES_TIME,
    SileroVADArgs,
    VADAnalyzerArgs,
    SileroVADAnalyzerArgs,
)


# NOTE: MRO: Method Resolution Order to init, and member name
class SileroVADAnalyzer(BaseVADAnalyzer):
    TAG = "silero_vad_analyzer"

    def __init__(self, **args):
        self.args = SileroVADAnalyzerArgs(**args)
        self._vad = SileroVAD(
            **SileroVADArgs(
                sample_rate=self.args.sample_rate,
                repo_or_dir=self.args.repo_or_dir,
                model=self.args.model,
                source=self.args.source,
                force_reload=self.args.force_reload,
                verbose=self.args.verbose,
                onnx=self.args.onnx,
                silero_sensitivity=self.args.silero_sensitivity,
                is_pad_tensor=self.args.is_pad_tensor,
                check_frames_mode=self.args.check_frames_mode,
            ).__dict__
        )
        super().__init__(
            **VADAnalyzerArgs(
                sample_rate=self.args.sample_rate,
                num_channels=self.args.num_channels,
                confidence=self.args.confidence,
                start_secs=self.args.start_secs,
                stop_secs=self.args.stop_secs,
                min_volume=self.args.min_volume,
            ).__dict__
        )
        self._last_reset_time = 0

    @property
    def vad(self):
        return self._vad

    def num_frames_required(self) -> int:
        return self._vad.get_sample_info()[1]

    def voice_confidence(self, buffer) -> float:
        try:
            audio_chunk = self._vad.process_audio_buffer(buffer)
            vad_prob = self._vad.model(audio_chunk, self.args.sample_rate).item()
            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= SILERO_MODEL_RESET_STATES_TIME:
                self._vad.model.reset_states()
                self._last_reset_time = curr_time

            return vad_prob
        except Exception as ex:
            # This comes from an empty audio array
            logging.exception(f"Error analyzing audio with Silero VAD: {ex}")
            return 0
