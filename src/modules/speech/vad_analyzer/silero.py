import logging
import time


from .base import BaseVADAnalyzer
from src.modules.speech.detector.silero_vad import SileroVAD
from src.common.types import SILERO_MODEL_RESET_STATES_TIME


# NOTE: MRO: Method Resolution Order to init, and member name
class SileroVADAnalyzer(BaseVADAnalyzer, SileroVAD):
    TAG = "silero_vad_analyzer"

    def __init__(self, **args):
        super().__init__(**args)
        SileroVAD(**args)
        self._last_reset_time = 0

    def num_frames_required(self) -> int:
        return self.get_sample_info()[1]

    def voice_confidence(self, buffer) -> float:
        try:
            audio_chunk = self.process_audio_buffer(buffer)
            vad_prob = self.model(audio_chunk, self.args.sample_rate).item()
            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= SILERO_MODEL_RESET_STATES_TIME:
                self.model.reset_states()
                self._last_reset_time = curr_time

            return vad_prob
        except Exception as ex:
            # This comes from an empty audio array
            logging.exception(f"Error analyzing audio with Silero VAD: {ex}")
            return 0
