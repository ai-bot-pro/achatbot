import logging


from src.common.utils.audio_utils import convertSampleRateTo16khz
from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import RATE, CHUNK
from src.common.factory import EngineClass


class BaseVAD(EngineClass, IDetector):
    async def detect(self, session: Session):
        pass

    def get_sample_info(self):
        return RATE, CHUNK

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.audio_buffer = audio_data
            if hasattr(self.args, "sample_rate") is False:
                return
            if self.args.sample_rate != 16000:
                audio_data_16 = convertSampleRateTo16khz(audio_data, self.args.sample_rate)
                logging.debug(
                    f"rate {self.args.sample_rate} convertSampleRateTo16khz len(audio_data):{len(audio_data)} -> len(audio_data_16):{len(audio_data_16)}"
                )
                self.audio_buffer = audio_data_16

    def close(self):
        pass
