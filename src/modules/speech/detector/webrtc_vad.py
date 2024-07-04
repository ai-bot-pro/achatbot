from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import WebRTCVADArgs, RATE
from src.common.factory import EngineClass


class WebrtcVAD(EngineClass, IDetector):
    TAG = "webrtc_vad"

    def __init__(self, **args: WebRTCVADArgs) -> None:
        import webrtcvad
        self.args = WebRTCVADArgs(**args)
        self.vad = webrtcvad.Vad(self.args.aggressiveness)

    async def detect(self, session: Session):
        return self.vad.is_speech()

    def get_sample_info(self):
        return RATE, int(len(self.audio_buffer) / 2)

    def set_audio_data(self, audio_data):
        self.audio_buffer = audio_data

    def close(self):
        pass
