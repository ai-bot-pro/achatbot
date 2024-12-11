import logging

from .base import BaseVADAnalyzer
from src.common.types import DAILY_WEBRTC_VAD_RESET_PERIOD_MS


class DailyWebRTCVADAnalyzer(BaseVADAnalyzer):
    TAG = "daily_webrtc_vad_analyzer"

    def __init__(self, **args):
        super().__init__(**args)
        from daily import Daily

        self._webrtc_vad = Daily.create_native_vad(
            reset_period_ms=DAILY_WEBRTC_VAD_RESET_PERIOD_MS,
            sample_rate=self._args.sample_rate,
            channels=self._args.num_channels,
        )
        logging.debug("Loaded native WebRTC VAD")

    def voice_confidence(self, buffer) -> float:
        confidence = 0
        if len(buffer) > 0:
            confidence = self._webrtc_vad.analyze_frames(buffer)
        return confidence
