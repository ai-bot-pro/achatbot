import logging

from src.common.types import (
    VAD_CHECK_PER_FRAMES,
    VAD_CHECK_ALL_FRAMES,
    WebRTCVADArgs,
    SileroVADArgs,
    WebRTCSileroVADArgs,
)
from src.common.session import Session
from .base import BaseVAD
from .webrtc_vad import WebrtcVAD
from .silero_vad import SileroVAD


class WebrtcSileroVAD(BaseVAD):
    TAG = "webrtc_silero_vad"

    def __init__(self, **args) -> None:
        self.args = WebRTCSileroVADArgs(**args)
        self.webrtc_vad = WebrtcVAD(
            **WebRTCVADArgs(
                aggressiveness=self.args.aggressiveness,
                sample_rate=self.args.sample_rate,
                check_frames_mode=self.args.check_frames_mode,
                frame_duration_ms=self.args.frame_duration_ms,
            ).__dict__
        )
        self.silero_vad = SileroVAD(
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

    async def detect(self, session: Session):
        if self.args.check_frames_mode not in [VAD_CHECK_ALL_FRAMES, VAD_CHECK_PER_FRAMES]:
            return False

        self.webrtc_vad.set_audio_data(self.audio_buffer)
        is_webrtc_speech = await self.webrtc_vad.detect(session)
        if is_webrtc_speech is False:  # fast path
            return False
        logging.debug("webRTC has detected a possible speech")

        # check noise with silero vad
        self.silero_vad.set_audio_data(self.audio_buffer)
        is_silero_speech = await self.silero_vad.detect(session)
        if is_silero_speech is False:
            logging.debug("silero VAD has detected a noise")
            return False
        logging.debug("silero VAD has detected a possible speech")
        return True
