import logging
import math

from src.common.types import (
    VAD_CHECK_PER_FRAMES,
    VAD_CHECK_ALL_FRAMES,
    WebRTCVADArgs,
)
from src.common.session import Session
from .base import BaseVAD


class WebrtcVAD(BaseVAD):
    TAG = "webrtc_vad"

    def __init__(self, **args) -> None:
        import webrtcvad

        self.args = WebRTCVADArgs(**args)
        self.model = webrtcvad.Vad(self.args.aggressiveness)
        self.audio_buffer = None

    async def detect(self, session: Session):
        if self.args.check_frames_mode not in [VAD_CHECK_ALL_FRAMES, VAD_CHECK_PER_FRAMES]:
            return False

        # Number of audio frames per millisecond
        frame_length = int(self.args.sample_rate * self.args.frame_duration_ms / 1000)
        # just process channels:1 sample_width:2 audio
        num_frames = math.ceil(len(self.audio_buffer) / (2 * frame_length))
        speech_frames = 0
        logging.debug(
            f"{self.TAG} Speech detected audio_len:{len(self.audio_buffer)} frame_len:{frame_length} num_frames {num_frames}"
        )

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = self.audio_buffer[start_byte:end_byte]
            # print(len(frame))
            if self.model.is_speech(frame, self.args.sample_rate):
                speech_frames += 1
                if self.args.check_frames_mode == VAD_CHECK_PER_FRAMES:
                    logging.debug(
                        f"{self.TAG} Speech detected in frame {i + 1}" f" of {num_frames}"
                    )
                    # await save_audio_to_file(frame, "webrtc_vad_frame.wav")
                    return True
        if self.args.check_frames_mode == VAD_CHECK_ALL_FRAMES:
            if speech_frames == num_frames:
                logging.debug(
                    f"{self.TAG} Speech detected in {speech_frames} of " f"{num_frames} frames"
                )
            else:
                logging.debug(f"{self.TAG} Speech not detected in all {num_frames} frames")
            return speech_frames == num_frames

        logging.debug(f"{self.TAG} Speech not detected in any of {num_frames} frames")
        return False
