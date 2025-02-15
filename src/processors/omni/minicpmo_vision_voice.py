import logging
from typing import AsyncGenerator

from PIL import Image
from apipeline.frames import *
import librosa
import numpy as np

from src.common.session import Session
from src.core.llm.transformers.manual_vision_voice_minicpmo import (
    TransformersManualVisionVoiceMiniCPMO,
)
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.processors.omni.base import VisionVoiceProcessorBase
from src.types.frames.data_frames import PathAudioRawFrame, VisionImageVoiceRawFrame


class MiniCPMoVisionVoiceProcessor(VisionVoiceProcessorBase):
    """ """

    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(
            llm=TransformersManualVisionVoiceMiniCPMO(**kwargs), session=session, **kwargs
        )

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": TransformersManualVisionVoiceMiniCPMO.RATE,
            "channels": 1,
        }

    async def run(self, frame: VisionImageVoiceRawFrame) -> AsyncGenerator[Frame, None]:
        if not self._llm:
            logging.error(f"{self} error: llm not available")
            yield ErrorFrame("llm not available")
            return

        logging.debug(f"VisionImageVoiceRawFrame: {frame}")
        self._session.ctx.state["prompt"] = []

        # frame.text and self._session.ctx.state["prompt"].append(frame.text)

        if frame.images:
            for image_frame in frame.images:
                image = Image.frombytes(image_frame.mode, image_frame.size, image_frame.image)
                self._session.ctx.state["prompt"].append(image)

        if frame.audio and frame.audio.audio:
            if isinstance(frame.audio, PathAudioRawFrame):
                audio_nparr, _ = librosa.load(frame.audio.path, sr=16000, mono=True)
            else:
                audio_nparr = bytes2NpArrayWith16(frame.audio.audio)
            self._session.ctx.state["prompt"].append(audio_nparr)

        async for item in self.gen():
            yield item
