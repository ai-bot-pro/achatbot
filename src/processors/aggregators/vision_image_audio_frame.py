from apipeline.frames import *
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames.data_frames import VisionImageVoiceRawFrame


class VisionImageAudioFrameAggregator(FrameProcessor):
    def __init__(self, pass_audio: bool = False):
        super().__init__()
        self._pass_audio = pass_audio
        self._audio_frame = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self._audio_frame = frame
            if self._pass_audio:
                await self.push_frame(frame, direction)
        elif isinstance(frame, VisionImageVoiceRawFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, ImageRawFrame):
            if self._audio_frame:
                frame = VisionImageVoiceRawFrame(text=None, audio=self._audio_frame, images=[frame])
                await self.push_frame(frame)
                self._audio_frame = None
        else:
            await self.push_frame(frame, direction)
