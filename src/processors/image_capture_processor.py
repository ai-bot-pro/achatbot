import logging

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import Frame

from src.common.audio_stream.helper import RingBuffer
from src.types.frames.data_frames import UserImageRawFrame


class ImageCaptureProcessor(FrameProcessor):
    def __init__(self, capture_cn: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._capture_imgs = RingBuffer(capture_cn)

    @property
    def capture_imgs(self):
        return self._capture_imgs

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame):
            self._capture_imgs.append(frame)
        await self.push_frame(frame, direction)
