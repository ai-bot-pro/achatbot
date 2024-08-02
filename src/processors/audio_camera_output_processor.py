import logging
import asyncio
import itertools
from typing import List

from PIL import Image
from apipeline.frames.sys_frames import SystemFrame, CancelFrame
from apipeline.frames.control_frames import StartFrame, ControlFrame, EndFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame, DataFrame, ImageRawFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.output_processor import OutputProcessor

from src.common.types import AudioCameraParams
from src.types.frames.control_frames import BotSpeakingFrame, TTSStartedFrame, TTSStoppedFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from src.types.frames.data_frames import SpriteFrame, TransportMessageFrame


class AudioCameraOutputProcessor(OutputProcessor):
    def __init__(
            self,
            params: AudioCameraParams,
            name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None,
            **kwargs):
        super().__init__(name=name, loop=loop, **kwargs)
        self._params = params

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # We will write 20ms audio at a time. If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._params.audio_out_sample_rate / 100) * \
            self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * 2

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Create media threads queues and task
        if self._params.camera_out_enabled:
            self._camera_out_queue = asyncio.Queue()
            self._camera_out_task = self.get_event_loop().create_task(self._camera_out_task_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # Cancel and wait for the camera output task to finish.
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

    async def send_message(self, frame: TransportMessageFrame):
        pass

    #
    # Sink frame
    #
    async def sink(self, frame: DataFrame):
        if isinstance(frame, TransportMessageFrame):
            await self.send_message(frame)
        elif isinstance(frame, AudioRawFrame) and self._params.audio_out_enabled:
            await self._handle_audio(frame)
        elif isinstance(frame, ImageRawFrame) and self._params.camera_out_enabled:
            await self._set_camera_image(frame)
        elif isinstance(frame, SpriteFrame) and self._params.camera_out_enabled:
            await self._set_camera_images(frame.images)
        self._sink_event.set()

    async def sink_control_frame(self, frame: ControlFrame):
        if isinstance(frame, TTSStartedFrame):
            await self.queue_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.queue_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self.queue_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self.queue_frame(frame)
        return await super().sink_control_frame(frame)

    #
    # Audio out
    #

    async def send_audio(self, frame: AudioRawFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _handle_audio(self, frame: AudioRawFrame):
        audio = frame.audio
        # print(f"len audio:{len(audio)}, audio_chunk_size{self._audio_chunk_size}")
        for i in range(0, len(audio), self._audio_chunk_size):
            chunk = audio[i: i + self._audio_chunk_size]
            await self.write_raw_audio_frames(chunk)
            await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    #
    # Camera out
    #

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _set_camera_image(self, image: ImageRawFrame):
        if self._params.camera_out_is_live:
            await self._camera_out_queue.put(image)
        else:
            self._camera_images = itertools.cycle([image])

    async def _set_camera_images(self, images: List[ImageRawFrame]):
        self._camera_images = itertools.cycle(images)

    async def _camera_out_task_handler(self):
        while True:
            try:
                if self._params.camera_out_is_live:
                    image = await self._camera_out_queue.get()
                    await self._draw_image(image)
                    self._camera_out_queue.task_done()
                elif self._camera_images:
                    image = next(self._camera_images)
                    await self._draw_image(image)
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
                else:
                    await asyncio.sleep(1.0 / self._params.camera_out_framerate)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception(f"{self} error writing to camera: {e}")

    async def _draw_image(self, frame: ImageRawFrame):
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)

        if frame.size != desired_size:
            image = Image.frombytes(frame.format, frame.size, frame.image)
            resized_image = image.resize(desired_size)
            logging.warning(
                f"{frame} does not have the expected size {desired_size}, resizing")
            frame = ImageRawFrame(resized_image.tobytes(), resized_image.size, resized_image.format)

        await self.write_frame_to_camera(frame)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        pass
