import logging
import asyncio
import itertools
import time
from typing import List

from PIL import Image
from apipeline.frames.sys_frames import SystemFrame, CancelFrame, StartInterruptionFrame
from apipeline.frames.control_frames import StartFrame, ControlFrame, EndFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame, DataFrame, ImageRawFrame, TextFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.output_processor import OutputProcessor

from src.common.types import AudioCameraParams
from src.types.frames.control_frames import (
    BotSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from src.types.frames.data_frames import SpriteFrame, TransportMessageFrame, AnimationAudioRawFrame


class AudioCameraOutputProcessor(OutputProcessor):
    def __init__(
        self,
        params: AudioCameraParams,
        name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs,
    ):
        super().__init__(name=name, loop=loop, **kwargs)
        self._params = params

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = (
            int(self._params.audio_out_sample_rate / 100) * self._params.audio_out_channels * 2
        )
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks
        # Audio accumlation buffer for 16-bit samples to write out stream device
        self._audio_out_buff = bytearray()

        # Indicates if the bot is currently speaking. This is useful when we
        # have an interruption since all the queued messages will be thrown
        # away and we would lose the TTSStoppedFrame.
        self._bot_speaking = False

        # Audio queue and task
        self._audio_out_queue = None
        self._audio_out_task = None

        # Camera queue and task
        self._camera_out_queue = None
        self._camera_out_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Create media threads queues and task
        if self._params.camera_out_enabled:
            self._camera_out_queue = asyncio.Queue()
            self._camera_out_task = self.get_event_loop().create_task(
                self._camera_out_task_handler()
            )
        if self._params.audio_out_enabled:
            self._audio_out_queue = asyncio.Queue()
            self._audio_out_task = self.get_event_loop().create_task(self._audio_out_task_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        # Cancel and wait for the camera output task to finish.
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task
        if self._params.audio_out_enabled:
            self._audio_out_task.cancel()
            await self._audio_out_task

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._params.camera_out_enabled:
            self._camera_out_task.cancel()
            await self._camera_out_task
        if self._params.audio_out_enabled:
            self._audio_out_task.cancel()
            await self._audio_out_task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

    async def send_message(self, frame: TransportMessageFrame):
        pass

    async def send_text(self, frame: TextFrame):
        pass

    async def _handle_interruptions(self, frame: Frame):
        await super()._handle_interruptions(frame)
        if isinstance(frame, StartInterruptionFrame):
            # Let's send a bot stopped speaking if we have to.
            if self._bot_speaking:
                await self._bot_stopped_speaking()

    #
    # Sink frame
    #

    async def sink(self, frame: DataFrame):
        if isinstance(frame, TextFrame):  # text
            await self.send_text(frame)
        elif isinstance(frame, TransportMessageFrame):
            # transport mssage
            await self.send_message(frame)
        elif isinstance(frame, AnimationAudioRawFrame):  # audio + animation
            await self._handle_audio_animation(frame)
        elif isinstance(frame, AudioRawFrame):  # audio
            await self._handle_audio(frame)
        elif isinstance(frame, ImageRawFrame) or isinstance(frame, SpriteFrame):  # image
            await self._handle_image(frame)
        # !TODO: video
        else:
            # push other frame for downstream like FunctionCallResultFrame
            await self.queue_frame(frame)
        self._sink_event.set()

    async def sink_control_frame(self, frame: ControlFrame):
        if isinstance(frame, TTSStartedFrame):
            await self._bot_started_speaking()
            await self.queue_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self._bot_stopped_speaking()
            await self.queue_frame(frame)
        return await super().sink_control_frame(frame)

    async def _bot_started_speaking(self):
        logging.debug("Bot started speaking")
        self._bot_speaking = True
        await self.queue_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _bot_stopped_speaking(self):
        logging.debug("Bot stopped speaking")
        self._bot_speaking = False
        await self.queue_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    #
    # Audio out
    #

    async def send_audio(self, frame: AudioRawFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def write_animation_audio_frame(self, frame: AnimationAudioRawFrame):
        """
        - default call handel_audio to process audio raw frame
        - subclass can override this method to process animation audio raw frame
            - fastapi_websocket_output_processor to process animation audio raw frame
        """
        # logging.info(f"no subclass implement {frame=}")
        await self._handle_audio(frame)

    async def _handle_audio_animation(self, frame: AnimationAudioRawFrame):
        if not self._params.audio_out_enabled:
            return
        await self.write_animation_audio_frame(frame)

    async def _handle_audio(self, frame: AudioRawFrame):
        if not self._params.audio_out_enabled:
            return

        audio = frame.audio
        #!TODO: if tts processor is slow(network,best way to deloy all model on local machine(CPU,GPU)), we need to buffer the audio,e.g.buff 1 second :)
        # print(f"len audio:{len(audio)}, audio_chunk_size:{self._audio_chunk_size}")
        # self._audio_out_buff.extend(audio)
        # if len(audio) >= self._audio_chunk_size:
        # print( f"len audio_out_buff:{len(self._audio_out_buff)}, audio_chunk_size{self._audio_chunk_size}")
        for i in range(0, len(audio), self._audio_chunk_size):
            chunk = audio[i : i + self._audio_chunk_size]
            # if len(chunk) % 2 != 0: don't do that, need subclass to do(write_raw_audio_frames)
            #    chunk = chunk[:len(chunk) - 1]
            await self._audio_out_queue.put(chunk)
            await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
        # self._audio_out_buff.clear()

    async def write_raw_audio_frames(self, frames: bytes):
        """
        Subclass audio output stream transport to implement.
        """
        logging.info(f"no subclass implement, dump len(frames): {len(frames)}")

    async def _audio_out_task_handler(self):
        while True:
            try:
                chunk = await asyncio.wait_for(self._audio_out_queue.get(), timeout=1)
                await self.write_raw_audio_frames(chunk)
                self._audio_out_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logging.info(f"{self.name} _audio_out_task_handler cancelled")
                break
            except Exception as e:
                logging.exception(f"{self} error writing audio: {e}")
                if self.get_event_loop().is_closed():
                    logging.warning(f"{self.name} event loop is closed")
                    break

    #
    # Camera out
    #

    async def _handle_image(self, frame: ImageRawFrame | SpriteFrame):
        if not self._params.camera_out_enabled:
            return

        if self._params.camera_out_is_live:
            # NOTE:
            # out processor is last start to init camera out queue,
            # if pipeline other processor is start slow,
            # and push frame before out processor init, out processor will lost frame
            self._camera_out_queue and await self._camera_out_queue.put(frame)
        else:
            if isinstance(frame, ImageRawFrame):
                await self._set_camera_images([frame])
            if isinstance(frame, SpriteFrame):
                await self._set_camera_images(frame.images)

    async def send_image(self, frame: ImageRawFrame | SpriteFrame):
        await self.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _set_camera_images(self, images: List[ImageRawFrame]):
        # cycle images for display image frame like video
        self._camera_images = itertools.cycle(images)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        """
        Subclass camera output stream transport to implement.
        """
        logging.info(f"no subclass implement, dump frame: {frame}")

    async def _draw_image(self, frame: ImageRawFrame):
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)

        # @TODO: self._params need in the ctx
        if frame.size != desired_size:
            image = Image.frombytes(frame.mode, frame.size, frame.image)
            resized_image = image.resize(desired_size)
            # logging.warning(f"{frame} does not have the expected size {desired_size}, resizing")
            frame = ImageRawFrame(
                image=resized_image.tobytes(),
                size=resized_image.size,
                format=resized_image.format,
                mode=resized_image.mode,
            )

        await self.write_frame_to_camera(frame)

    async def _camera_out_is_live_handler(self):
        image = await self._camera_out_queue.get()

        # We get the start time as soon as we get the first image.
        if not self._camera_out_start_time:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0

        # Calculate how much time we need to wait before rendering next image.
        real_elapsed_time = time.time() - self._camera_out_start_time
        real_render_time = self._camera_out_frame_index * self._camera_out_frame_duration
        delay_time = self._camera_out_frame_duration + real_render_time - real_elapsed_time

        if abs(delay_time) > self._camera_out_frame_reset:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0
        elif delay_time > 0:
            await asyncio.sleep(delay_time)
            self._camera_out_frame_index += 1

        # Render image
        await self._draw_image(image)

        self._camera_out_queue.task_done()

    async def _camera_out_task_handler(self):
        self._camera_out_start_time = None
        self._camera_out_frame_index = 0
        self._camera_out_frame_duration = 1 / self._params.camera_out_framerate
        self._camera_out_frame_reset = self._camera_out_frame_duration * 5
        while True:
            try:
                if self._params.camera_out_is_live:
                    await self._camera_out_is_live_handler()
                elif self._camera_images:
                    image = next(self._camera_images)
                    await self._draw_image(image)
                    await asyncio.sleep(self._camera_out_frame_duration)
                else:
                    await asyncio.sleep(self._camera_out_frame_duration)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception(f"{self} error writing to camera: {e}")
