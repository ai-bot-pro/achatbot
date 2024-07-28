import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.common.interface import IVADAnalyzer
from src.common.types import AudioVADParams, VADState
from apipeline.frames.base import Frame
from apipeline.frames.sys_frames import StartFrame, StartInterruptionFrame, StopInterruptionFrame
from apipeline.frames.data_frames import AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.input_processor import InputProcessor
from types.frames.control_frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame


class AudioVADInputProcessor(InputProcessor):
    def __init__(
            self,
            params: AudioVADParams,
            name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None,
            **kwargs):
        super().__init__(name=name, loop=loop, **kwargs)
        self._params = params
        self._executor = ThreadPoolExecutor(max_workers=3)

    @property
    def vad_analyzer(self) -> IVADAnalyzer | None:
        return self._params.vad_analyzer

    #
    # Audio task
    #

    async def start(self, frame: StartFrame):
        # Create audio input queue and task if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())

    async def stop(self):
        # Wait for the task to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_task.cancel()
            await self._audio_task

    async def push_audio_frame(self, frame: AudioRawFrame):
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_in_queue.put(frame)

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            try:
                frame: AudioRawFrame = await self._audio_in_queue.get()

                audio_passthrough = True

                # Check VAD and push event if necessary. We just care about
                # changes from QUIET to SPEAKING and vice versa.
                if self._params.vad_enabled:
                    vad_state = await self._handle_vad(frame.audio, vad_state)
                    audio_passthrough = self._params.vad_audio_passthrough

                # Push audio downstream if passthrough.
                if audio_passthrough:
                    await self.queue_frame(frame)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception(f"{self} error reading audio frames: {e}")

    async def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frames)
        if new_vad_state != vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_interruptions(frame)

            vad_state = new_vad_state
        return vad_state

    #
    # Handle interruptions
    #

    async def _handle_interruptions(self, frame: Frame):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band
            if isinstance(frame, UserStartedSpeakingFrame):
                logging.debug("User started speaking")
                await super()._handle_interruptions(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logging.debug("User stopped speaking")
                await self.push_frame(StopInterruptionFrame())
        await self.queue_frame(frame)

    #
    # Process frame
    #
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        return await super().process_frame(frame, direction)
