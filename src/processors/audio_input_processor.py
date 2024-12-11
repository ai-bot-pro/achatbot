import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from apipeline.frames.base import Frame
from apipeline.frames.control_frames import StartFrame
from apipeline.frames.sys_frames import (
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    CancelFrame,
)
from apipeline.frames.data_frames import AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection
from apipeline.processors.input_processor import InputProcessor

from src.common.interface import IVADAnalyzer
from src.common.types import AudioVADParams, VADState
from src.types.frames.control_frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from src.types.frames.sys_frames import BotInterruptionFrame


class AudioVADInputProcessor(InputProcessor):
    def __init__(
        self,
        params: AudioVADParams,
        name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs,
    ):
        super().__init__(name=name, loop=loop, **kwargs)
        self._params = params
        self._executor = ThreadPoolExecutor(max_workers=3)

        self._vad_analyzer: IVADAnalyzer | None = params.vad_analyzer

    @property
    def vad_analyzer(self) -> IVADAnalyzer | None:
        return self._params.vad_analyzer

    def set_vad_analyzer(self, analyzer: IVADAnalyzer):
        self._vad_analyzer = analyzer

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

        # Wait for the push frame task to finish. It will finish when the
        # EndFrame is actually processed. wait timeout 1s
        try:
            self._push_frame_task and await asyncio.wait_for(self._push_frame_task, 1)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass

    async def cancel(self, frame: CancelFrame):
        """
        Cancel all the tasks and wait for them to finish.
        """
        if self._params.audio_in_enabled or self._params.vad_enabled:
            # Wait for audio_task cancel to finish
            self._audio_task.cancel()
            await self._audio_task

        # Wait for async processor push_frame_task cancel to finish
        await self.cleanup()

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

    async def _vad_analyze(self, audio_frames: bytes) -> VADState:
        state = VADState.QUIET
        if self.vad_analyzer:
            state = await self.get_event_loop().run_in_executor(
                self._executor, self.vad_analyzer.analyze_audio, audio_frames
            )
        return state

    async def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frames)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_interruptions(frame, True)

            vad_state = new_vad_state
        return vad_state

    #
    # Handle interruptions
    #
    async def _start_interruption(self):
        if not self.interruptions_allowed:
            return

        # use async frame processor _handle_interruptions
        # (cancel old then create new)
        await super()._handle_interruptions(StartInterruptionFrame())

    async def _stop_interruption(self):
        if not self.interruptions_allowed:
            return

        await self.push_frame(StopInterruptionFrame())

    async def _handle_interruptions(self, frame: Frame, push_frame: bool):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band
            if isinstance(frame, BotInterruptionFrame):
                logging.debug("Bot interruption")
                await self._start_interruption()
            elif isinstance(frame, UserStartedSpeakingFrame):
                logging.debug("User started speaking")
                await self._start_interruption()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logging.debug("User stopped speaking")
                await self._stop_interruption()

        if push_frame:
            await self.queue_frame(frame)

    #
    # Process frame
    #

    async def process_sys_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BotInterruptionFrame):
            await self._handle_interruptions(frame, False)
        elif isinstance(frame, StartInterruptionFrame):
            await self._start_interruption()
        elif isinstance(frame, StopInterruptionFrame):
            await self._stop_interruption()
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        return await super().process_frame(frame, direction)
