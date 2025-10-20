from collections import deque
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

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

from src.common.ringbuffer import RingBuffer
from src.common.interface import IVADAnalyzer
from src.common.types import AudioVADTurnParams, VADState
from src.types.frames.control_frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from src.types.frames.sys_frames import BotInterruptionFrame, MetricsFrame
from src.types.speech.turn_analyzer import EndOfTurnState
from src.types.frames.data_frames import VADStateAudioRawFrame


class AudioVADInputProcessor(InputProcessor):
    def __init__(
        self,
        params: AudioVADTurnParams,
        name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs,
    ):
        super().__init__(name=name, loop=loop, **kwargs)
        self._params = params
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._bytes_buffer = RingBuffer(
            self._params.audio_in_buffer_secs
            * self._params.audio_in_sample_rate
            * self._params.audio_in_channels
            * self._params.audio_in_sample_width
        )

        self._vad_analyzer: IVADAnalyzer | None = params.vad_analyzer
        self._num_bytes_required = 0
        if self._vad_analyzer:
            self._num_bytes_required = (
                self._vad_analyzer.num_frames_required()
                * self._params.audio_in_channels
                * self._params.audio_in_sample_width
            )

    @property
    def vad_analyzer(self) -> IVADAnalyzer | None:
        return self._params.vad_analyzer

    def set_vad_analyzer(self, analyzer: IVADAnalyzer):
        self._vad_analyzer = analyzer

    #
    # Audio task
    #

    async def start(self, frame: StartFrame):
        logging.info(f"{self.name} start, params: {self._params}")
        # Create audio input queue and task if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())

    async def stop(self):
        # Wait for the task to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            logging.info("stop to Cancelling audio task")
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
            logging.info("cancel to Cancelling audio task")
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
                frame: AudioRawFrame = await asyncio.wait_for(self._audio_in_queue.get(), timeout=1)

                if self._num_bytes_required == 0:
                    continue

                self._bytes_buffer.extend_bytes(frame.audio)
                while self._bytes_buffer.size() >= self._num_bytes_required:
                    chunk_bytes = self._bytes_buffer.pop_bytes(self._num_bytes_required)
                    previous_vad_state = vad_state
                    # Check VAD and push event if necessary. We just care about
                    # changes from QUIET to SPEAKING and vice versa.
                    if self._params.vad_enabled:
                        vad_state_frame, user_interuption_frame = await self._handle_vad(
                            chunk_bytes, vad_state
                        )
                        vad_state = vad_state_frame.state

                    if user_interuption_frame and isinstance(
                        user_interuption_frame, UserStartedSpeakingFrame
                    ):  # start speak
                        await self._handle_interruptions(user_interuption_frame, True)

                    # Push audio downstream if passthrough.
                    if self._params.vad_enabled and self._params.vad_audio_passthrough:
                        if len(vad_state_frame.audio) > 0:
                            await self.push_frame(vad_state_frame)
                    else:
                        await self.push_frame(frame)

                    if user_interuption_frame and isinstance(
                        user_interuption_frame, UserStoppedSpeakingFrame
                    ):  # stop speak without turn analyzer
                        await self._handle_interruptions(user_interuption_frame, True)

                    if self._params.turn_analyzer:  # stop speak with turn analyzer
                        await self._run_turn_analyzer(frame, vad_state, previous_vad_state)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logging.info(f"{self.name} audio_task_handler cancelled")
                break
            except Exception as e:
                logging.exception(f"{self} error reading audio frames: {e}")
                if self.get_event_loop().is_closed():
                    logging.warning(f"{self.name} event loop is closed")
                    break
                # Handle RuntimeError for shutdown executor
                if isinstance(
                    e, RuntimeError
                ) and "cannot schedule new futures after shutdown" in str(e):
                    logging.warning(f"{self.name} executor shutdown, stopping audio task")
                    break

    async def _vad_analyze(self, audio_bytes: bytes) -> VADStateAudioRawFrame:
        vad_state_frame = VADStateAudioRawFrame(state=VADState.QUIET, audio=audio_bytes)
        if self.vad_analyzer:
            vad_state_frame: VADStateAudioRawFrame = await self.get_event_loop().run_in_executor(
                self._executor, self.vad_analyzer.analyze_audio, audio_bytes
            )
        return vad_state_frame

    async def _handle_vad(self, audio_bytes: bytes, vad_state: VADState) -> VADStateAudioRawFrame:
        vad_state_frame = await self._vad_analyze(audio_bytes)
        new_vad_state = vad_state_frame.state
        user_interuption_frame = None
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            can_create_user_frames = (
                self._params.turn_analyzer is None
                or not self._params.turn_analyzer.speech_triggered
            )
            if new_vad_state == VADState.SPEAKING:
                if can_create_user_frames:
                    user_interuption_frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                if can_create_user_frames:
                    user_interuption_frame = UserStoppedSpeakingFrame()

            vad_state_frame.state = new_vad_state

        return vad_state_frame, user_interuption_frame

    async def _run_turn_analyzer(
        self, frame: AudioRawFrame, vad_state: VADState, previous_vad_state: VADState
    ):
        """Run turn analysis on audio frame and handle results."""
        is_speech = vad_state == VADState.SPEAKING or vad_state == VADState.STARTING
        # If silence exceeds threshold, we are going to receive EndOfTurnState.COMPLETE
        end_of_turn_state = self._params.turn_analyzer.append_audio(frame.audio, is_speech)
        if end_of_turn_state == EndOfTurnState.COMPLETE:
            await self._handle_end_of_turn_complete(end_of_turn_state)
        # Otherwise we are going to trigger to check if the turn is completed based on the VAD
        elif vad_state == VADState.QUIET and vad_state != previous_vad_state:
            await self._handle_end_of_turn()

    async def _handle_end_of_turn_complete(self, state: EndOfTurnState):
        """Handle completion of end-of-turn analysis."""
        if state == EndOfTurnState.COMPLETE:
            await self._handle_interruptions(UserStoppedSpeakingFrame(), True)

    async def _handle_end_of_turn(self):
        """Handle end-of-turn analysis and generate prediction results."""
        if self._params.turn_analyzer:
            state, prediction = await self._params.turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            await self._handle_end_of_turn_complete(state)

    async def _handle_prediction_result(self, result: Optional[Dict[str, Any]] = None):
        """Handle a prediction result event from the turn analyzer."""
        pass
        # todo: add metrics frame to push

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
                logging.info("User started speaking")
                await self._start_interruption()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logging.info("User stopped speaking")
                await self._stop_interruption()

        if push_frame:
            await self.push_frame(frame)

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
