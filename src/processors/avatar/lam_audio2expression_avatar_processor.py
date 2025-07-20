import time
import logging
from typing import AsyncGenerator, cast
import asyncio

from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames import Frame, StartFrame, EndFrame, CancelFrame
import numpy as np

from src.modules.avatar.lam_audio2expression import LAMAudio2ExpressionAvatar
from src.types.avatar import SpeechAudio, AudioSlice, AvatarStatus
from src.processors.avatar.base import AvatarProcessorBase, SegmentedAvatarProcessor
from src.processors.avatar.help.speech_audio_slicer import SpeechAudioSlicer
from src.types.frames import AudioRawFrame, AnimationAudioRawFrame
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.types.frames.control_frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)


# class LAMAudio2ExpressionAvatarProcessor(AvatarProcessorBase):
class LAMAudio2ExpressionAvatarProcessor(SegmentedAvatarProcessor):
    def __init__(self, avatar: LAMAudio2ExpressionAvatar, **kwargs):
        self._avatar = avatar
        self.input_audio_slice_duration = float(kwargs.get("input_audio_slice_duration", "1"))
        super().__init__(sample_rate=self._avatar.args.speaker_audio_sample_rate, **kwargs)

        # running
        self._session_running = False
        self._session_start_time = 0

        # running task
        self._audio2expression_task: asyncio.Task = None
        # running asyncio task queue
        self._audio_slice_queue: asyncio.Queue = None

        # audio input slice
        self._speech_audio_slicer: SpeechAudioSlicer = None

        # don't to load avatar, outside to load, processor just to run with session
        # self._avatar.load()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session_running = True
        self._init()
        self._start_tasks()
        self._session_start_time = time.time()

    def _init(self):
        self._speech_audio_slicer = SpeechAudioSlicer(
            self._avatar.args.speaker_audio_sample_rate,  # input
            self._avatar.args.avatar_audio_sample_rate,  # output for avatar input audio sample rate
            self.input_audio_slice_duration,
        )

    def _start_tasks(self):
        self._audio_slice_queue = asyncio.Queue()
        self._audio2expression_task = self.get_event_loop().create_task(
            self._audio2expression_loop()
        )

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop()

    async def _stop(self):
        logging.info(
            f"stop avatar processor, totol session time {time.time() - self._session_start_time:.3f}",
        )
        self._session_running = False
        if self._audio2expression_task is not None:
            self._audio2expression_task.cancel()
            await self._audio2expression_task
            self._audio2expression_task = None
        logging.info("avatar processor stopped")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.queue_frame(
                AnimationAudioRawFrame(
                    audio=b"",
                    avatar_status=str(AvatarStatus.LISTENING),
                )
            )
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.queue_frame(
                AnimationAudioRawFrame(
                    audio=b"",
                    avatar_status=str(AvatarStatus.THINKING),
                )
            )
        if isinstance(frame, TTSStartedFrame):
            await self.queue_frame(
                AnimationAudioRawFrame(
                    audio=b"",
                    avatar_status=str(AvatarStatus.RESPONDING),
                )
            )
        elif isinstance(frame, TTSStoppedFrame):
            await self.queue_frame(
                AnimationAudioRawFrame(
                    audio=b"",
                    avatar_status=str(AvatarStatus.LISTENING),
                )
            )

    async def run_avatar(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        await self._add_audio(
            SpeechAudio(
                end_of_speech=True,
                speech_id="",
                sample_rate=frame.sample_rate,
                audio_data=frame.audio,
            )
        )

        # inner push/queue audio/image frame, yield None
        yield None

    async def _add_audio(self, speech_audio: SpeechAudio):
        try:
            audio_slices = self._speech_audio_slicer.get_speech_audio_slice(speech_audio)
            for audio_slice in audio_slices:
                await self._audio_slice_queue.put(audio_slice)
        except Exception as e:
            logging.exception(e)

    async def _audio2expression_loop(self):
        """
        generate signal for signal2img
        """
        logging.info("audio2expression loop started")
        audio_slice = None
        target_round_time = 0.0
        inference_context = None
        data_time = time.time()
        while self._session_running:
            start_time = time.time()
            try:
                audio_slice: AudioSlice = await asyncio.wait_for(
                    self._audio_slice_queue.get(), timeout=0.1
                )
                logging.debug(
                    f"audio2expression input audio durtaion {audio_slice.get_audio_duration()}"
                )
                if self._avatar.infer is None:
                    continue

                data_time = time.time()
                target_round_time = audio_slice.get_audio_duration()

                np_audio = bytes2NpArrayWith16(audio_slice.algo_audio_data)
                result, context_update = await asyncio.to_thread(
                    self._avatar.infer.infer_streaming_audio,
                    np_audio,
                    audio_slice.algo_audio_sample_rate,
                    inference_context,
                )
                inference_context = context_update
                if audio_slice.end_of_speech:
                    inference_context = None

                expression = result.get("expression")
                animation_json = self._avatar.export_animation_json(expression)
                avatar_status = str(AvatarStatus.SPEAKING)
                await self.queue_frame(
                    AnimationAudioRawFrame(
                        audio=audio_slice.algo_audio_data,
                        sample_rate=audio_slice.algo_audio_sample_rate,
                        animation_json=animation_json,
                        avatar_status=avatar_status,
                    )
                )

                # NOTE: just for local align audio and expression
                # if not, need remote(recieve) to align
                cost = time.time() - start_time
                sleep_time = target_round_time - cost
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logging.info(f"{self.name} audio2expression_loop task cancelled")
                break
            except asyncio.TimeoutError:
                if time.time() - data_time > 1:
                    await self.queue_frame(
                        AnimationAudioRawFrame(
                            audio=b"",
                            avatar_status=str(AvatarStatus.RESPONDING),
                        )
                    )
                    data_time = time.time()
                continue
            except Exception as e:
                logging.exception(f"audio2expression_loop task error: {e}")
                continue
        logging.info("audio2expression loop stopped")
