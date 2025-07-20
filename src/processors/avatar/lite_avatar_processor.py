import time
import logging
from fractions import Fraction
from typing import AsyncGenerator
import asyncio
import uuid

import av
import cv2
import numpy as np
from apipeline.frames import Frame, StartFrame, EndFrame, CancelFrame

from src.modules.avatar.lite_avatar import AudioSlice, LiteAvatar
from src.types.avatar import SpeechAudio
from src.types.avatar.lite_avatar import (
    AudioResult,
    AvatarStatus,
    MouthResult,
    SignalResult,
    VideoResult,
)
from src.processors.avatar.base import AvatarProcessorBase, SegmentedAvatarProcessor
from src.processors.avatar.help.speech_audio_slicer import SpeechAudioSlicer
from src.processors.avatar.help.video_audio_aligner import VideoAudioAligner
from src.types.frames import AudioRawFrame, OutputAudioRawFrame, OutputImageRawFrame


# class LiteAvatarProcessor(AvatarProcessorBase):
class LiteAvatarProcessor(SegmentedAvatarProcessor):
    def __init__(self, avatar: LiteAvatar, **kwargs):
        self._avatar = avatar
        self._init_option = self._avatar.init_option
        super().__init__(sample_rate=self._init_option.audio_sample_rate, **kwargs)
        logging.info(f"init {__name__} init_option: {self._init_option}")

        # running
        self._session_running = False
        self._session_start_time = 0
        self._current_speech_id = ""

        # running task
        self._audio2signal_task: asyncio.Task = None
        self._signal2img_task: asyncio.Task = None
        self._mouth2full_task: asyncio.Task = None
        # running asyncio task queue
        self._audio_slice_queue: asyncio.Queue = None
        self._signal_queue: asyncio.Queue = None
        self._mouth_img_queue: asyncio.Queue = None

        # for av frame
        self._global_frame_count = 0
        self._current_audio_pts = 0  # in ms
        self._current_video_pts = 0
        self._last_speech_ended = True

        # audio input slice
        self._speech_audio_slicer: SpeechAudioSlicer = None
        # video and audio frame aligner
        self._video_audio_aligner: VideoAudioAligner = None

        # is display video debug text
        self._is_show_video_debug_text = self._init_option.is_show_video_debug_text

        # load avatar
        self._avatar.load()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session_running = True
        self._init()
        self._start_tasks()
        self._session_start_time = time.time()

    def _init(self):
        avatar_config = self._avatar.get_config()
        self._speech_audio_slicer = SpeechAudioSlicer(
            self._init_option.audio_sample_rate,  # input
            avatar_config.input_audio_sample_rate,  # output for avatar input audio sample rate
            avatar_config.input_audio_slice_duration,
            enable_fast_mode=self._init_option.enable_fast_mode,
        )
        self._video_audio_aligner = VideoAudioAligner(self._init_option.video_frame_rate)

    def _start_tasks(self):
        self._audio_slice_queue = asyncio.Queue()
        self._signal_queue = asyncio.Queue()
        self._mouth_img_queue = asyncio.Queue()
        self._audio2signal_task = self.get_event_loop().create_task(self._audio2signal_loop())
        self._signal2img_task = self.get_event_loop().create_task(self._signal2img_loop())
        self._mouth2full_task = self.get_event_loop().create_task(self._mouth2full_loop())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop()

    async def cancle(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop()

    async def _stop(self):
        logging.info(
            f"stop avatar processor, totol session time {time.time() - self._session_start_time:.3f}",
        )
        self._session_running = False
        if self._signal2img_task is not None:
            self._signal2img_task.cancel()
            await self._signal2img_task
            self._signal2img_task = None
        if self._audio2signal_task is not None:
            self._audio2signal_task.cancel()
            await self._audio2signal_task
            self._audio2signal_task = None
        if self._mouth2full_task is not None:
            self._mouth2full_task.cancel()
            await self._mouth2full_task
            self._mouth2full_task = None
        logging.info("avatar processor stopped")

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
        audio_slices = self._speech_audio_slicer.get_speech_audio_slice(speech_audio)
        for audio_slice in audio_slices:
            await self._audio_slice_queue.put(audio_slice)

    async def _audio2signal_loop(self):
        """
        generate signal for signal2img
        """
        logging.info("audio2signal loop started")
        speech_id = str(uuid.uuid4())
        audio_slice = None
        target_round_time = 0.9
        while self._session_running:
            start_time = time.time()
            try:
                audio_slice: AudioSlice = await asyncio.wait_for(
                    self._audio_slice_queue.get(), timeout=0.1
                )
                target_round_time = audio_slice.get_audio_duration() - 0.1

                speech_id = audio_slice.speech_id
                if speech_id != self._current_speech_id:
                    self._last_speech_ended = False
                    self._current_speech_id = speech_id
                if audio_slice.end_of_speech:
                    self._last_speech_ended = True

                logging.info(
                    f"audio2signal input audio durtaion {audio_slice.get_audio_duration()}"
                )
                signal_vals = await asyncio.to_thread(self._avatar.audio2signal, audio_slice)
                avatar_status = AvatarStatus.SPEAKING

                # remove front padding audio and relative frames
                front_padding_duration = audio_slice.front_padding_duration
                target_round_time = audio_slice.get_audio_duration() - front_padding_duration - 0.1
                padding_frame_count = int(
                    front_padding_duration * self._init_option.video_frame_rate
                )
                signal_vals = signal_vals[padding_frame_count:]
                padding_audio_count = (
                    int(front_padding_duration) * self._init_option.audio_sample_rate * 2
                )
                audio_slice.play_audio_data = audio_slice.play_audio_data[padding_audio_count:]

                audio_slice.play_audio_data = (
                    self._video_audio_aligner.get_speech_level_algined_audio(
                        audio_slice.play_audio_data,
                        audio_slice.play_audio_sample_rate,
                        len(signal_vals),
                        audio_slice.speech_id,
                        audio_slice.end_of_speech,
                    )
                )

                for i, signal in enumerate(signal_vals):
                    end_of_speech = audio_slice.end_of_speech and i == len(signal_vals) - 1
                    middle_result = SignalResult(
                        speech_id=speech_id,
                        end_of_speech=end_of_speech,
                        middle_data=signal,
                        frame_id=i,
                        global_frame_id=self._global_frame_count,
                        avatar_status=avatar_status,
                        audio_slice=audio_slice if i == 0 else None,
                    )
                    self._signal_queue.put_nowait(middle_result)
                cost = time.time() - start_time
                sleep_time = target_round_time - cost
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logging.warning("audio2signal_loop task cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.exception(f"audio2signal_loop task error: {e}")
                continue
        logging.info("audio2signal loop stopped")

    async def _signal2img_loop(self):
        """
        generate image and do callbacks
        """
        logging.info("signal2img loop started")
        start_time = -1
        timestamp = 0

        # delay start to ensure no extra audio and video generated
        await asyncio.sleep(0.5)

        while self._session_running:
            try:
                if self._signal_queue.empty():
                    # generate idle
                    signal_val = self._avatar.get_idle_signal(1)[0]
                    avatar_status = (
                        AvatarStatus.LISTENING if self._last_speech_ended else AvatarStatus.SPEAKING
                    )
                    signal = SignalResult(
                        speech_id=self._current_speech_id,
                        end_of_speech=False,
                        middle_data=signal_val,
                        frame_id=0,
                        avatar_status=avatar_status,
                        audio_slice=self._get_idle_audio_slice(1),
                    )
                else:
                    # no empty, so get nowait
                    signal: SignalResult = self._signal_queue.get_nowait()

                out_image, bg_frame_id = await asyncio.to_thread(
                    self._avatar.signal2img,
                    signal.middle_data,
                )
                # create mouth result
                mouth_result = MouthResult(
                    speech_id=signal.speech_id,
                    mouth_image=out_image,
                    bg_frame_id=bg_frame_id,
                    end_of_speech=signal.end_of_speech,
                    avatar_status=signal.avatar_status,
                    audio_slice=signal.audio_slice,
                    global_frame_id=self._global_frame_count,
                )

                self._global_frame_count += 1

                await self._mouth_img_queue.put(mouth_result)

                if start_time == -1:
                    start_time = time.time()
                    timestamp = 0
                else:
                    timestamp += 1 / self._init_option.video_frame_rate
                    wait = start_time + timestamp - time.time()
                    if wait > 0:
                        await asyncio.sleep(wait)

            except asyncio.CancelledError:
                logging.warning("signal2img task cancelled")
                break
            except Exception as e:
                logging.exception(f"signal2img task error: {e}")
                continue

        logging.info("signal2img loop ended")

    async def _mouth2full_loop(self):
        logging.info("combine img loop started")
        while self._session_running:
            try:
                mouth_reusult: MouthResult = await asyncio.wait_for(
                    self._mouth_img_queue.get(), timeout=0.1
                )
                image = mouth_reusult.mouth_image
                bg_frame_id = mouth_reusult.bg_frame_id
                full_img = await asyncio.to_thread(self._avatar.mouth2full, image, bg_frame_id)

                if mouth_reusult.audio_slice is not None:
                    # create audio result
                    audio_data = mouth_reusult.audio_slice.play_audio_data
                    audio_frame = av.AudioFrame.from_ndarray(
                        np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1),
                        format="s16",
                        layout="mono",
                    )
                    audio_time_base = Fraction(1, self._init_option.audio_sample_rate)
                    audio_frame.time_base = audio_time_base
                    audio_frame.pts = self._current_audio_pts
                    audio_frame.sample_rate = mouth_reusult.audio_slice.play_audio_sample_rate
                    self._current_audio_pts += len(audio_data) // 2

                    audio_result = AudioResult(
                        audio_frame=audio_frame, speech_id=mouth_reusult.audio_slice.speech_id
                    )
                    logging.debug(
                        "create audio with duration {:.3f}s, status: {}",
                        mouth_reusult.audio_slice.get_audio_duration(),
                        mouth_reusult.avatar_status,
                    )

                    await self._callback_audio(audio_result)

                # create video result
                if self._is_show_video_debug_text:
                    full_img = cv2.putText(
                        full_img,
                        f"{mouth_reusult.avatar_status} {mouth_reusult.global_frame_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                if self._init_option.is_flip is True:
                    full_img = cv2.flip(full_img, 1)
                video_frame = av.VideoFrame.from_ndarray(full_img, format="bgr24")
                video_frame.time_base = Fraction(1, self._init_option.video_frame_rate)
                video_frame.pts = self._current_video_pts
                self._current_video_pts += 1

                image_result = VideoResult(
                    video_frame=video_frame,
                    speech_id=mouth_reusult.speech_id,
                    avatar_status=mouth_reusult.avatar_status,
                    end_of_speech=mouth_reusult.end_of_speech,
                    bg_frame_id=bg_frame_id,
                )

                await self._callback_image(image_result)
            except asyncio.CancelledError:
                logging.warning("combine img mouth2full_loop task cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.exception(f"combine img mouth2full_loop task error: {e}")
                continue

        logging.info("combine img loop ended")

    async def _callback_image(self, image_result: VideoResult):
        video_frame: av.VideoFrame = image_result.video_frame
        img = video_frame.to_image()
        if self._session_running:
            await self.push_frame(
                OutputImageRawFrame(
                    image=img.tobytes(),
                    size=img.size,
                    format=img.format,
                    mode=img.mode,
                )
            )

    async def _callback_audio(self, audio_result: AudioResult):
        audio_frame: av.AudioFrame = audio_result.audio_frame
        if self._session_running:
            await self.queue_frame(
                OutputAudioRawFrame(
                    audio=audio_frame.to_ndarray().tobytes(),
                    sample_rate=audio_frame.sample_rate,
                )
            )

    def _get_idle_audio_slice(self, idle_frame_count):
        speech_id = "" if self._last_speech_ended else self._current_speech_id
        # generate silence audio
        frame_rate = self._init_option.video_frame_rate
        play_audio_sample_rate = self._init_option.audio_sample_rate
        idle_duration_seconds = idle_frame_count / frame_rate
        idle_data_length = int(2 * idle_duration_seconds * play_audio_sample_rate)
        idle_audio_data = bytes(idle_data_length)
        idle_audio_slice = AudioSlice(
            speech_id=speech_id,
            play_audio_data=idle_audio_data,
            play_audio_sample_rate=play_audio_sample_rate,
            algo_audio_data=None,
            algo_audio_sample_rate=0,
            end_of_speech=False,
        )
        return idle_audio_slice
