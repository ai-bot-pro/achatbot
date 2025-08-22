import asyncio
from datetime import datetime
import os
import logging

from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.pipeline.pipeline import FrameDirection
from apipeline.processors.async_frame_processor import AsyncFrameProcessor

from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR, VADState
from src.types.frames.data_frames import PathAudioRawFrame, VADStateAudioRawFrame, VADAudioRawFrame


class VADAudioProcessor(AsyncFrameProcessor):
    """
    VAD segment audio: {start} Speaking -> Quiet {end} audio segment
    """

    def __init__(
        self,
        prefix_name: str = "record",
        save_dir: str = RECORDS_DIR,
        pass_raw_audio: bool = True,
    ):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix_name = prefix_name
        self.pass_raw_audio = pass_raw_audio
        self.vad_audio_bytes = b""
        self.cur_speech_id = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADStateAudioRawFrame):
            if self.pass_raw_audio:
                await self.queue_frame(frame, direction)

            self.vad_audio_bytes += frame.audio
            if frame.is_final is True and frame.speech_id != self.cur_speech_id:
                self.cur_speech_id = frame.speech_id
                self.push(frame)
        else:
            await self.push_frame(frame, direction)

    async def push(self, frame: VADStateAudioRawFrame) -> str:
        await self.push_frame(
            VADAudioRawFrame(
                audio=self.vad_audio_bytes,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                sample_width=frame.num_frames,
                speech_id=self.cur_speech_id,
                start_at_s=frame.start_at_s,
                end_at_s=frame.end_at_s,
            )
        )
        self.vad_audio_bytes = b""


class VADAudioSaveProcessor(AsyncFrameProcessor):
    """
    VADAudioProcessor + AudioSaveProcessor
    """

    def __init__(
        self,
        prefix_name: str = "record",
        save_dir: str = RECORDS_DIR,
        pass_raw_audio: bool = True,
        pass_path_audio: bool = False,
        save_type: str = "local",  # local or remote s3
    ):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix_name = prefix_name
        self.pass_raw_audio = pass_raw_audio
        self.pass_path_audio = pass_path_audio
        self.save_type = save_type

        self.vad_audio_bytes = b""
        self.cur_speech_id = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADStateAudioRawFrame):
            if self.pass_raw_audio:
                # await self.push_frame(frame, direction) # sync to debug :)
                await self.queue_frame(frame, direction)  # async to run

            if frame.state == VADState.SPEAKING:
                self.vad_audio_bytes += frame.audio
            # print("-->", frame, self.cur_speech_id)
            if frame.is_final is True and frame.speech_id != self.cur_speech_id:
                self.cur_speech_id = frame.speech_id
                file_path = ""
                match self.save_type:
                    case "local":
                        file_path = await self.save_local(frame)
                if file_path and self.pass_path_audio is True and self.pass_raw_audio is False:
                    path_frame = PathAudioRawFrame(
                        path=file_path,
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        sample_width=frame.sample_width,
                        num_channels=frame.num_channels,
                    )
                    await self.queue_frame(path_frame, direction)
                self.vad_audio_bytes = b""
        else:
            await self.push_frame(frame, direction)

    async def save_local(self, frame: AudioRawFrame) -> str:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
        output_file = os.path.join(
            self.save_dir, f"{self.cur_speech_id}_{self.prefix_name}_{formatted_time}.wav"
        )
        file_path = await save_audio_to_file(
            self.vad_audio_bytes,
            output_file,
            audio_dir=self.save_dir,
            channles=frame.num_channels,
            sample_rate=frame.sample_rate,
            sample_width=frame.sample_width,
        )
        logging.info(f"save frame:{frame} to path:{file_path}")
        return file_path
