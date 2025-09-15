from datetime import datetime
import os
import time
import logging

from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.frames.control_frames import StartFrame, EndFrame
from apipeline.frames.sys_frames import CancelFrame
from apipeline.pipeline.pipeline import FrameDirection
from apipeline.processors.frame_processor import FrameProcessor

from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR
from src.types.frames.data_frames import PathAudioRawFrame


class AudioSaveProcessor(FrameProcessor):
    """
    Save AudioRawFrame to file
    """

    def __init__(
        self, prefix_name: str = "record", save_dir: str = RECORDS_DIR, pass_raw_audio: bool = False
    ):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix_name = prefix_name
        self.pass_raw_audio = pass_raw_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            file_path = await self.save(frame)
            if self.pass_raw_audio:
                await self.push_frame(frame, direction)
            else:
                path_frame = PathAudioRawFrame(
                    path=file_path,
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    sample_width=frame.sample_width,
                    num_channels=frame.num_channels,
                )
                await self.push_frame(path_frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def save(self, frame: AudioRawFrame) -> str:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
        output_file = os.path.join(self.save_dir, f"{self.prefix_name}_{formatted_time}.wav")
        file_path = await save_audio_to_file(
            frame.audio,
            output_file,
            audio_dir=self.save_dir,
            channles=frame.num_channels,
            sample_rate=frame.sample_rate,
            sample_width=frame.sample_width,
        )
        logging.debug(f"save frame:{frame} to path:{file_path}")
        return file_path


class SaveAllAudioProcessor(FrameProcessor):
    """
    Save all audio to file
    """

    def __init__(
        self,
        prefix_name: str = "record",
        save_dir: str = RECORDS_DIR,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        interval_seconds: int = 0,
    ):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix_name = prefix_name

        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self.interval_seconds = interval_seconds
        self._curr_time = 0

        self.audio_bytes = b""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            logging.info(f"{self.name} started")
            self.audio_bytes = b""
            self.accumulate_time = 0
            self._curr_time = time.time()

        if isinstance(frame, AudioRawFrame):
            self.audio_bytes += frame.audio
            interval_s = time.time() - self._curr_time
            if self.interval_seconds > 0 and interval_s > self.interval_seconds:
                await self.save()
                self._curr_time = time.time()

        await self.push_frame(frame, direction)

        if isinstance(frame, EndFrame):
            logging.info(f"{self.name} end")
            await self.save()
            self.audio_bytes = b""
        if isinstance(frame, CancelFrame):
            logging.info(f"{self.name} cancelled")
            await self.save()
            self.audio_bytes = b""

    async def save(self) -> str:
        if len(self.audio_bytes) == 0:
            return
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
        output_file = os.path.join(self.save_dir, f"{self.prefix_name}_{formatted_time}.wav")
        file_path = await save_audio_to_file(
            self.audio_bytes,
            output_file,
            audio_dir=self.save_dir,
            channles=self.channels,
            sample_rate=self.sample_rate,
            sample_width=self.sample_width,
        )
        logging.info(f"save {len(self.audio_bytes)=} to path:{file_path}")
