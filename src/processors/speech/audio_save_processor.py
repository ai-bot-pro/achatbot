import datetime
import os
import logging

from apipeline.frames.data_frames import Frame, AudioRawFrame
from apipeline.pipeline.pipeline import FrameDirection
from apipeline.processors.frame_processor import FrameProcessor

from src.common.utils.wav import save_audio_to_file
from src.common.types import RECORDS_DIR


class AudioSaveProcessor(FrameProcessor):
    def __init__(self, prefix_name: str = "record", save_dir: str = RECORDS_DIR):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.prefix_name = prefix_name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self.save(frame)
        await self.push_frame(frame, direction)

    async def save(self, frame: AudioRawFrame) -> str:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.save_dir, f"{self.prefix_name}_{formatted_time}.wav")
        file_path = await save_audio_to_file(
            frame.audio,
            output_file,
            audio_dir=RECORDS_DIR,
            channles=frame.channels,
            sample_rate=frame.sample_rate,
            sample_width=frame.sample_width,
        )
        logging.info(f"save {frame} to {file_path}")
        return file_path
