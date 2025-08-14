import asyncio
import logging
import math
import time
import traceback

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import CancelFrame, ErrorFrame
from apipeline.frames.control_frames import Frame, StartFrame, EndFrame

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.utils.ffmpeg_piper import FFmpegPiper, FFmpegState
from src.types.frames.data_frames import AudioRawFrame

SENTINEL = object()  # unique sentinel object for end of stream marker, like golang {}


class FFMPEGPipeProcessor(FrameProcessor):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        out_format: str = "s16le",
        acodec: str = "pcm_s16le",
        min_chunk_s: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ffmpeg_piper = FFmpegPiper(
            sample_rate=sample_rate,
            channels=channels,
            out_format=out_format,
            acodec=acodec,
        )
        self._ffmpeg_error = None

        async def handle_ffmpeg_error(error_type: str):
            logging.error(f"FFmpeg error: {error_type}")
            self._ffmpeg_error = error_type

        self.ffmpeg_piper.on_error_callback = handle_ffmpeg_error

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples_per_sec = int(sample_rate * min_chunk_s)
        self.bytes_per_sample = 2
        self.chunk_bytes_per_sec = self.chunk_samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_stopping = False
        self.pcm_buffer = bytearray()

        self.ffmpeg_reader_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            await self.process_audio(AudioRawFrame.audio)
            return

        if isinstance(frame, StartFrame):
            await self.start(frame)
        if isinstance(frame, EndFrame):
            await self.stop(frame)
        if isinstance(frame, CancelFrame):
            await self.cancel(frame)
        await self.push_frame(frame, direction)

    async def start(self, frame: StartFrame):
        success = await self.ffmpeg_piper.start()
        if not success:
            logging.error("Failed to start FFmpeg Piper")

        self.ffmpeg_reader_task = self.get_event_loop().create_task(self.ffmpeg_stdout_reader())
        logging.info("start FFMPEGPipProcessor")

    async def stop(self, frame: EndFrame):
        await self.ffmpeg_piper.stop()

    async def cancel(self, frame: CancelFrame):
        self.ffmpeg_reader_task.cancel()
        await self.ffmpeg_reader_task

    async def process_audio(self, audio_bytes):
        """Process incoming audio data."""
        if not audio_bytes:
            logging.info("Empty audio audio_bytes received, initiating stop sequence.")
            self.is_stopping = True
            # Signal FFmpeg manager to stop accepting data
            await self.ffmpeg_piper.stop()
            return

        if self.is_stopping:
            logging.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        success = await self.ffmpeg_piper.write_data(audio_bytes)
        if not success:
            ffmpeg_state = await self.ffmpeg_piper.get_state()
            if ffmpeg_state == FFmpegState.FAILED:
                logging.error("FFmpeg is in FAILED state, cannot process audio")
            else:
                logging.warning("Failed to write audio data to FFmpeg")

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it."""
        beg = time()

        while True:
            try:
                # Check if FFmpeg is running
                state = await self.ffmpeg_piper.get_state()
                if state == FFmpegState.FAILED:
                    logging.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logging.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    logging.warning(f"FFmpeg is in {state} state, waiting...")
                    await asyncio.sleep(0.5)
                    continue

                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = current_time

                chunk = await self.ffmpeg_piper.read_data(buffer_size)

                if not chunk:
                    if self.is_stopping:
                        logging.info("FFmpeg stdout closed, stopping.")
                        break
                    else:
                        # No data available, but not stopping - FFmpeg might be restarting
                        await asyncio.sleep(0.1)
                        continue

                self.pcm_buffer.extend(chunk)

                # Process when enough data
                if len(self.pcm_buffer) >= self.chunk_bytes_per_sec:
                    if len(self.pcm_buffer) > self.max_bytes_per_sec:
                        logging.warning(
                            f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                            f"Consider using a smaller model."
                        )

                    # pcm_array = bytes2NpArrayWith16(self.pcm_buffer[: self.max_bytes_per_sec])
                    self.push_frame(
                        AudioRawFrame(
                            audio=self.pcm_buffer[: self.max_bytes_per_sec].copy(),
                            sample_rate=self.sample_rate,
                            num_channels=self.channels,
                        )
                    )
                    self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec :]

            except Exception as e:
                logging.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logging.warning(f"Traceback: {traceback.format_exc()}")
                # Try to recover by waiting a bit
                await asyncio.sleep(1)

                # Check if we should exit
                if self.is_stopping:
                    break

        logging.info("FFmpeg stdout processing finished.")
