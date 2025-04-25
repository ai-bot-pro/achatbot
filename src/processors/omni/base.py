import asyncio
import logging
from abc import abstractmethod
import queue
import threading
from typing import AsyncGenerator
import uuid

from apipeline.pipeline.pipeline import FrameDirection
from apipeline.frames import *
import numpy as np

from src.common.factory import EngineClass
from src.common.interface import ILlm
from src.common.session import Session
from src.common.types import CHANNELS, RATE, SessionCtx
from src.types.frames.data_frames import Frame, VisionImageVoiceRawFrame, PathAudioRawFrame
from src.processors.ai_processor import AsyncAIProcessor


class VisionVoiceProcessorBase(AsyncAIProcessor):
    """
    VisionVoiceProcessorBase is a base class for vision+voice processors.
    input: vision + voice frame
    use omni lm to process vision + voice frames
    output: text+audio frame
    """

    def __init__(
        self,
        llm: ILlm | EngineClass | None = None,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert llm is not None, "llm is required"
        self._llm = llm
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._queue = queue.Queue()
        self._input_queue = queue.Queue()
        self._generate_thread = None
        self._sleep_time = no_stream_sleep_time

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {"sample_rate": RATE, "channels": CHANNELS}

    def _generate(self):
        while True:
            try:
                session = self._input_queue.get()
                if session is None:
                    self._queue.put(None)  # Signal the end of the stream
                    break  # Signal to stop the thread
                tensor_audio_stream = self._llm.generate(session)
                for item in tensor_audio_stream:
                    self._queue.put(item)
                self._queue.put(None)  # Signal the end of the stream
            except Exception as e:
                logging.error(f"Exception generate: {e}", exc_info=True)
                self._queue.put(None)  # Signal the end of the stream
                break

    @abstractmethod
    async def run(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield frame

    async def say(self, text: str):
        logging.info(f"say: {text}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionImageVoiceRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._create_push_task()
        self._generate_thread = threading.Thread(target=self._generate)
        self._generate_thread.start()
        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("cancel done")

    def send_input(self, session: Session):
        self._input_queue.put(session)

    async def gen(self) -> AsyncGenerator[Frame, None]:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    break  # End of the stream
                logging.debug(f"generate data: {item}")
                tensor_audio = item.pop("audio_wav", None)
                text = item.pop("text", "").strip()
                if text != "":
                    await self.push_frame(TextFrame(text=text))

                if tensor_audio is not None:  # don't use if tensor_audio to check
                    audio_bytes = (
                        (tensor_audio.float().detach().cpu().numpy() * 32768)
                        .astype(np.int16)
                        .tobytes()
                    )
                    logging.info(
                        f"audio tensor:{tensor_audio.shape},push audio len:{len(audio_bytes)}"
                    )
                    await self.push_frame(
                        AudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self.stream_info["sample_rate"],
                            num_channels=self.stream_info["channels"],
                        )
                    )
                yield None
            except queue.Empty:
                # yield asysncio.sleep to allow other tasks to run, e.g.: sink task (write audio)
                await asyncio.sleep(self._sleep_time)
                continue


class MockVisionVoiceProcessor(VisionVoiceProcessorBase):
    async def run(self, frame: VisionImageVoiceRawFrame) -> AsyncGenerator[Frame, None]:
        logging.debug(f"VisionImageVoiceRawFrame: {frame}")
        self._session.ctx.state["prompt"] = []

        # frame.text and self._session.ctx.state["prompt"].append(frame.text)

        if frame.text:
            yield TextFrame(text=f"{frame.text}")

        if frame.images:
            for image_frame in frame.images:
                yield TextFrame(text=f"{image_frame}")

        if frame.audio and frame.audio.audio:
            yield AudioRawFrame(
                audio=frame.audio.audio,
                sample_rate=frame.audio.sample_rate,
            )
