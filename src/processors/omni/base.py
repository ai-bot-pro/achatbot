import logging
from abc import abstractmethod
from typing import AsyncGenerator
import uuid

from apipeline.pipeline.pipeline import FrameDirection
from apipeline.frames import *
import numpy as np

from src.common.factory import EngineClass
from src.common.interface import ILlm
from src.common.session import Session
from src.common.types import CHANNELS, RATE, SessionCtx
from src.types.frames.data_frames import Frame, VisionImageVoiceRawFrame
from src.processors.ai_processor import AsyncAIProcessor


class VisionVoiceProcessorBase(AsyncAIProcessor):
    """
    VisionVoiceProcessorBase is a base class for vision+voice processors.
    j
    input: vision + voice frame
    use omni lm to process vision + voice frames
    output: text+audio frame
    """

    def __init__(
        self,
        llm: ILlm | EngineClass | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = llm
        self._session = session
        if self._session is None:
            self._session = Session(**SessionCtx(uuid.uuid4()).__dict__)

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {"sample_rate": RATE, "channels": CHANNELS}

    @abstractmethod
    async def run(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield frame

    async def say(self, text: str):
        pass

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
        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    async def gen(self) -> AsyncGenerator[Frame, None]:
        """
        - gen tensor audio streamer
        = push text, audio frame
        """
        tensor_audio_stream = self._llm.generate(self._session)
        for item in tensor_audio_stream:
            logging.debug(f"generate data: {item}")
            tensor_audio = item.pop("audio_wav", None)
            rate = item.pop("sampling_rate", RATE)
            text = item.pop("text", "").strip()
            if text != "":
                await self.push_frame(TextFrame(text=text))

            if tensor_audio is not None:  # don't use if tensor_audio to check
                audio_bytes = (
                    (tensor_audio.float().detach().cpu().numpy() * 32768).astype(np.int16).tobytes()
                )
                logging.debug(
                    f"audio tensor:{tensor_audio.shape},push audio len:{len(audio_bytes)}"
                )
                await self.queue_frame(
                    AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=rate,
                    )
                )
            yield None
