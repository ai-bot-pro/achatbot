import asyncio
import io
import uuid
import logging
import threading
import queue
from typing import AsyncGenerator

import librosa
import numpy as np
from apipeline.frames import *

from src.core.llm.transformers.manual_voice_kimi import (
    TransformersManualTextVoiceKimiLLM,
    TransformersManualVoiceKimiLLM,
)

from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import (
    bytes2TorchTensorWith16,
)
from src.types.frames import PathAudioRawFrame


class KimiVoiceProcessor(VoiceProcessorBase):
    def __init__(
        self,
        *,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._model: TransformersManualVoiceKimiLLM = None
        self._queue = queue.Queue()
        self._input_queue = queue.Queue()
        self._generate_thread = None
        self._sleep_time = no_stream_sleep_time

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": TransformersManualVoiceKimiLLM.RATE,
            "channels": 1,
        }

    async def say(self, text: str):
        logging.info(f"say: {text}")

    def _generate(self):
        while True:
            try:
                session = self._input_queue.get()
                if session is None:
                    self._queue.put(None)  # Signal the end of the stream
                    break  # Signal to stop the thread
                tensor_audio_stream = self._model.generate(session)
                for item in tensor_audio_stream:
                    self._queue.put(item)
                self._queue.put(None)  # Signal the end of the stream
            except Exception as e:
                logging.error(f"Exception generate: {e}", exc_info=True)
                self._queue.put(None)  # Signal the end of the stream
                break

    async def start(self, frame: StartFrame):
        await super().start(frame)
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
                    logging.debug(
                        f"audio tensor:{tensor_audio.shape},push audio len:{len(audio_bytes)}"
                    )
                    await self.push_frame(
                        AudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self._model.RATE,
                        )
                    )
                yield None
            except queue.Empty:
                # yield asysncio.sleep to allow other tasks to run, e.g.: sink task (write audio)
                await asyncio.sleep(self._sleep_time)
                continue

    def send_input(self, session: Session):
        self._input_queue.put(session)


class KimiAudioVoiceProcessor(KimiVoiceProcessor):
    """
    - A1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(session=session, no_stream_sleep_time=no_stream_sleep_time, **kwargs)

        self._model = TransformersManualVoiceKimiLLM(**kwargs)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            in_audio_wav = frame.path
        else:
            in_audio_wav = bytes2TorchTensorWith16(frame.audio)

        self._session.ctx.state["prompt"] = [
            {"role": "user", "message_type": "audio", "content": in_audio_wav}
        ]
        self.send_input(self._session)
        async for item in self.gen():
            yield item


class KimiTextVoiceProcessor(KimiVoiceProcessor):
    """
    - T1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(session=session, no_stream_sleep_time=no_stream_sleep_time, **kwargs)

        self._model = TransformersManualTextVoiceKimiLLM(**kwargs)

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        self._session.ctx.state["prompt"] = [
            {
                "role": "user",
                "message_type": "text",
                "content": user_input,
            }
        ]
        self.send_input(self._session)
        async for item in self.gen():
            yield item
