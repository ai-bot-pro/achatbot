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

from src.core.llm.transformers.manual_vision_speech_phi import TransformersManualAudioChatPhiLM

from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import RATE, SessionCtx
from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
)
from src.types.frames import PathAudioRawFrame


class Phi4SpeechProcessor(VoiceProcessorBase):
    def __init__(
        self,
        *,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._model: TransformersManualAudioChatPhiLM = None
        self._queue = queue.Queue()
        self._input_queue = queue.Queue()
        self._generate_thread = None
        self._sleep_time = no_stream_sleep_time

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
                            sample_rate=RATE,  # default sample rate, now only support text output, no audio
                        )
                    )
                yield None
            except queue.Empty:
                # yield asysncio.sleep to allow other tasks to run, e.g.: sink task (write audio)
                await asyncio.sleep(self._sleep_time)
                continue

    def send_input(self, session: Session):
        self._input_queue.put(session)


class Phi4AudioTextProcessor(Phi4SpeechProcessor):
    """
    - A1->T2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        no_stream_sleep_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(session=session, no_stream_sleep_time=no_stream_sleep_time, **kwargs)

        self._model = TransformersManualAudioChatPhiLM(**kwargs)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_nparr, _ = librosa.load(frame.path, sr=16000, mono=True)
        else:
            audio_nparr = bytes2NpArrayWith16(frame.audio)
        self._session.ctx.state["prompt"] = [{"type": "audio", "audio": audio_nparr}]
        self.send_input(self._session)
        async for item in self.gen():
            yield item
