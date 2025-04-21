import asyncio
import io
import uuid
import logging
from typing import AsyncGenerator

import librosa
import numpy as np
from apipeline.frames import *

from src.core.llm.transformers.manual_vision_voice_qwen import (
    TransformersManualQwen2_5OmniLLM,
    TransformersManualVoiceQwen2_5OmniLLM,
    TransformersManualTextVoiceQwen2_5OmniLLM,
)

from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
)
from src.types.frames import PathAudioRawFrame


class Qwen2_5OmniVoiceProcessor(VoiceProcessorBase):
    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._model: TransformersManualQwen2_5OmniLLM = None

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": TransformersManualQwen2_5OmniLLM.RATE,
            "channels": 1,
        }

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
        tensor_audio_stream = self._model.generate(self._session)
        for item in tensor_audio_stream:
            logging.debug(f"generate data: {item}")
            tensor_audio = item.pop("audio_wav", None)
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
                        sample_rate=self._model.RATE,
                    )
                )
            yield None


class Qwen2_5OmniAudioVoiceProcessor(Qwen2_5OmniVoiceProcessor):
    """
    qwen2.5omni voice
    - A1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)

        self._model = TransformersManualVoiceQwen2_5OmniLLM(**kwargs)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_nparr, _ = librosa.load(frame.path, sr=16000, mono=True)
        else:
            audio_nparr = bytes2NpArrayWith16(frame.audio)

        self._session.ctx.state["prompt"] = [{"type": "audio", "audio": audio_nparr}]
        async for item in self.gen():
            yield item


class Qwen2_5OmniTextVoiceProcessor(Qwen2_5OmniVoiceProcessor):
    """
    - T1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)

        self._model = TransformersManualTextVoiceQwen2_5OmniLLM(**kwargs)

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        self._session.ctx.state["prompt"] = [{"type": "text", "text": user_input}]
        async for item in self.gen():
            yield item
