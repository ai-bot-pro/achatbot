import asyncio
import io
import uuid
import logging
from typing import AsyncGenerator

import librosa
import numpy as np
import soundfile as sf
from apipeline.frames import *

from src.core.llm.transformers.manual_vision_voice_minicpmo import (
    TransformersManualMiniCPMO,
    TransformersManualTextSpeechMiniCPMO,
    TransformersManualVoiceMiniCPMO,
)

from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
)
from src.types.frames import PathAudioRawFrame


class MiniCPMoVoiceProcessor(VoiceProcessorBase):
    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._model: TransformersManualMiniCPMO = None

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": TransformersManualVoiceMiniCPMO.RATE,
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


class MiniCPMoMimickVoiceProcessor(MiniCPMoVoiceProcessor):
    """
    mimick voice
    - A1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)

        kwargs["voice_task"] = "mimick"
        self.mimick_prompt = kwargs.pop(
            "mimick_prompt",
            "Please repeat each user's speech, including voice style and speech content.",
        )
        self._model = TransformersManualVoiceMiniCPMO(**kwargs)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_nparr, _ = librosa.load(frame.path, sr=16000, mono=True)
        else:
            audio_nparr = bytes2NpArrayWith16(frame.audio)

        self._session.ctx.state["prompt"] = [self.mimick_prompt, audio_nparr]
        async for item in self.gen():
            yield item


class MiniCPMoAudioVoiceProcessor(MiniCPMoVoiceProcessor):
    """
    - A1->T2A2
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)

        kwargs["voice_task"] = kwargs.get("voice_task", "audio_assistant")
        if kwargs["voice_task"] not in ["audio_roleplay", "audio_assistant"]:
            raise ValueError("voice_task must be audio_roleplay or audio_assistant")
        self._model = TransformersManualVoiceMiniCPMO(**kwargs)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_nparr, _ = librosa.load(frame.path, sr=16000, mono=True)
        else:
            audio_nparr = bytes2NpArrayWith16(frame.audio)

        self._session.ctx.state["prompt"] = ["", audio_nparr]
        async for item in self.gen():
            yield item


class MiniCPMoTextVoiceProcessor(MiniCPMoVoiceProcessor):
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

        self._model = TransformersManualTextSpeechMiniCPMO(**kwargs)

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        self._session.ctx.state["prompt"] = [user_input]
        async for item in self.gen():
            yield item
