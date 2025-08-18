import logging
from typing import AsyncGenerator
import uuid

from apipeline.frames import StartFrame
from apipeline.frames.data_frames import Frame
from apipeline.frames.sys_frames import ErrorFrame

from src.common.factory import EngineClass
from src.common.session import Session, SessionCtx
from src.common.utils.time import time_now_iso8601
from src.common.interface import IAsrLive
from src.processors.speech.asr.base import VADSegmentedASRProcessor
from src.types.frames.data_frames import TranscriptionFrame
from src.types.speech.language import Language


class ASRLiveProcessor(VADSegmentedASRProcessor):
    def __init__(
        self,
        *,
        asr: IAsrLive | EngineClass | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self._asr = asr
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

    def set_asr(self, asr: IAsrLive):
        self._asr = asr

    async def set_asr_args(self, **args):
        """maybe reload model"""
        self._asr.set_args(**args)

    async def set_model(self, model: str):
        """need reload model"""
        pass

    async def set_language(self, language: Language):
        """need reload model"""
        pass

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._asr.reset()

    async def run_asr(self, audio: bytes, **kwargs) -> AsyncGenerator[Frame, None]:
        if self._asr is None:
            logging.error(f"{self} error: ASR engine not available")
            yield ErrorFrame("ASR engine not available")
            return
        is_start = kwargs.get("is_start", False)
        if is_start is True:
            self._asr.reset()

        self._session.ctx.state["is_last"] = kwargs.get("is_last", False)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        language = None
        args = self._asr.get_args_dict()
        if "language" in args:
            language = Language(args["language"])

        i = 0
        self.session.ctx.state["audio"] = audio
        async for segment in self._asr.streaming_transcribe(self.session):
            if i == 0:  # for first chunk transcription cost time
                await self.stop_ttfb_metrics()
            i += 1
            text = segment.get("text")
            if text:
                logging.info(f"{self._asr.SELECTED_TAG} Transcription: [{text}]")
                yield TranscriptionFrame(text, "", time_now_iso8601(), language)

        await self.stop_processing_metrics()
