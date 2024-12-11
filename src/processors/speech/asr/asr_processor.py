import logging
from typing import AsyncGenerator

from apipeline.frames.data_frames import Frame
from apipeline.frames.sys_frames import ErrorFrame

from src.common.factory import EngineClass
from src.common.session import Session
from src.common.utils.time import time_now_iso8601
from src.common.interface import IAsr
from src.processors.speech.asr.base import SegmentedASRProcessor
from src.types.frames.data_frames import TranscriptionFrame
from src.types.speech.language import Language


class ASRProcessor(SegmentedASRProcessor):
    def __init__(
        self,
        *,
        min_volume: float = 0.6,
        max_silence_secs: float = 0.3,
        max_buffer_secs: float = 1.5,
        sample_rate: int = 16000,
        num_channels: int = 1,
        asr: IAsr | EngineClass | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(
            min_volume=min_volume,
            max_silence_secs=max_silence_secs,
            max_buffer_secs=max_buffer_secs,
            sample_rate=sample_rate,
            num_channels=num_channels,
            **kwargs,
        )
        self._asr = asr
        self._session = session

    def set_asr(self, asr: IAsr):
        self._asr = asr

    async def set_asr_args(self, **args):
        self._asr.set_args(**args)

    async def run_asr(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._asr is None:
            logging.error(f"{self} error: ASR engine not available")
            yield ErrorFrame("ASR engine not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        self._asr.set_audio_data(audio)
        text: str = ""
        async for segment in self._asr.transcribe_stream(self._session):
            text += f"{segment}"

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        language = None
        args = self._asr.get_args_dict()
        if "language" in args:
            language = Language(args["language"])

        if text:
            logging.info(f"{self._asr.SELECTED_TAG} Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601(), language)
