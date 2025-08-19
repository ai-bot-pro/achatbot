import io
import logging
from typing import AsyncGenerator
import uuid

from apipeline.frames import StartFrame
from apipeline.frames.data_frames import Frame
from apipeline.frames.sys_frames import ErrorFrame

from src.common.factory import EngineClass
from src.common.session import Session, SessionCtx
from src.common.utils.time import time_now_iso8601
from src.common.utils.audio_utils import (
    bytes2NpArrayWith16,
    convertSampleRateTo16khz,
    read_wav_to_np,
)
from src.common.interface import IAsrLive
from src.processors.speech.asr.base import UserStartedSpeakingFrame, VADSegmentedASRProcessor
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
        super().__init__(**kwargs)
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

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        await super()._handle_user_started_speaking(frame)
        self._asr.reset()

    async def run_asr(self, audio: bytes, **kwargs) -> AsyncGenerator[Frame, None]:
        if self._asr is None:
            logging.error(f"{self} error: ASR engine not available")
            yield ErrorFrame("ASR engine not available")
            return

        self._session.ctx.state["is_last"] = kwargs.get("is_last", False)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        language = None
        args = self._asr.get_args_dict()
        if "language" in args:
            language = Language(args["language"])

        # print(len(audio), audio[:45])
        if io.BytesIO(audio).read(4) == b"RIFF":  # head len: 44
            audio_np, _ = read_wav_to_np(io.BytesIO(audio))
            # audio_np = bytes2NpArrayWith16(audio[44:])
        else:
            audio_np = bytes2NpArrayWith16(audio)
        self._session.ctx.state["audio_chunk"] = audio_np
        i = 0
        async for segment in self._asr.streaming_transcribe(self._session):
            if i == 0:  # for first chunk transcription cost time
                await self.stop_ttfb_metrics()
            i += 1
            text = segment.get("text")
            if text:
                logging.info(f"{self._asr.SELECTED_TAG} Transcription: [{text}]")
                yield TranscriptionFrame(
                    text, "", time_now_iso8601(), language, segment.get("timestamps", [])
                )

        await self.stop_processing_metrics()


"""
python -m src.processors.speech.asr.asr_live_processor
TEXTNORM=1 python -m src.processors.speech.asr.asr_live_processor
"""
if __name__ == "__main__":
    import os
    import asyncio

    from src.common.time_utils import to_timestamp
    from src.modules.speech.asr_live import ASRLiveEnvInit
    from src.common.session import Session, SessionCtx
    from src.common.types import ASSETS_DIR
    from src.common.utils.wav import read_wav_to_bytes

    async def run():
        engine = ASRLiveEnvInit.initEngine(textnorm=bool(os.getenv("TEXTNORM", "")))
        session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
        processor = ASRLiveProcessor(
            asr=engine, session=session, sample_rate=16000, chunk_length_seconds=0.1
        )
        processor.start(frame=StartFrame())

        sr = 24000
        wav_path = os.path.join(ASSETS_DIR, "Chinese_prompt.wav")
        print(wav_path)
        audio_bytes, sr = read_wav_to_bytes(wav_path)
        print(len(audio_bytes), audio_bytes[:10], sr)

        target_sample_rate = 16000
        if sr != target_sample_rate:
            audio_bytes = convertSampleRateTo16khz(audio_bytes, original_sample_rate=sr)
            sr = target_sample_rate
        print(len(audio_bytes), audio_bytes[:10], sr)

        pre_len = 0
        step = int(0.1 * sr * 2)
        start_times = []
        result = {}
        for i in range(0, len(audio_bytes), step):
            audio = audio_bytes[i : i + step]
            is_start = i == 0
            is_last = i + step >= len(audio_bytes)
            async for res in processor.run_asr(audio, is_start=is_start, is_last=is_last):
                assert isinstance(res, TranscriptionFrame)
                print(res)
                for timestamp in res.timestamps[pre_len:]:
                    start_times.append(to_timestamp(timestamp, msec=1))
                result = res.__dict__
                result["start_times"] = start_times
                pre_len = len(res.timestamps)
        print(result)

    asyncio.run(run())
