import logging
from typing import AsyncGenerator
import uuid

from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.common.factory import EngineClass
from src.common.types import SessionCtx
from src.common.session import Session
from src.common.interface import ITts
from src.processors.speech.tts.base import TTSProcessorBase
from src.modules.speech.tts import TTSEnvInit


class TTSProcessor(TTSProcessorBase):
    """This processor uses the TTS API to generate audio from text.
    The returned audio is PCM encoded from tts Interface get_stream_info['rate'] HZ. When using the AudioStream, e.g. set the sample rate in the DailyParams accordingly:
    ```
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=tts.get_stream_info['rate'],
    )
    ```
    """

    def __init__(
        self, *, tts: ITts | EngineClass | None = None, session: Session | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        if tts is None:
            tts = TTSEnvInit.initTTSEngine()
        self._tts: ITts | EngineClass = tts
        if session is None:
            session = Session(**SessionCtx(uuid.uuid4()).__dict__)
        self._session: Session = session

    def can_generate_metrics(self) -> bool:
        return True

    def set_tts(self, tts: ITts):
        self._tts = tts

    async def set_tts_args(self, **args):
        self._tts.set_args(**args)

    async def set_voice(self, voice: str):
        logging.info(f"Switching TTS voice to: [{voice}]")
        self._tts.set_voice(voice)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logging.info(f"Generating TTS: [{text}]")

        self._session.ctx.state["tts_text"] = text
        try:
            stream_info = self._tts.get_stream_info()
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            async for chunk in self._tts.synthesize(self._session):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(
                        audio=chunk,
                        sample_rate=stream_info["rate"],
                        num_channels=stream_info["channels"],
                        sample_width=stream_info["sample_width"],
                    )
                    yield frame
        except Exception as e:
            logging.exception(f"{self} error generating TTS: {e}")
        finally:
            self._tts_done_event.set()

    def get_stream_info(self) -> dict:
        stream_info = self._tts.get_stream_info()
        return {
            "sample_rate": stream_info["rate"],
            "sample_width": stream_info["sample_width"],
            "channels": stream_info["channels"],
        }
