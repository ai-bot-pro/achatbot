import os
import logging
from typing import AsyncGenerator

from apipeline.frames.sys_frames import CancelFrame
from apipeline.frames.control_frames import EndFrame, StartFrame
from apipeline.frames.data_frames import Frame

from src.processors.speech.asr.base import ASRProcessorBase
from src.common.utils.time import time_now_iso8601
from src.types.frames.data_frames import InterimTranscriptionFrame, TranscriptionFrame
from src.types.speech.language import Language

try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        ListenWebSocketOptions,
        LiveResultResponse,
        AsyncListenWebSocketClient,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Deepgram, you need to `pip install deepgram`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class DeepgramAsrProcessor(ASRProcessorBase):
    def __init__(
        self,
        *,
        api_key: str = "",
        url: str = "",
        language: str = "en",
        model: str = "nova-2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._live_options: ListenWebSocketOptions = ListenWebSocketOptions(
            encoding="linear16",
            language=language,
            model=model,
            sample_rate=16000,
            channels=1,
            interim_results=True,
            smart_format=True,
        )
        api_key = os.getenv("DEEPGRAM_API_KEY", api_key)
        self._client = DeepgramClient(
            api_key, config=DeepgramClientOptions(url=url, options={"keepalive": "true"})
        )
        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)

    async def run_asr(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        await self._connection.send(audio)
        yield None
        await self.stop_processing_metrics()

    async def set_model(self, model: str):
        logging.debug(f"Switching STT model to: [{model}]")
        self._live_options.model = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        logging.debug(f"Switching STT language to: [{language}]")
        self._live_options.language = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        if await self._connection.start(self._live_options):
            logging.info(f"{self}: Connected to Deepgram")
        else:
            logging.error(f"{self}: Unable to connect to Deepgram")

    async def _disconnect(self):
        if self._connection.is_connected:
            await self._connection.finish()
            logging.info(f"{self}: Disconnected from Deepgram")

    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            if is_final:
                logging.info(f"transcript Text: [{transcript}]")
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        "",
                        time_now_iso8601(),
                        language,
                    )
                )
            else:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        "",
                        time_now_iso8601(),
                        language,
                    )
                )
