import os
import logging

from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames.sys_frames import CancelFrame, SystemFrame
from apipeline.frames.control_frames import EndFrame, StartFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.common.utils.time import time_now_iso8601
from src.processors.ai_processor import AsyncAIProcessor
from src.types.frames.data_frames import InterimTranscriptionFrame, TranscriptionFrame

try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        ListenWebSocketOptions,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Deepgram, you need to `pip install deepgram`. Also, set `DEEPGRAM_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class DeepgramAsrProcessor(AsyncAIProcessor):
    def __init__(self,
                 *,
                 api_key: str = "",
                 url: str = "",
                 language: str = "en",
                 model: str = "nova-2",
                 **kwargs):
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
            api_key, config=DeepgramClientOptions(url=url, options={"keepalive": "true"}))
        self._connection = self._client.listen.asyncwebsocket.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            await self._connection.send(frame.audio)
        else:
            await self.queue_frame(frame, direction)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if await self._connection.start(options=self._live_options):
            logging.info(f"{self}: Connected to Deepgram")
        else:
            logging.error(f"{self}: Unable to connect to Deepgram")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._connection.finish()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._connection.finish()

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            if is_final:
                logging.info(f'transcript Text: [{transcript}]')
                await self.queue_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))
            else:
                await self.queue_frame(InterimTranscriptionFrame(transcript, "", time_now_iso8601()))
