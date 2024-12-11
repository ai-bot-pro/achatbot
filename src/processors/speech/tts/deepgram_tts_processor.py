import logging
from typing import AsyncGenerator

import aiohttp
from apipeline.frames.sys_frames import ErrorFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.processors.speech.tts.base import TTSProcessorBase
from types.frames.control_frames import TTSStartedFrame


class DeepgramTTSProcessor(TTSProcessorBase):
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-helios-en",
        base_url: str = "https://api.deepgram.com/v1/speak",
        sample_rate: int = 16000,
        encoding: str = "linear16",
        aiohttp_session: aiohttp.ClientSession | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._base_url = base_url
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._aiohttp_session = aiohttp_session or aiohttp.ClientSession()
        self._close_aiohttp_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        return True

    async def cleanup(self):
        await super().cleanup()
        if self._close_aiohttp_session:
            await self._aiohttp_session.close()

    async def set_voice(self, voice: str):
        logging.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating TTS: [{text}]")

        base_url = self._base_url
        request_url = f"{base_url}?model={self._voice}&encoding={self._encoding}&container=none&sample_rate={self._sample_rate}"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}

        try:
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
                if r.status != 200:
                    response_text = await r.text()
                    # If we get a a "Bad Request: Input is unutterable", just print out a debug log.
                    # All other unsuccesful requests should emit an error frame. If not specifically
                    # handled by the running PipelineTask, the ErrorFrame will cancel the task.
                    if "unutterable" in response_text:
                        logging.debug(f"Unutterable text: [{text}]")
                        return

                    logging.error(
                        f"{self} error getting audio (status: {r.status}, error: {response_text})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status}, error: {response_text})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=self._sample_rate, num_channels=1)
                    yield frame
        except Exception as e:
            logging.exception(f"{self} exception: {e}")
        finally:
            self._tts_done_event.set()
