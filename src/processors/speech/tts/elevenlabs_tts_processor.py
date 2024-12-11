import os
import logging
from typing import AsyncGenerator

import aiohttp
from apipeline.frames.sys_frames import ErrorFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.processors.speech.tts.base import TTSProcessorBase


class ElevenLabsTTSProcessor(TTSProcessorBase):
    def __init__(
        self,
        *,
        voice_id: str = "pNInz6obpgDQGcFmaJgB",  # zh: VGcvPRjFP4qKhICQHO7d
        api_key: str = "",
        model_id: str = "eleven_multilingual_v2",
        language: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._api_key = os.getenv("ELEVENLABS_API_KEY", api_key)
        self._voice_id = voice_id
        self._model = model_id
        self._language = language
        self._aiohttp_session = aiohttp_session or aiohttp.ClientSession()
        self._close_aiohttp_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        return True

    async def cleanup(self):
        await super().cleanup()
        if self._close_aiohttp_session:
            await self._aiohttp_session.close()

    async def set_voice(self, voice: str):
        logging.info(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    # https://elevenlabs.io/docs/api-reference/streaming
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logging.info(f"Generating TTS: [{text}]")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

        payload = {
            "text": text,
            "model_id": self._model,
        }
        if self._model == "eleven_turbo_v2_5":
            payload["language_code"] = self._language

        querystring = {
            "output_format": "pcm_16000",
            "optimize_streaming_latency": 2,
        }

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        await self.start_ttfb_metrics()

        async with self._aiohttp_session.post(
            url, json=payload, headers=headers, params=querystring
        ) as r:
            if r.status != 200:
                text = await r.text()
                logging.error(f"{self} error getting audio (status: {r.status}, error: {text})")
                yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                self._tts_done_event.set()
                return

            await self.start_tts_usage_metrics(text)

            async for chunk in r.content:
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
                    yield frame
            self._tts_done_event.set()
