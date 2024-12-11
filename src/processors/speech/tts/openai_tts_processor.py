import logging
from typing import AsyncGenerator, Literal

try:
    from openai import AsyncOpenAI, BadRequestError
except ModuleNotFoundError as e:
    logging.error(
        "In order to use OpenAI, you need to `pip install openai`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
from apipeline.frames.sys_frames import ErrorFrame
from apipeline.frames.data_frames import Frame, AudioRawFrame

from src.processors.speech.tts.base import TTSProcessorBase


class OpenAITTSProcessor(TTSProcessorBase):
    """This processor uses the OpenAI TTS API to generate audio from text.
    The returned audio is PCM encoded at 24kHz. When using the DailyTransport, set the sample rate in the DailyParams accordingly:
    ```
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,
    )
    ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice = voice
        self._model = model

        self._client = AsyncOpenAI(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logging.debug(f"Switching TTS voice to: [{voice}]")
        self._voice = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logging.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self._model,
                voice=self._voice,
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logging.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    self._tts_done_event.set()
                    return
                await self.start_tts_usage_metrics(text)
                async for chunk in r.iter_bytes(8192):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = AudioRawFrame(audio=chunk, sample_rate=24_000, num_channels=1)
                        yield frame
                self._tts_done_event.set()
        except BadRequestError as e:
            logging.exception(f"{self} error generating TTS: {e}")
            self._tts_done_event.set()

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 24_000,
            "sample_width": 2,
            "channels": 1,
        }
