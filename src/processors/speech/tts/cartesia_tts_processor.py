import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncGenerator
import uuid


try:
    import websockets
except ModuleNotFoundError as e:
    logging.error(
        "In order to use Cartesia, you need to `pip install websockets`. Also, set `CARTESIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
from apipeline.pipeline.pipeline import FrameDirection
from apipeline.frames.data_frames import TextFrame, Frame, AudioRawFrame
from apipeline.frames.sys_frames import StartInterruptionFrame, CancelFrame
from apipeline.frames.control_frames import StartFrame, EndFrame

from src.processors.speech.tts.base import TTSProcessorBase
from src.types.frames.control_frames import LLMFullResponseEndFrame

# https://docs.cartesia.ai/getting-started/available-models
# !NOTE: Timestamps are not supported for language 'zh'


class CartesiaTTSProcessor(TTSProcessorBase):
    TAG = "cartesia_tts_processor"

    def __init__(
        self,
        voice_id: str = "2ee87190-8f84-4925-97da-e52547f9462c",
        api_key: str = "",
        cartesia_version: str = "2024-06-10",
        url: str = "wss://api.cartesia.ai/tts/websocket",
        model_id: str = "sonic-multilingual",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for
        # a full sentence should only "cost" us 15ms or so with GPT-4o or a Llama 3
        # model, and it's worth it for the better audio quality.
        self._aggregate_sentences = True

        # we don't want to automatically push LLM response text frames, because the
        # context aggregators will add them to the LLM context even if we're
        # interrupted. cartesia gives us word-by-word timestamps. we can use those
        # to generate text frames ourselves aligned with the playout timing of the audio!
        self._push_text_frames = False

        api_key = os.getenv("CARTESIA_API_KEY", api_key)
        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self._language = language

        self._websocket = None
        self._context_id = None
        self._context_id_start_timestamp = None
        self._timestamped_words_buffer = []
        self._receive_task = None
        self._context_appending_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logging.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()
        logging.info("cancel done")

    async def _connect(self):
        try:
            uri = f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
            self._websocket = await websockets.connect(uri)
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            self._context_appending_task = self.get_event_loop().create_task(
                self._context_appending_task_handler()
            )
        except Exception as e:
            logging.exception(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()
            if self._context_appending_task:
                logging.info("context_appending_task.cancel......")
                self._context_appending_task.cancel()
                await self._context_appending_task
                self._context_appending_task = None
            if self._websocket:
                logging.info("websocket close......")
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                logging.info("receive_task.cancel......")
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None
            self._context_id = None
            self._context_id_start_timestamp = None
            self._timestamped_words_buffer = []
        except Exception as e:
            logging.exception(f"{self} error closing websocket: {e}")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        self._context_id = None
        self._context_id_start_timestamp = None
        self._timestamped_words_buffer = []
        await self.stop_all_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def _receive_task_handler(self):
        try:
            while True:
                async for message in self._websocket:
                    msg = json.loads(message)
                    # logging.info(f"Received message: {msg['type']} {msg['context_id']}")
                    if not msg or msg["context_id"] != self._context_id:
                        continue
                    if msg["type"] == "error":
                        logging.error(f"Received message error msg: {msg}")
                        self._tts_done_event.set()
                    elif msg["type"] == "done":
                        await self.stop_ttfb_metrics()
                        # unset _context_id but not the _context_id_start_timestamp
                        # because we are likely still playing out audio
                        # and need the timestamp to set send context frames
                        self._context_id = None
                        self._timestamped_words_buffer.append(("LLMFullResponseEndFrame", 0))
                        self._tts_done_event.set()
                    elif msg["type"] == "timestamps":
                        # logging.debug(f"TIMESTAMPS: {msg}")
                        self._timestamped_words_buffer.extend(
                            list(
                                zip(msg["word_timestamps"]["words"], msg["word_timestamps"]["end"])
                            )
                        )
                    elif msg["type"] == "chunk":
                        await self.stop_ttfb_metrics()
                        if not self._context_id_start_timestamp:
                            self._context_id_start_timestamp = time.time()
                        frame = AudioRawFrame(
                            audio=base64.b64decode(msg["data"]),
                            sample_rate=self._output_format["sample_rate"],
                            num_channels=1,
                        )
                        await self.push_frame(frame)
        except asyncio.CancelledError:
            self._tts_done_event.set()
        except Exception as e:
            logging.exception(f"{self} exception: {e}")
            self._tts_done_event.set()

    async def _context_appending_task_handler(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                if not self._context_id_start_timestamp:
                    continue
                elapsed_seconds = time.time() - self._context_id_start_timestamp
                # pop all words from self._timestamped_words_buffer that are older than the
                # elapsed time and print a message about them to the console
                while (
                    self._timestamped_words_buffer
                    and self._timestamped_words_buffer[0][1] <= elapsed_seconds
                ):
                    word, timestamp = self._timestamped_words_buffer.pop(0)
                    if word == "LLMFullResponseEndFrame" and timestamp == 0:
                        await self.push_frame(LLMFullResponseEndFrame())
                        continue
                    # print(f"Word '{word}' with timestamp {timestamp:.2f}s has been spoken.")
                    await self.push_frame(TextFrame(word))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.exception(f"{self} exception: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logging.info(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            if not self._context_id:
                await self.start_ttfb_metrics()
                self._context_id = str(uuid.uuid4())

            msg = {
                "transcript": text + " ",
                "continue": True,
                "context_id": self._context_id,
                "model_id": self._model_id,
                "voice": {
                    "mode": "id",
                    "id": self._voice_id,
                    "__experimental_controls": {
                        "speed": "normal",
                    },
                },
                "output_format": self._output_format,
                "language": self._language,
                "add_timestamps": True,
            }
            if self._language == "zh":
                msg["add_timestamps"] = False
            # logging.info(f"SENDING MESSAGE {json.dumps(msg)}")
            try:
                await self._websocket.send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logging.exception(f"{self} error sending message: {e}")
                await self._disconnect()
                await self._connect()
                self._tts_done_event.set()
                return
            yield None
        except Exception as e:
            logging.exception(f"{self} exception: {e}")
            self._tts_done_event.set()
