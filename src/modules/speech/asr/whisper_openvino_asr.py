import os
import logging
import asyncio
from typing import AsyncGenerator

from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase


class WhisperOpenVINOAsr(ASRBase):
    """
    - https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html
    - https://github.com/openvinotoolkit/openvino.genai/tree/releases/2025/2/samples/python/whisper_speech_recognition
    """

    TAG = "whisper_openvino_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)
        from openvino import Core
        import openvino_genai as ov_genai

        core = Core()
        available_devices = core.available_devices
        logging.info(f"Available devices: {available_devices}")
        if "GPU" in available_devices:
            selected_device = "GPU"
        else:
            gpu_devices = [d for d in available_devices if d.startswith("GPU")]
            selected_device = gpu_devices[0] if gpu_devices else "CPU"
        self.device = selected_device

        # Initialize the ASR pipeline
        self.pipe = ov_genai.WhisperPipeline(self.args.model_name_or_path, device=self.device)

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        if not self.args.language.startswith("<|"):
            self.args.language = f"<|{self.args.language}|>"

        def streamer(item):
            if "streamer" in session.ctx.state:
                session.ctx.state["streamer"](item)
            else:
                print(item, end="|", flush=True)

        outputs = await asyncio.to_thread(
            self.pipe.generate,
            self.asr_audio if isinstance(self.asr_audio, str) else self.asr_audio.copy(),
            streamer=streamer,
            language=self.args.language,
            task="transcribe",
            return_timestamps=True,
        )
        for item in outputs.chunks:
            yield item.text

    async def transcribe(self, session: Session) -> dict:
        if not self.args.language.startswith("<|"):
            self.args.language = f"<|{self.args.language}|>"
        outputs = await asyncio.to_thread(
            self.pipe.generate,
            self.asr_audio if isinstance(self.asr_audio, str) else self.asr_audio.copy(),
            language=self.args.language,
            task="transcribe",
            return_timestamps=True,
        )
        text_res = ""
        for text in outputs.texts:
            text_res += text.strip()

        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": text_res,
            "words": [
                {"text": item.text, "start": item.start_ts, "end": item.end_ts}
                for item in outputs.chunks
            ],
        }
        return res
