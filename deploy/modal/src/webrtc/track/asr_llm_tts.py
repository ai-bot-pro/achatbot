import asyncio
import fractions
from pathlib import Path
import traceback
from typing import Union

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame, AudioResampler
from av.frame import Frame
from av.packet import Packet

from apipeline.frames.control_frames import EndFrame, StartFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames.data_frames import AudioRawFrame
from apipeline.processors.logger import FrameLogger
from apipeline.processors.output_processor import OutputFrameProcessor


from achatbot.cmd.bots.base import AIBot
from achatbot.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from achatbot.processors.audio_input_processor import AudioVADInputProcessor
from achatbot.common.types import AudioVADParams
from achatbot.processors.speech.asr.base import ASRProcessorBase
from achatbot.processors.llm.base import LLMProcessor
from achatbot.processors.speech.tts.tts_processor import TTSProcessor
from achatbot.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)

from .base import BaseTrack


defualt_bot_config = {
    "vad": {"tag": "silero_vad_analyzer", "args": {"stop_secs": 0.7}},
    "asr": {
        "tag": "sense_voice_asr",
        "args": {
            "language": "zn",
            "model_name_or_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
        },
    },
    "llm": {
        "tag": "openai_llm_processor",
        "base_url": "https://api.together.xyz/v1",
        "model": "Qwen/Qwen2-72B-Instruct",
        "language": "zh",
        "messages": [
            {
                "role": "system",
                "content": "你是一名叫奥利给的智能助理。保持回答简短和清晰。请用中文回答。",
            }
        ],
    },
    "tts": {
        "tag": "tts_edge",
        "args": {"voice_name": "zh-CN-YunjianNeural", "language": "zh", "gender": "Male"},
    },
}

ai_bot = None
vad_analyzer = None
asr_processor = None
llm_processor = None
tts_processor = None


def load(kwargs=None):
    if kwargs is None:
        kwargs={"bot_config": defualt_bot_config}
    print(f"asr_llm_tts init load with kwargs: {kwargs}")
    global ai_bot, vad_analyzer, asr_processor, llm_processor, tts_processor
    if ai_bot is not None:
        print(f"{ai_bot} already loaded")
        return
    ai_bot = AIBot(**kwargs)
    ai_bot.init_bot_config()
    vad_analyzer = VADAnalyzerEnvInit.initVADAnalyzerEngine()
    asr_processor = ai_bot.get_asr_processor()
    llm_processor = ai_bot.get_llm_processor()
    tts_processor = ai_bot.get_tts_processor()
    print(f"asr_llm_tts init load DONE")


class AsrLlmTtsTrack(BaseTrack):
    """
    audio input -> asr -> llm -> tts -> output(audio frame)
    """

    def __init__(self, in_track: MediaStreamTrack = None) -> None:
        super().__init__(in_track)

        try:
            self.init()
        except Exception as e:
            print(f"Error initializing AsrLlmTtsTrack: {e}")

    def init(self):
        self._timestamp = 0
        self._in_sample_rate = 16000
        self._samples_per_10ms = self._in_sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit (2 bytes per sample)
        self._resampler = AudioResampler("s16", "mono", self._in_sample_rate)

        self.audio_input_processor = AudioVADInputProcessor(
            params=AudioVADParams(
                audio_in_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=vad_analyzer,
            )
        )
        self.stream_info = tts_processor.get_stream_info()

        messages = []
        if ai_bot.bot_config().llm.messages:
            messages = ai_bot.bot_config().llm.messages
        user_response = LLMUserResponseAggregator(messages)
        assistant_response = LLMAssistantResponseAggregator(messages)

        self.out_processor = OutputFrameProcessor()

        self.task = PipelineTask(
            Pipeline(
                [
                    FrameLogger(include_frame_types=[StartFrame]),
                    self.audio_input_processor,
                    asr_processor,
                    user_response,
                    llm_processor,
                    tts_processor,
                    FrameLogger(include_frame_types=[AudioRawFrame]),
                    self.out_processor,
                    assistant_response,
                ]
            ),
            params=PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
                send_initial_empty_metrics=False,
            ),
        )
        self.runner = PipelineRunner()
        print("========= init DONE =============")

    async def run(self):
        await self.runner.run(self.task)

    # this is the essential method we need to implement
    # to create a custom MediaStreamTrack
    async def recv(self) -> Union[Frame, Packet]:
        assert self.in_track is not None
        try:
            frame = await self.in_track.recv()
            if self.in_track.kind == "audio":
                # in
                in_audio_frame: AudioFrame = frame
                if in_audio_frame.sample_rate > self._in_sample_rate:
                    resampled_frames = self._resampler.resample(frame)
                    for resampled_frame in resampled_frames:
                        # 16-bit PCM bytes
                        pcm_bytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                        await self.audio_input_processor.push_audio_frame(
                            frame=AudioRawFrame(
                                audio=pcm_bytes,
                                sample_rate=self._in_sample_rate,
                            )
                        )
                # out
                try:
                    frame_out = await asyncio.wait_for(self.out_processor.out_queue.get(), 0.01)
                    chunk = bytes(self._bytes_per_10ms)  # silence
                    if isinstance(frame_out, AudioRawFrame):
                        chunk = frame_out.audio
                    self.out_processor.out_queue.task_done()
                    self.out_processor.set_sink_event()
                except asyncio.TimeoutError:
                    chunk = bytes(self._bytes_per_10ms)  # silence
                # Convert the byte data to an ndarray of int16 samples
                samples = np.frombuffer(chunk, dtype=np.int16)

                # Create AudioFrame
                frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
                frame.sample_rate = self.stream_info["sample_rate"]
                frame.pts = self._timestamp
                frame.time_base = fractions.Fraction(1, self.stream_info["sample_rate"])
                self._timestamp += self._samples_per_10ms

            return frame
        except Exception as e:
            print(f"Error processing audio: {e} {traceback.format_exc()}")
