import os
import logging

import wave
import unittest
from apipeline.pipeline.pipeline import Pipeline, FrameDirection
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from apipeline.frames import TextFrame, AudioRawFrame, EndFrame

from src.processors.voice.glm_voice_processor import GLMAudioVoiceProcessor
from src.common.types import TEST_DIR
from src.common.logger import Logger
from src.types.frames import *

from dotenv import load_dotenv

load_dotenv(override=True)

"""
python -m unittest test.integration.processors.test_glm_voice_processor.TestProcessor
"""


class TestProcessor(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.test_file = os.path.join(TEST_DIR, "audio_files/hi.wav")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        args = {
            "bnb_quant_type": "int4",
            "device": "cuda",
            "lm_gen_args": {
                "max_new_token": 2000,
                "temperature": 0.2,
                "top_p": 0.8
            },
            "model_path": "./models/THUDM/glm-4-voice-9b",
            "torch_dtype": "auto",
            "voice_decoder_path": "./models/THUDM/glm-4-voice-decoder",
            "voice_out_args": {
                "audio_channels": 1,
                "audio_sample_rate": 22050
            },
            "voice_tokenizer_path": "./models/THUDM/glm-4-voice-tokenizer"
        }

        voice_processor = GLMAudioVoiceProcessor(**args)

        pipeline = Pipeline([
            FrameLogger(include_frame_types=[AudioRawFrame]),
            voice_processor,  # output TextFrame and AudioRawFrame
            FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
        ])

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=False,  # close pipeline interruptions, use model interrupt
                enable_metrics=True,
            )
        )

    async def asyncTearDown(self):
        pass

    async def test_run(self):
        runner = PipelineRunner()

        with wave.open(self.test_file, 'rb') as wav_file:
            audio_frame = AudioRawFrame(
                audio=wav_file.readframes(wav_file.getnframes()),
                sample_rate=wav_file.getframerate(),
                num_channels=wav_file.getnchannels(),
                sample_width=wav_file.getsampwidth()
            )
            path_frame = PathAudioRawFrame(
                path=self.test_file,
                audio=wav_file.readframes(wav_file.getnframes()),
                sample_rate=wav_file.getframerate(),
                num_channels=wav_file.getnchannels(),
                sample_width=wav_file.getsampwidth()
            )

        await self.task.queue_frames([
            path_frame,
            # EndFrame(),
        ])
        await runner.run(self.task)
