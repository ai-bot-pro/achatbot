import asyncio
import os
import logging

import unittest
from livekit import rtc
from apipeline.pipeline.pipeline import Pipeline
from apipeline.processors.logger import FrameLogger
from apipeline.pipeline.runner import PipelineRunner
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.frames.control_frames import EndFrame
from apipeline.frames.data_frames import TextFrame, DataFrame, AudioRawFrame, ImageRawFrame
from apipeline.processors.output_processor import OutputFrameProcessor

from src.modules.speech.asr import ASREnvInit
from src.processors.speech.asr.asr_processor import ASRProcessor
from src.services.help.livekit_room import LivekitRoom
from src.services.help import RoomManagerEnvInit
from src.common.types import LivekitParams
from src.common.logger import Logger
from src.transports.livekit import LivekitTransport
from src.types.frames.data_frames import TranscriptionFrame

from dotenv import load_dotenv

load_dotenv(override=True)

r"""
LIVEKIT_BOT_NAME=chat-bot \
    python -m unittest test.integration.processors.test_livekit_asr_processor.TestASRProcessor

LIVEKIT_BOT_NAME=chat-bot  \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    ASR_VERBOSE=True \
    ASR_TAG=sense_voice_asr
    python -m unittest test.integration.processors.test_livekit_asr_processor.TestASRProcessor
"""


class TestASRProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)
        cls.room_name = os.getenv("ROOM_NAME", "chat-room")

    @classmethod
    def tearDownClass(cls):
        pass

    async def asyncSetUp(self):
        self.texts = []

        self.livekit_room: LivekitRoom = RoomManagerEnvInit.initEngine("livekit_room")
        self.asr = ASREnvInit.initASREngine()

    async def asyncTearDown(self):
        await self.livekit_room.close_session()

    async def test_tts_daily_output(self):
        token = await self.livekit_room.gen_token(self.room_name)
        transport = LivekitTransport(
            token,
            params=LivekitParams(
                audio_in_enabled=True,
                audio_in_sample_rate=48000,
            ),
        )
        self.assertGreater(len(transport.event_names), 0)

        asr_processor = ASRProcessor(asr=self.asr)
        out_processor = OutputFrameProcessor(cb=self.sink_callback)

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    # FrameLogger(include_frame_types=[AudioRawFrame]),
                    asr_processor,
                    FrameLogger(include_frame_types=[TranscriptionFrame]),
                    out_processor,
                ]
            ),
            params=PipelineParams(allow_interruptions=True),
        )

        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_disconnected", self.on_participant_disconnected)
        transport.add_event_handler("on_disconnected", self.on_disconnected)
        # sometime don't get on_connection_state_changed when disconnected
        transport.add_event_handler("on_connection_state_changed", self.on_connection_state_changed)

        runner = PipelineRunner()
        try:
            await asyncio.wait_for(runner.run(self.task), int(os.getenv("RUN_TIMEOUT", "60")))
        except asyncio.TimeoutError:
            logging.warning("Test run timeout. Exiting")
            await self.task.queue_frame(EndFrame())

    async def sink_callback(self, frame: DataFrame):
        print(f"sink_callback ----> print frame: {frame}")
        if not isinstance(frame, TextFrame):
            return

        self.texts.append(frame.text)
        if len(self.texts) == 2:  # asr 2 times (speech to text)
            print("sink_callback ----> send end frame")
            await self.task.queue_frame(EndFrame())
            return

    async def on_first_participant_joined(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.debug(f"transport---->{transport}")
        logging.debug(f"participant---->{participant}")
        participant_name = participant.name if participant.name else participant.identity
        await transport.send_message(
            f"hello,你好，{participant_name}, 我是机器人。",
            participant_id=participant.identity,
        )

    async def on_participant_disconnected(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        logging.info(f"Partcipant {participant} left. Exiting.")

    async def on_disconnected(self, transport: LivekitTransport, reason: str):
        logging.info("disconnected reason %s, Exiting." % reason)
        await self.task.queue_frame(EndFrame())

    async def on_connection_state_changed(
        self, transport: LivekitTransport, state: rtc.ConnectionState
    ):
        logging.info("connection state %s " % state)
        if state == rtc.ConnectionState.CONN_DISCONNECTED:
            await self.task.queue_frame(EndFrame())
