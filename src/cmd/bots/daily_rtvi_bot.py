import argparse
import asyncio
import json
import logging
import os

from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask

from src.processors.daily_input_transport_processor import DailyInputTransportProcessor
from src.processors.rtvi_processor import RTVIConfig, RTVIProcessor, RTVISetup
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams
from src.services.daily.client import DailyTransport
from .base import BaseBot, register_rtvi_bots


@register_rtvi_bots.register
class DailyRTVIBot(BaseBot):
    def __init__(self, room_url, token, bot_config, bot_name=None, **kwargs) -> None:
        super().__init__(room_url, token, bot_config, bot_name, **kwargs)
        if bot_name is None or len(bot_name) == 0:
            bot_name = self.__class__.__name__
        self.transport = DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            ))
        input_processor = DailyInputTransportProcessor(
            self.transport.client, self.transport.params)

        rtai = RTVIProcessor(
            transport=self.transport,
            setup=RTVISetup(config=RTVIConfig(**bot_config)),
            llm_api_key=os.getenv("OPENAI_API_KEY", ""),
            tts_api_key=os.getenv("CARTESIA_API_KEY", ""))

        pipeline = Pipeline([input_processor, rtai])

        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ))
        self.transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined)
        self.transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        self.transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

    async def on_first_participant_joined(self, transport, participant):
        transport.capture_participant_transcription(participant["id"])
        logging.info("First participant joined")

    async def on_participant_left(self, transport, participant, reason):
        await self.task.queue_frame(EndFrame())
        logging.info("Partcipant left. Exiting.")

    async def on_call_state_updated(self, transport, state):
        logging.info("Call state %s " % state)
        if state == "left":
            await self.task.queue_frame(EndFrame())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTVI Bot Example")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-c", type=str, help="Bot configuration blob")
    config = parser.parse_args()

    bot_config = json.loads(config.c) if config.c else {}

    if config.u and config.t and bot_config:
        bot = DailyRTVIBot(config.u, config.t, bot_config)
        asyncio.run(bot.run())
    else:
        logging.error("Room URL and Token are required")
