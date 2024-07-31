import argparse
import asyncio
import json
import logging
import os

from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.daily_input_transport_processor import DailyInputTransportProcessor
from src.processors.rtvi_processor import RTVIConfig, RTVIProcessor, RTVISetup
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams, DailyRoomBotArgs
from src.services.daily.client import DailyTransport
from .base import DailyRoomBot, register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)


@register_daily_room_bots.register
class DailyRTVIBot(DailyRoomBot):
    def __init__(self, **args) -> None:
        super().__init__(**args)
        try:
            logging.debug(f'config: {self.args.bot_config}')
            self._bot_config: RTVIConfig = RTVIConfig(**self.args.bot_config)
        except Exception as e:
            raise Exception("Failed to parse bot configuration")

    def bot_config(self):
        return self._bot_config.model_dump()

    async def _run(self):
        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            ))
        input_processor = DailyInputTransportProcessor(
            transport.client, transport.params)

        rtai = RTVIProcessor(
            transport=transport,
            setup=RTVISetup(config=self._bot_config),
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

        transport.add_event_handler(
            "on_first_participant_joined",
            self.on_first_participant_joined)
        transport.add_event_handler(
            "on_participant_left",
            self.on_participant_left)
        transport.add_event_handler(
            "on_call_state_updated",
            self.on_call_state_updated)

        await PipelineRunner().run(self.task)

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


r"""
python -m src.cmd.bots.daily_rtvi_bot -u https://weedge.daily.co/DummyBot   -c $'{"llm":{"model":"llama-3.1-8b-instant","messages":[{"role":"system","content":"You are Chatbot, a friendly, helpful robot. Your output will be converted to audio so don\'t include special characters other than \'\u0021\' or \'?\' in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by saying hello."}]},"tts":{"voice":"79a125e8-cd45-4c13-8a67-188112f4dd22"}}' -t eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyIjoiRHVtbXlCb3QiLCJvIjp0cnVlLCJleHAiOjE3MjIzNTQ1MjUsImQiOiIyZjE5OGNlNC02NzIwLTQwMzEtYTQ1Ny05ODBkNTJlODhiNzgiLCJpYXQiOjE3MjIzNTI3MjZ9.3Q4BAoOCouxVQTTRgvbl3BhwWqj70nBwVWDSOJvT2CQ
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTVI Bot Example")
    parser.add_argument("-u", type=str, default="https://weedge.daily.co/chat-bot", help="Room URL")
    parser.add_argument("-t", type=str, default="", help="Token")
    parser.add_argument("-c", type=str, help="Bot configuration blob")
    config = parser.parse_args()

    bot_config = json.loads(config.c) if config.c else {}

    if config.u and config.t and bot_config:
        kwargs = DailyRoomBotArgs(
            bot_config=bot_config,
            room_url=config.u,
            token=config.t,
        ).__dict__
        bot = DailyRTVIBot(**kwargs)
        bot.run()
    else:
        logging.error("Room URL and Token are required")
