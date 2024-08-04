import argparse
import asyncio
import json
import logging
import os

from apipeline.frames.control_frames import EndFrame
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.processors.llm.openai_llm_processor import OpenAILLMProcessor
from src.processors.speech.tts.cartesia_tts_processor import CartesiaTTSProcessor
from src.processors.rtvi_processor import RTVIConfig, RTVIProcessor, RTVISetup
from src.processors.speech.asr.asr_processor import AsrProcessor
from src.modules.speech.vad_analyzer.silero import SileroVADAnalyzer
from src.common.types import DailyParams, DailyRoomBotArgs, DailyTranscriptionSettings
from src.transports.daily import DailyTransport
from src.common.session import Session
from src.common.types import SessionCtx
from src.cmd.init import Env
from .base import DailyRoomBot, register_daily_room_bots

from dotenv import load_dotenv
load_dotenv(override=True)


@register_daily_room_bots.register
class DailyAsrRTVIBot(DailyRoomBot):
    """
    !NOTE: just for Chinese(zh) chat bot
    """

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
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
                transcription_enabled=False,
            ))

        # !NOTE: u can config env in .env file
        asr = Env.initASREngine()
        self.session = Session(**SessionCtx("").__dict__)
        asr_processor = AsrProcessor(asr=asr, session=self.session)

        # !TODO: need config processor with bot config (redefine api params) @weedge
        # bot config: Dict[str, Dict[str,Any]]
        # e.g. {"llm":{"key":val,"tag":TAG}, "tts":{"key":val,"tag":TAG}}
        llm_processor = OpenAILLMProcessor(
            model=self._bot_config.llm.model,
            base_url="https://api.groq.com/openai/v1",
        )
        # https://docs.cartesia.ai/getting-started/available-models
        # !NOTE: Timestamps are not supported for language 'zh'
        tts_processor = CartesiaTTSProcessor(
            voice_id=self._bot_config.tts.voice,
            cartesia_version="2024-06-10",
            model_id="sonic-multilingual",
            language="zh",
        )

        rtai = RTVIProcessor(
            transport=transport,
            setup=RTVISetup(config=self._bot_config),
            llm_processor=llm_processor,
            tts_processor=tts_processor,
        )

        self.task = PipelineTask(
            Pipeline([
                transport.input_processor(),
                asr_processor,
                rtai,
            ]),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

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
        self.session.set_client_id(participant['id'])
        logging.info(f"First participant {participant['id']} joined")

    async def on_participant_left(self, transport, participant, reason):
        await self.task.queue_frame(EndFrame())
        logging.info("Partcipant left. Exiting.")

    async def on_call_state_updated(self, transport, state):
        logging.info("Call state %s " % state)
        if state == "left":
            await self.task.queue_frame(EndFrame())


r"""
python -m src.cmd.bots.daily_rtvi_bot -u https://weedge.daily.co/chat-room -c $'{"llm":{"model":"llama-3.1-8b-instant","messages":[{"role":"system","content":"你是一位很有帮助中文AI助理机器人。你的目标是用简洁的方式展示你的能力,请用中文简短回答，回答限制在1-5句话内。你的输出将转换为音频，所以不要在你的答案中包含特殊字符。以创造性和有帮助的方式回应用户说的话。"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}'
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTVI Bot Example")
    parser.add_argument("-u", type=str, default="https://weedge.daily.co/chat-bot", help="Room URL")
    parser.add_argument("-t", type=str, default="", help="Token")
    parser.add_argument("-c", type=str, help="Bot configuration blob")
    config = parser.parse_args()

    bot_config = json.loads(config.c) if config.c else {}

    if config.u and bot_config:
        kwargs = DailyRoomBotArgs(
            bot_config=bot_config,
            room_url=config.u,
            token=config.t,
        ).__dict__
        bot = DailyAsrRTVIBot(**kwargs)
        bot.run()
    else:
        logging.error("Room URL and Token are required")
