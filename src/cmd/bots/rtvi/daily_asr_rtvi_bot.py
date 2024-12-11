import argparse
import json
import logging

from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner

from src.services.help.daily_rest import DailyRESTHelper
from src.processors.llm.base import LLMProcessor
from src.processors.speech.tts.tts_processor import TTSProcessor
from src.processors.rtvi.rtvi_asr_llm_tts_processor import RTVIProcessor, RTVISetup
from src.common.types import DailyParams, BotRunArgs
from src.transports.daily import DailyTransport
from src.cmd.bots.base_daily import DailyRoomBot
from src.cmd.bots import register_ai_room_bots


@register_ai_room_bots.register
class DailyAsrRTVIBot(DailyRoomBot):
    """
    use asr processor, don't use daily transcirption
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

    def bot_config(self):
        return self._bot_config.model_dump()

    async def arun(self):
        vad_analyzer = self.get_vad_analyzer()
        daily_params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            transcription_enabled=False,
        )

        asr_processor = self.get_asr_processor()

        llm_processor: LLMProcessor = self.get_llm_processor()

        tts_processor: TTSProcessor = self.get_tts_processor()
        stream_info = tts_processor.get_stream_info()
        daily_params.audio_out_sample_rate = stream_info["sample_rate"]
        daily_params.audio_out_channels = stream_info["channels"]

        transport = DailyTransport(
            self.args.room_url,
            self.args.token,
            self.args.bot_name,
            daily_params,
        )

        rtai = RTVIProcessor(
            transport=transport,
            setup=RTVISetup(config=self._bot_config),
            llm_processor=llm_processor,
            tts_processor=tts_processor,
        )

        self.task = PipelineTask(
            Pipeline(
                [
                    transport.input_processor(),
                    asr_processor,
                    rtai,
                ]
            ),
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                send_initial_empty_metrics=False,
            ),
        )

        transport.add_event_handler("on_first_participant_joined", self.on_first_participant_joined)
        transport.add_event_handler("on_participant_left", self.on_participant_left)
        transport.add_event_handler("on_call_state_updated", self.on_call_state_updated)

        await PipelineRunner().run(self.task)


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
        kwargs = BotRunArgs(
            room_name=DailyRESTHelper.get_name_from_url(config.u),
            bot_config=bot_config,
            room_url=config.u,
            token=config.t,
        ).__dict__
        bot = DailyAsrRTVIBot(**kwargs)
        bot.run()
    else:
        logging.error("Room URL and Token are required")
