import logging

import uuid
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.parallel_pipeline import ParallelPipeline
from apipeline.pipeline.task import PipelineParams, PipelineTask
from apipeline.pipeline.runner import PipelineRunner
from apipeline.processors.logger import FrameLogger
from livekit import rtc

from src.processors.speech.tts.tts_processor import TTSProcessor
from src.processors.speech.audio_save_processor import SaveAllAudioProcessor
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.modules.speech.asr_live import ASRLiveEnvInit
from src.common.types import LivekitParams
from src.transports.livekit import LivekitTransport
from src.cmd.bots.base_livekit import LivekitRoomBot
from src.cmd.bots import register_ai_room_bots
from src.types.frames.data_frames import TextFrame, AudioRawFrame
from src.core.llm import LLMEnvInit
from src.processors.translation.llm_translate_processor import LLMTranslateProcessor
from src.common.session import Session, SessionCtx
from src.processors.punctuation_processor import PunctuationProcessor
from src.modules.punctuation import PuncEnvInit

from dotenv import load_dotenv

load_dotenv(override=True)

"""
some ways to run, below: 

# 1. run cmd signal webrtc bot
TOKENIZERS_PARALLELISM=false python -m src.cmd.bots.main -f config/bots/livekit_asr_translate_llamacpp_tts_bot.json

# 2. run webrtc room http signal bot server
TOKENIZERS_PARALLELISM=false python -m src.cmd.http.server.fastapi_room_bot_serve -f config/bots/livekit_asr_translate_llamacpp_tts_bot.json
# LivekitASRTranslateTTSBot join chat-room
curl -XPOST "http://0.0.0.0:4321/bot_join/chat-room/LivekitASRTranslateTTSBot"

# 3. run webrtc room http bots server (experimental)
TOKENIZERS_PARALLELISM=false python -m src.cmd.http.server.fastapi_livekit_bot_serve
# LivekitASRTranslateTTSBot join chat-room with bot config
curl --location 'http://0.0.0.0:4321/bot_join/chat-room/LivekitASRTranslateTTSBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "LivekitASRTranslateTTSBot",
    "room_name": "chat-room",
    "room_url": "",
    "token": "",
    "room_manager": {
        "tag": "livekit_room",
        "args": {
            "privacy": "public"
        }
    },
    "services": {
        "pipeline": "achatbot",
        "vad": "silero",
        "asr": "sense_voice",
        "punctuation": "punc_ct_tranformer_onnx_offline",
        "translate_llm": "llm_llamacpp_generator",
        "tts": "edge"
    },
    "config": {
        "vad": {
            "tag": "silero_vad_analyzer",
            "args": {
                "start_secs": 0.032,
                "stop_secs": 0.32,
                "confidence": 0.7,
                "min_volume": 0.6,
                "onnx": true
            }
        },
        "asr": {
            "tag": "sense_voice_asr",
            "args": {
                "language": "zn",
                "model_name_or_path": "./models/FunAudioLLM/SenseVoiceSmall"
            }
        },
        "punctuation": {
            "tag": "punc_ct_tranformer_onnx_offline",
            "args": {
                "model": "./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
            }
        },
        "translate_llm": {
            "init_prompt": "hi, welcome to speak with translation bot.",
            "model": "./models/ByteDance-Seed/Seed-X-PPO-7B",
            "src": "zh",
            "target": "en",
            "streaming": false,
            "tag": "llm_llamacpp_generator",
            "args": {
                "save_chat_history": false,
                "model_path": "./models/Seed-X-PPO-7B.Q2_K.gguf",
                "model_type": "generate",
                "llm_temperature": 0.0,
                "llm_stop_ids": [
                    2
                ],
                "llm_max_tokens": 2048
            }
        },
        "tts": {
            "aggregate_sentences": false,
            "push_text_frames": true,
            "remove_punctuation": false,
            "tag": "tts_edge",
            "args": {
                "voice_name": "en-US-GuyNeural",
                "language": "en",
                "gender": "Male"
            }
        }
    },
    "config_list": []
}'

# 4. websocket + webrtc bots server
"""


@register_ai_room_bots.register
class LivekitASRTranslateTTSBot(LivekitRoomBot):
    """
    livekit transport(webrtc) with vad -> asr -> translate LLM -> punc -> tts
    """

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.init_bot_config()

        self.vad_analyzer = None
        self.asr_engine = None
        self.generator = None
        self.tts_engine = None
        self.asr_punc_engine = None

    def load(self):
        self.vad_analyzer = self.get_vad_analyzer()
        self.asr_engine = self.get_asr()
        self.tts_engine = self.get_tts()
        self.generator = self.get_translate_llm_generator()

        # load punctuation engine
        if self.asr_engine.get_args_dict().get("textnorm", False) is False:
            if self._bot_config.punctuation:
                tag = self._bot_config.punctuation.tag
                args = self._bot_config.punctuation.args or {}
                self.asr_punc_engine = PuncEnvInit.initEngine(tag, **args)

    async def arun(self):
        self.params = LivekitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=self.vad_analyzer,
            vad_audio_passthrough=True,
        )

        stream_info = self.tts_engine.get_stream_info()
        self.params.audio_out_sample_rate = stream_info["rate"]
        self.params.audio_out_channels = stream_info["channels"]

        transport = LivekitTransport(
            self.args.token,
            self.params,
        )

        asr_processor = self.get_asr_processor(asr_engine=self.asr_engine)

        punc_processor = None
        if self.asr_punc_engine:
            punc_processor = PunctuationProcessor(engine=self.asr_punc_engine, session=self.session)

        tl_processor = None
        if self.generator is not None:
            tl_processor = LLMTranslateProcessor(
                tokenizer=self.get_hf_tokenizer(),
                generator=self.generator,
                session=self.session,
                src=self._bot_config.translate_llm.src,
                target=self._bot_config.translate_llm.target,
                streaming=self._bot_config.translate_llm.streaming,
                prompt_tpl=self._bot_config.translate_llm.prompt_tpl,
            )

        self.tts_processor: TTSProcessor = self.get_tts_processor(tts_engine=self.tts_engine)

        # record_save_processor = SaveAllAudioProcessor(
        #    prefix_name="livekit_asr_translate_tts_bot",
        #    sample_rate=self.params.audio_in_sample_rate,
        #    channels=self.params.audio_in_channels,
        #    sample_width=self.params.audio_in_sample_width,
        # )
        processors = [
            transport.input_processor(),
            # record_save_processor,
            # FrameLogger(include_frame_types=[AudioRawFrame]),
            asr_processor,
            FrameLogger(include_frame_types=[TextFrame]),
            punc_processor,
            FrameLogger(include_frame_types=[TextFrame]),
            ParallelPipeline(
                [transport.output_processor()],
                [
                    tl_processor,
                    FrameLogger(include_frame_types=[TextFrame]),
                    self.tts_processor,
                    FrameLogger(include_frame_types=[TextFrame, AudioRawFrame]),
                    transport.output_processor(),
                ],
            ),
        ]
        processors = [p for p in processors if p is not None]
        logging.info(f"{processors=}")

        self.task = PipelineTask(
            Pipeline(processors=processors),
            params=PipelineParams(),
        )

        transport.add_event_handlers(
            "on_first_participant_joined",
            [self.on_first_participant_joined, self.on_first_participant_say_hi],
        )

        self.runner = PipelineRunner(handle_sigint=self._handle_sigint)
        await self.runner.run(self.task)

    async def on_first_participant_say_hi(
        self, transport: LivekitTransport, participant: rtc.RemoteParticipant
    ):
        self.session.set_client_id(participant.sid)
        await self.tts_processor.say("hi, welcome to chat with translation bot.")
