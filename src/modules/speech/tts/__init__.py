import logging
import os

from src.common.types import MODELS_DIR, RECORDS_DIR, CosyVoiceTTSArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

from dotenv import load_dotenv
load_dotenv(override=True)


class TTSEnvInit():

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        from . import coqui_tts
        from . import chat_tts
        from . import pyttsx3_tts
        from . import g_tts
        from . import edge_tts
        from . import cosy_voice_tts
        # from . import openai_tts

        # tts
        tag = os.getenv('TTS_TAG', "tts_edge")
        kwargs = TTSEnvInit.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initTTSEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_tts_chat_args() -> dict:
        kwargs = {}
        kwargs["local_path"] = os.getenv('LOCAL_PATH', os.path.join(
            MODELS_DIR, "2Noise/ChatTTS"))
        kwargs["source"] = os.getenv('TTS_CHAT_SOURCE', "custom")
        return kwargs

    @staticmethod
    def get_tts_coqui_args() -> dict:
        kwargs = {}
        kwargs["model_path"] = os.getenv('TTS_MODEL_PATH', os.path.join(
            MODELS_DIR, "coqui/XTTS-v2"))
        kwargs["conf_file"] = os.getenv(
            'TTS_CONF_FILE', os.path.join(MODELS_DIR, "coqui/XTTS-v2/config.json"))
        kwargs["reference_audio_path"] = os.getenv('TTS_REFERENCE_AUDIO_PATH', os.path.join(
            RECORDS_DIR, "me.wav"))
        kwargs["tts_stream"] = bool(os.getenv('TTS_STREAM', ""))
        return kwargs

    @staticmethod
    def get_tts_cosy_voice_args() -> dict:
        kwargs = CosyVoiceTTSArgs().__dict__
        return kwargs

    @staticmethod
    def get_tts_edge_args() -> dict:
        kwargs = {}
        kwargs["voice_name"] = os.getenv('VOICE_NAME', "zh-CN-XiaoxiaoNeural")
        kwargs["language"] = os.getenv('LANGUAGE', "zh")
        return kwargs

    @staticmethod
    def get_tts_g_args() -> dict:
        kwargs = {}
        kwargs["language"] = os.getenv('LANGUAGE', "zh")
        kwargs["speed_increase"] = float(os.getenv('SPEED_INCREASE', "1.5"))
        return kwargs

    # TAG : config
    map_config_func = {
        'tts_coqui': get_tts_coqui_args,
        'tts_cosy_voice': get_tts_cosy_voice_args,
        'tts_chat': get_tts_chat_args,
        'tts_edge': get_tts_edge_args,
        'tts_g': get_tts_g_args,
    }
