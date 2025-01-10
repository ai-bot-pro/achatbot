import logging
import os

from dotenv import load_dotenv

from src.common.types import MODELS_DIR, RECORDS_DIR, CosyVoiceTTSArgs
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

load_dotenv(override=True)


class TTSEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ITts | EngineClass:
        if "tts_coqui" in tag:
            from . import coqui_tts
        elif "tts_chat" in tag:
            from . import chat_tts
        elif "tts_pyttsx3" in tag:
            from . import pyttsx3_tts
        elif "tts_g" in tag:
            from . import g_tts
        elif "tts_edge" in tag:
            from . import edge_tts
        elif "tts_cosy_voice" in tag:
            from . import cosy_voice_tts
        elif "tts_f5" in tag:
            from . import f5_tts
        elif "tts_openvoicev2" in tag:
            from . import openvoicev2_tts
        elif "tts_kokoro" in tag:
            from . import kokoro_tts
        elif "tts_onnx_kokoro" in tag:
            from . import kokoro_onnx_tts
        # elif "tts_openai" in tag:
        # from . import openai_tts

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initTTSEngine(select_tag=None) -> interface.ITts | EngineClass:
        # tts
        tag = os.getenv("TTS_TAG", "tts_edge")
        if select_tag:
            tag = select_tag
        kwargs = TTSEnvInit.map_config_func[tag]()
        engine = TTSEnvInit.getEngine(tag, **kwargs)
        logging.info(f"initTTSEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_tts_chat_args() -> dict:
        kwargs = {}
        kwargs["local_path"] = os.getenv("LOCAL_PATH", os.path.join(MODELS_DIR, "2Noise/ChatTTS"))
        kwargs["source"] = os.getenv("TTS_CHAT_SOURCE", "custom")
        return kwargs

    @staticmethod
    def get_tts_coqui_args() -> dict:
        kwargs = {}
        kwargs["model_path"] = os.getenv(
            "TTS_MODEL_PATH", os.path.join(MODELS_DIR, "coqui/XTTS-v2")
        )
        kwargs["conf_file"] = os.getenv(
            "TTS_CONF_FILE", os.path.join(MODELS_DIR, "coqui/XTTS-v2/config.json")
        )
        kwargs["reference_audio_path"] = os.getenv(
            "TTS_REFERENCE_AUDIO_PATH", os.path.join(RECORDS_DIR, "me.wav")
        )
        kwargs["tts_stream"] = bool(os.getenv("TTS_STREAM", ""))
        return kwargs

    @staticmethod
    def get_tts_cosy_voice_args() -> dict:
        kwargs = CosyVoiceTTSArgs().__dict__
        return kwargs

    @staticmethod
    def get_tts_f5_args() -> dict:
        from src.types.speech.tts.f5 import F5TTSArgs, F5TTSDiTModelConfig

        kwargs = F5TTSArgs(
            model_type=os.getenv("F5TTS_MODEL_TYPE", "F5-TTS"),
            model_cfg=F5TTSDiTModelConfig().__dict__,
            model_ckpt_path=os.getenv(
                "F5TTS_MODEL_CKPT_PATH",
                os.path.join(MODELS_DIR, "SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"),
            ),
            vocab_file=os.getenv("F5TTS_VOCAB_FILE", ""),
            vocoder_name=os.getenv("F5TTS_VOC_NAME", "vocos"),
            vocoder_ckpt_dir=os.getenv(
                "F5TTS_VOCODER_MODEL_DIR",
                os.path.join(MODELS_DIR, "charactr/vocos-mel-24khz"),
            ),
            ref_audio_file=os.getenv("F5TTS_AUDIO_PATH", ""),
            ref_text=os.getenv("F5TTS_REFERENCE_TEXT", ""),
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_openvoicev2_args() -> dict:
        from src.types.speech.tts.openvoicev2 import OpenVoiceV2TTSArgs

        kwargs = OpenVoiceV2TTSArgs(
            language=os.getenv("OPENVOICEV2_LANGUAGE", "ZH"),
            tts_ckpt_path=os.getenv(
                "OPENVOICEV2_TTS_CKPT_PATH",
                os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/checkpoint.pth"),
            ),
            tts_config_path=os.getenv(
                "OPENVOICEV2_TTS_CONFIG_PATH",
                os.path.join(MODELS_DIR, "myshell-ai/MeloTTS-Chinese/config.json"),
            ),
            enable_clone=bool(os.getenv("ENABLE_CLONE", "")),
            converter_ckpt_path=os.getenv(
                "OPENVOICEV2_CONVERTER_CKPT_PATH",
                os.path.join(MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/checkpoint.pth"),
            ),
            converter_conf_path=os.getenv(
                "OPENVOICEV2_CONVERTER_CONF_PATH",
                os.path.join(MODELS_DIR, "myshell-ai/OpenVoiceV2/converter/config.json"),
            ),
            src_se_ckpt_path=os.getenv(
                "OPENVOICEV2_SRC_SE_CKPT_PATH",
                os.path.join(MODELS_DIR, f"myshell-ai/OpenVoiceV2/base_speakers/ses/zh.pth"),
            ),
            target_se_ckpt_path=os.getenv("OPENVOICEV2_TARGET_SE_CKPT_PATH", ""),
            is_save=bool(os.getenv("IS_SAVE", "")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_kokoro_args() -> dict:
        kwargs = {}
        kwargs["language"] = os.getenv("KOKORO_LANGUAGE", "a")
        kwargs["voice"] = os.getenv("KOKORO_VOICE", "af")
        kwargs["tts_stream"] = bool(os.getenv("TTS_STREAM", ""))
        return kwargs

    @staticmethod
    def get_tts_onnx_kokoro_args() -> dict:
        kwargs = {}
        kwargs["language"] = os.getenv("KOKORO_LANGUAGE", "en-us")
        kwargs["voice"] = os.getenv("KOKORO_VOICE", "af")
        kwargs["espeak_ng_lib_path"] = os.getenv("KOKORO_ESPEAK_NG_LIB_PATH", None)
        kwargs["espeak_ng_data_path"] = os.getenv("KOKORO_ESPEAK_NG_DATA_PATH", None)
        kwargs["tts_stream"] = bool(os.getenv("TTS_STREAM", ""))
        return kwargs

    @staticmethod
    def get_tts_edge_args() -> dict:
        kwargs = {}
        kwargs["voice_name"] = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
        kwargs["language"] = os.getenv("TTS_LANG", "zh")
        return kwargs

    @staticmethod
    def get_tts_g_args() -> dict:
        kwargs = {}
        kwargs["language"] = os.getenv("TTS_LANG", "zh")
        kwargs["speed_increase"] = float(os.getenv("SPEED_INCREASE", "1.5"))
        return kwargs

    # TAG : config
    map_config_func = {
        "tts_coqui": get_tts_coqui_args,
        "tts_cosy_voice": get_tts_cosy_voice_args,
        "tts_f5": get_tts_f5_args,
        "tts_openvoicev2": get_tts_openvoicev2_args,
        "tts_kokoro": get_tts_kokoro_args,
        "tts_onnx_kokoro": get_tts_onnx_kokoro_args,
        "tts_chat": get_tts_chat_args,
        "tts_edge": get_tts_edge_args,
        "tts_g": get_tts_g_args,
    }
