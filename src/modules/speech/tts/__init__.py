import logging
import os

from dotenv import load_dotenv

from src.common.types import MODELS_DIR, RECORDS_DIR
from src.common import interface
from src.common.factory import EngineClass, EngineFactory

load_dotenv(override=True)


class TTSEnvInit:
    @staticmethod
    def getEngine(tag, **kwargs) -> interface.ITts | EngineClass:
        if "tts_coqui" == tag:
            from . import coqui_tts
        elif "tts_chat" == tag:
            from . import chat_tts
        elif "tts_pyttsx3" == tag:
            from . import pyttsx3_tts
        elif "tts_g" == tag:
            from . import g_tts
        elif "tts_edge" == tag:
            from . import edge_tts
        elif "tts_cosy_voice2" == tag:
            from . import cosy_voice2_tts
        elif "tts_cosy_voice" == tag:
            from . import cosy_voice_tts
        elif "tts_f5" == tag:
            from . import f5_tts
        elif "tts_openvoicev2" == tag:
            from . import openvoicev2_tts
        elif "tts_kokoro" == tag:
            from . import kokoro_tts
        elif "tts_onnx_kokoro" == tag:
            from . import kokoro_onnx_tts
        elif "tts_fishspeech" == tag:
            from . import fish_speech_tts
        elif "tts_llasa" == tag:
            from . import llasa_tts
        # elif "tts_openai" in tag:
        # from . import openai_tts

        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initTTSEngine(tag=None, **kwargs) -> interface.ITts | EngineClass:
        # tts
        tag = tag or os.getenv("TTS_TAG", "tts_edge")
        kwargs = kwargs or TTSEnvInit.map_config_func[tag]()
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
        from src.common.types import CosyVoiceTTSArgs

        model_dir = os.path.join(MODELS_DIR, "FunAudioLLM/CosyVoice-300M-SFT")
        model_dir = os.getenv("COSY_VOICE_MODELS_DIR", model_dir)
        kwargs = CosyVoiceTTSArgs(
            model_dir=model_dir,
            reference_text=os.getenv("COSY_VOICE_REFERENCE_TEXT", ""),
            reference_audio_path=os.getenv("COSY_VOICE_REFERENCE_AUDIO_PATH", ""),
            src_audio_path=os.getenv("COSY_VOICE_SRC_AUDIO_PATH", ""),
            instruct_text=os.getenv("COSY_VOICE_INSTRUCT_TEXT", ""),
            spk_id=os.getenv(
                "COSY_VOICE_SPK_ID", ""
            ),  # for cosyvoice sft/struct inference, cosyvoice2 don't use it
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_fishspeech_args() -> dict:
        from src.types.speech.tts.fish_speech import FishSpeechTTSArgs

        lm_checkpoint_dir = os.path.join(MODELS_DIR, "fishaudio/fish-speech-1.5")
        lm_checkpoint_dir = os.getenv("FS_LM_CHECKPOINT_DIR", lm_checkpoint_dir)
        gan_checkpoint_path = os.path.join(
            MODELS_DIR,
            "fishaudio/fish-speech-1.5",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        )
        gan_checkpoint_path = os.getenv("FS_GAN_CHECKPOINT_PATH", gan_checkpoint_path)
        kwargs = FishSpeechTTSArgs(
            warm_up_text=os.getenv("FS_WARM_UP_TEXT", ""),
            lm_checkpoint_dir=lm_checkpoint_dir,
            gan_checkpoint_path=gan_checkpoint_path,
            gan_config_path=os.getenv(
                "FS_GAN_CONFIG_PATH",
                "../../../../deps/FishSpeech/fish_speech/configs",
            ),
            ref_audio_path=os.getenv("FS_REFERENCE_AUDIO_PATH", None),
            ref_text=os.getenv("FS_REFERENCE_TEXT", ""),
        ).__dict__
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

    @staticmethod
    def get_tts_llasa_args() -> dict:
        from src.types.speech.tts.llasa import LlasaTTSArgs
        from src.types.llm.transformers import TransformersSpeechLMArgs
        from src.types.codec import CodecArgs

        kwargs = LlasaTTSArgs(
            lm_args=TransformersSpeechLMArgs(
                lm_model_name_or_path=os.getenv(
                    "LLASA_LM_MODEL_PATH", os.path.join(MODELS_DIR, "HKUSTAudio/Llasa-1B")
                ),
                lm_device=os.getenv("LLASA_LM_DEVICE", None),
                warnup_steps=int(os.getenv("LLASA_WARNUP_STEPS", "0")),
                lm_gen_top_k=int(os.getenv("LLASA_LM_GEN_TOP_K", "10")),
                lm_gen_top_p=float(os.getenv("LLASA_LM_GEN_TOP_P", "1.0")),
                lm_gen_temperature=float(os.getenv("LLASA_LM_GEN_TEMPERATURE", "0.8")),
                lm_gen_repetition_penalty=float(
                    os.getenv("LLASA_LM_GEN_REPETITION_PENALTY", "1.1")
                ),
            ).__dict__,
            xcode2_args=CodecArgs(
                model_dir=os.getenv(
                    "XCODE2_MODEL_PATH", os.path.join(MODELS_DIR, "HKUSTAudio/xcodec2")
                ),
            ).__dict__,
            ref_audio_file_path=os.getenv("LLASA_REF_AUDIO_PATH", ""),
            prompt_text=os.getenv("LLASA_PROMPT_TEXT", ""),
            is_save=bool(os.getenv("LLASA_IS_SAVE", "")),
            output_codebook_indices_dir=os.getenv(
                "LLASA_CODEBOOK_INDICES_DIR", os.path.join(MODELS_DIR, "llasa_codebook_indices")
            ),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        "tts_coqui": get_tts_coqui_args,
        "tts_cosy_voice": get_tts_cosy_voice_args,
        "tts_cosy_voice2": get_tts_cosy_voice_args,
        "tts_fishspeech": get_tts_fishspeech_args,
        "tts_llasa": get_tts_llasa_args,
        "tts_f5": get_tts_f5_args,
        "tts_openvoicev2": get_tts_openvoicev2_args,
        "tts_kokoro": get_tts_kokoro_args,
        "tts_onnx_kokoro": get_tts_onnx_kokoro_args,
        "tts_chat": get_tts_chat_args,
        "tts_edge": get_tts_edge_args,
        "tts_g": get_tts_g_args,
    }
