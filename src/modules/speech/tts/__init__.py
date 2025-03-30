import logging
import os

from dotenv import load_dotenv

from src.core.llm import LLMEnvInit
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
        elif "tts_minicpmo" == tag:
            from . import minicpmo_tts
        elif "tts_zonos" == tag:
            from . import zonos_tts
        elif "tts_step" == tag:
            from . import step_tts
        elif "tts_spark" == tag:
            from . import spark_tts
        elif "tts_generator_spark" == tag:
            from . import spark_generator_tts
        elif "tts_orpheus" == tag:
            from . import orpheus_tts
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
                warmup_steps=int(os.getenv("LLASA_WARMUP_STEPS", "0")),
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

    @staticmethod
    def get_tts_orpheus_args() -> dict:
        from src.types.speech.tts.orpheus import OrpheusTTSArgs
        from src.types.llm.transformers import TransformersSpeechLMArgs
        from src.types.codec import CodecArgs

        kwargs = OrpheusTTSArgs(
            lm_args=TransformersSpeechLMArgs(
                lm_model_name_or_path=os.getenv(
                    "LM_MODEL_PATH", os.path.join(MODELS_DIR, "canopylabs/orpheus-3b-0.1-ft")
                ),
                lm_device=os.getenv("LM_DEVICE", None),
                warmup_steps=int(os.getenv("WARMUP_STEPS", "0")),
                lm_max_length=int(os.getenv("LM_MAX_LENGTH", "4096")),
                lm_gen_max_new_tokens=int(os.getenv("LM_GEN_MAX_NEW_TOKENS", "2048")),
                lm_gen_top_k=int(os.getenv("LM_GEN_TOP_K", "10")),
                lm_gen_top_p=float(os.getenv("LM_GEN_TOP_P", "0.95")),
                lm_gen_temperature=float(os.getenv("LM_GEN_TEMPERATURE", "0.6")),
                lm_gen_repetition_penalty=float(os.getenv("LM_GEN_REPETITION_PENALTY", "1.1")),
            ).__dict__,
            codec_args=CodecArgs(
                model_dir=os.getenv(
                    "CODEC_MODEL_PATH", os.path.join(MODELS_DIR, "hubertsiuzdak/snac_24khz")
                ),
            ).__dict__,
            voice_name=os.getenv("TTS_VOICE_NAME", "tara"),
            stream_factor=int(os.getenv("TTS_STREAM_FACTOR", "1")),
            token_overlap_len=int(os.getenv("TTS_TOKEN_OVERLAP_LEN", "0")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_step_args() -> dict:
        from src.types.speech.tts.step import StepTTSArgs
        from src.types.llm.transformers import TransformersSpeechLMArgs

        kwargs = StepTTSArgs(
            lm_args=TransformersSpeechLMArgs(
                lm_model_name_or_path=os.getenv(
                    "TTS_LM_MODEL_PATH", os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-TTS-3B")
                ),
                lm_device=os.getenv("TTS_LM_DEVICE", None),
                warmup_steps=int(os.getenv("TTS_WARMUP_STEPS", "1")),
                lm_gen_top_k=int(os.getenv("TTS_LM_GEN_TOP_K", "10")),
                lm_gen_top_p=float(os.getenv("TTS_LM_GEN_TOP_P", "1.0")),
                lm_gen_temperature=float(os.getenv("TTS_LM_GEN_TEMPERATURE", "0.8")),
                lm_gen_repetition_penalty=float(os.getenv("TTS_LM_GEN_REPETITION_PENALTY", "1.1")),
            ).__dict__,
            stream_factor=int(os.getenv("TTS_STREAM_FACTOR", "2")),
            tts_mode=os.getenv("TTS_MODE", "lm_gen"),
            speech_tokenizer_model_path=os.getenv(
                "TTS_TOKENIZER_MODEL_PATH",
                os.path.join(MODELS_DIR, "stepfun-ai/Step-Audio-Tokenizer"),
            ),
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_spark_args() -> dict:
        from src.types.speech.tts.spark import SparkTTSArgs
        from src.types.llm.transformers import TransformersSpeechLMArgs

        kwargs = SparkTTSArgs(
            device=os.getenv("TTS_DEVICE", None),
            model_dir=os.getenv(
                "TTS_MODEL_DIR", os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B")
            ),
            lm_args=TransformersSpeechLMArgs(
                lm_model_name_or_path=os.getenv(
                    "TTS_LM_MODEL_PATH", os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B/LLM")
                ),
                lm_device=os.getenv("TTS_LM_DEVICE", None),
                lm_gen_max_new_tokens=int(
                    os.getenv("TTS_LM_GEN_MAX_NEW_TOKENS", "8192")
                ),  # qwen2.5
                warmup_steps=int(os.getenv("TTS_WARMUP_sparkS", "1")),
                lm_gen_top_k=int(os.getenv("TTS_LM_GEN_TOP_K", "50")),
                lm_gen_top_p=float(os.getenv("TTS_LM_GEN_TOP_P", "0.95")),
                lm_gen_temperature=float(os.getenv("TTS_LM_GEN_TEMPERATURE", "0.8")),
                lm_gen_repetition_penalty=float(os.getenv("TTS_LM_GEN_REPETITION_PENALTY", "1.1")),
            ).__dict__,
            stream_factor=int(os.getenv("TTS_STREAM_FACTOR", "2")),
            stream_scale_factor=float(os.getenv("TTS_STREAM_SCALE_FACTOR", "1.0")),
            max_stream_factor=int(os.getenv("TTS_MAX_STREAM_FACTOR", "2")),
            token_overlap_len=int(os.getenv("TTS_TOKEN_OVERLAP_LEN", "0")),
            input_frame_rate=int(os.getenv("TTS_INPUT_FRAME_RATE", "25")),
        ).__dict__
        return kwargs

    def get_tts_generator_spark_args() -> dict:
        kwargs = TTSEnvInit.get_tts_spark_args()
        kwargs["lm_generator_tag"] = os.getenv("TTS_LM_GENERATOR_TAG", "llm_transformers_generator")
        # notice: lm_args use LLMEnvInit env params
        kwargs["lm_args"] = LLMEnvInit.map_config_func[kwargs["lm_generator_tag"]]()
        kwargs["lm_tokenzier_dir"] = os.getenv(
            "TTS_LM_TOKENIZER_DIR", os.path.join(MODELS_DIR, "SparkAudio/Spark-TTS-0.5B/LLM")
        )
        return kwargs

    @staticmethod
    def get_tts_minicpmo_args() -> dict:
        kwargs = LLMEnvInit.get_llm_transformers_args()
        kwargs["tts_task"] = os.getenv("TTS_TASK", "voice_cloning")
        kwargs["instruct_tpl"] = os.getenv("TTS_INSTRUCT_TPL", "")
        kwargs["voice_clone_instruct"] = os.getenv("VOICE_CLONE_INSTRUCT", "")
        kwargs["ref_audio_path"] = os.getenv("REF_AUDIO_PATH", None)
        return kwargs

    @staticmethod
    def get_tts_zonos_args() -> dict:
        from src.types.speech.tts.zonos import ZonosTTSArgs

        lm_checkpoint_dir = os.path.join(MODELS_DIR, "Zyphra/Zonos-v0.1-transformer")
        lm_checkpoint_dir = os.getenv("ZONOS_LM_CHECKPOINT_DIR", lm_checkpoint_dir)
        dac_model_dir = os.getenv("ZONOS_DAC_MODEL_DIR", "descript/dac_44khz")
        speaker_embedding_model_dir = os.getenv("SPEAKER_EMBEDDING_MODEL_DIR", None)
        kwargs = ZonosTTSArgs(
            lm_checkpoint_dir=lm_checkpoint_dir,
            dac_model_dir=dac_model_dir,
            speaker_embedding_model_dir=speaker_embedding_model_dir,
            language=os.getenv("TTS_LANG", "en-us"),  # don't use sys env LANGUAGE
            ref_audio_file_path=os.getenv("ZONOS_REF_AUDIO_PATH", ""),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        "tts_coqui": get_tts_coqui_args,
        "tts_cosy_voice": get_tts_cosy_voice_args,
        "tts_cosy_voice2": get_tts_cosy_voice_args,
        "tts_fishspeech": get_tts_fishspeech_args,
        "tts_llasa": get_tts_llasa_args,
        "tts_orpheus": get_tts_orpheus_args,
        "tts_f5": get_tts_f5_args,
        "tts_openvoicev2": get_tts_openvoicev2_args,
        "tts_kokoro": get_tts_kokoro_args,
        "tts_onnx_kokoro": get_tts_onnx_kokoro_args,
        "tts_chat": get_tts_chat_args,
        "tts_edge": get_tts_edge_args,
        "tts_g": get_tts_g_args,
        "tts_minicpmo": get_tts_minicpmo_args,
        "tts_zonos": get_tts_zonos_args,
        "tts_step": get_tts_step_args,
        "tts_spark": get_tts_spark_args,
        "tts_generator_spark": get_tts_generator_spark_args,
    }

    @staticmethod
    def get_tts_step_synth_args() -> dict:
        kwargs = {}
        kwargs["src_audio_path"] = os.getenv("SRC_AUDIO_PATH", None)
        return kwargs

    @staticmethod
    def get_tts_synth_args() -> dict:
        kwargs = {}
        return kwargs

    map_synthesize_config_func = {
        "tts_step": get_tts_step_synth_args,
    }
