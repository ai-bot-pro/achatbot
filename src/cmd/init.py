import os
import logging
import inspect
from typing import (Any)
import asyncio

import pyaudio

from src.common.logger import Logger
from src.common import interface
from src.common.config import Conf
from src.common.factory import EngineFactory, EngineClass
from src.common.types import (
    MODELS_DIR, RECORDS_DIR, CHUNK, CONFIG_DIR,
    SileroVADArgs,
    WebRTCVADArgs,
    WebRTCSileroVADArgs,
    AudioRecoderArgs,
    VADRecoderArgs,
    CosyVoiceTTSArgs,
    PersonalAIProxyArgs,
    PyAudioStreamArgs,
    DailyAudioStreamArgs,
)
from src.modules.functions.search.api import SearchFuncEnvInit
from src.modules.functions.weather.api import WeatherFuncEnvInit
# need import engine class -> EngineClass.__subclasses__
import src.modules.speech
import src.core.llm

from dotenv import load_dotenv
load_dotenv(override=True)


DEFAULT_SYSTEM_PROMPT = "你是一个中国人,一名中文助理，请用中文简短回答，回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"


class PlayStreamInit():
    # TTS_TAG : stream_info
    map_tts_player_stream_info = {
        'tts_coqui': {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 24000,
            "sample_width": 4,
        },
        'tts_chat': {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": 24000,
            "sample_width": 4,
        },
        'tts_edge': {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
            "sample_width": 2,
        },
        'tts_g': {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
            "sample_width": 2,
        },
        'tts_cosy_voice': {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
            "sample_width": 2,
        },
        'tts_daily_speaker': {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "sample_width": 2,
        },
        'tts_16k_speaker': {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "sample_width": 2,
        },
    }

    @staticmethod
    def get_stream_info() -> dict:
        tts_tag = os.getenv('TTS_TAG', "tts_chat")
        if tts_tag in PlayStreamInit.map_tts_player_stream_info:
            return PlayStreamInit.map_tts_player_stream_info[tts_tag]
        return {}


class PromptInit():
    @staticmethod
    def create_phi3_prompt(history: list[str],
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                           init_message: str = None):
        prompt = f'<|system|>\n{system_prompt}</s>\n'
        if init_message:
            prompt += f"<|assistant|>\n{init_message}</s>\n"

        return prompt + "".join(history) + "<|assistant|>\n"

    def create_qwen_prompt(history: list[str],
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                           init_message: str = None):
        prompt = f'<|system|>\n{system_prompt}<|end|>\n'
        if init_message:
            prompt += f"<|assistant|>\n{init_message}<|end|>\n"

        return prompt + "".join(history) + "<|assistant|>\n"

    @staticmethod
    def create_prompt(name: str, history: list[str],
                      system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                      init_message: str = None):
        if "phi-3" == name:
            return Env.create_phi3_prompt(history, system_prompt, init_message)
        if "qwen-2" == name:
            return Env.create_qwen_prompt(history, system_prompt, init_message)

        return None

    @staticmethod
    def get_user_prompt(name: str, text: str):
        if "phi-3" == name:
            return (f"<|user|>\n{text}</s>\n")
        if "qwen-2" == name:
            return (f"<|start|>user\n{text}<|end|>\n")
        return None

    @staticmethod
    def get_assistant_prompt(name: str, text: str):
        if "phi-3" == name:
            return (f"<|assistant|>\n{text}</s>\n")
        if "qwen-2" == name:
            return (f"<|assistant|>\n{text}<|end|>\n")
        return None


class Env(
    PromptInit, PlayStreamInit,
    SearchFuncEnvInit, WeatherFuncEnvInit,
):

    @staticmethod
    def initAudioInStreamEngine() -> interface.IAudioStream | EngineClass:
        # audio stream
        tag = os.getenv('AUDIO_IN_STREAM_TAG', "pyaudio_in_stream")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initAudioInStreamEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initAudioOutStreamEngine() -> interface.IAudioStream | EngineClass:
        # audio stream
        tag = os.getenv('AUDIO_OUT_STREAM_TAG', "pyaudio_out_stream")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initAudioOutStreamEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initWakerEngine() -> interface.IDetector | EngineClass:
        # waker
        recorder_tag = os.getenv('RECORDER_TAG', "rms_recorder")
        if "wake" not in recorder_tag:
            return None

        tag = os.getenv('WAKER_DETECTOR_TAG', "porcupine_wakeword")
        wake_words = os.getenv('WAKE_WORDS', "小黑")
        model_path = os.path.join(
            MODELS_DIR, "porcupine_params_zh.pv")
        keyword_paths = os.path.join(
            MODELS_DIR, "小黑_zh_mac_v3_0_0.ppn")
        kwargs = {}
        kwargs["wake_words"] = wake_words
        kwargs["keyword_paths"] = os.getenv(
            'KEYWORD_PATHS', keyword_paths).split(',')
        kwargs["model_path"] = os.getenv('MODEL_PATH', model_path)
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        # recorder
        tag = os.getenv('RECORDER_TAG', "rms_recorder")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initRecorderEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initVADEngine() -> interface.IDetector | EngineClass:
        # vad detector
        tag = os.getenv('VAD_DETECTOR_TAG', "webrtc_vad")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initVADEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initASREngine() -> interface.IAsr | EngineClass:
        # asr
        tag = os.getenv('ASR_TAG', "whisper_timestamped_asr")
        kwargs = {}
        kwargs["model_name_or_path"] = os.getenv(
            'ASR_MODEL_NAME_OR_PATH', 'base')
        kwargs["download_path"] = MODELS_DIR
        kwargs["verbose"] = bool(os.getenv('ASR_VERBOSE', 'True'))
        kwargs["language"] = os.getenv('ASR_LANG', 'zh')
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initASREngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initLLMEngine() -> interface.ILlm | EngineClass:
        # llm
        tag = os.getenv('LLM_TAG', "llm_llamacpp")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initLLMEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_llm_llamacpp_args() -> dict:
        kwargs = {}
        kwargs["model_name"] = os.getenv('LLM_MODEL_NAME', 'phi-3')
        kwargs["model_path"] = os.getenv('LLM_MODEL_PATH', os.path.join(
            MODELS_DIR, "Phi-3-mini-4k-instruct-q4.gguf"))
        kwargs["model_type"] = os.getenv('LLM_MODEL_TYPE', "generation")
        kwargs["n_threads"] = os.cpu_count()
        kwargs["verbose"] = False
        kwargs["n_gpu_layers"] = int(os.getenv('N_GPU_LAYERS', "0"))
        kwargs["flash_attn"] = bool(os.getenv('FLASH_ATTN', ""))
        kwargs["llm_stream"] = True
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs['llm_stop'] = [
            "<|end|>", "<|im_end|>", "<|endoftext|>", "<{/end}>",
            "</s>", "/s>", "</s", "<s>",
            "<|user|>", "<|assistant|>", "<|system|>",
        ]
        kwargs["llm_chat_system"] = os.getenv('LLM_CHAT_SYSTEM', DEFAULT_SYSTEM_PROMPT)
        kwargs["llm_tool_choice"] = os.getenv('LLM_TOOL_CHOICE', None)
        kwargs["chat_format"] = os.getenv('LLM_CHAT_FORMAT', None)
        kwargs["tokenizer_path"] = os.getenv('LLM_TOKENIZER_PATH', None)
        return kwargs

    @staticmethod
    def get_llm_personal_ai_proxy_args() -> dict:
        kwargs = PersonalAIProxyArgs(
            api_url=os.getenv('API_URL', "http://localhost:8787/"),
            chat_bot=os.getenv('CHAT_BOT', "openai"),
            model_type=os.getenv('CHAT_TYPE', "chat_only"),
            openai_api_base_url=os.getenv('OPENAI_API_BASE_URL', "https://api.groq.com/openai/v1/"),
            model_name=os.getenv('LLM_MODEL_NAME', "llama3-70b-8192"),
            llm_chat_system=os.getenv('LLM_CHAT_SYSTEM', DEFAULT_SYSTEM_PROMPT),
            llm_stream=False,
            llm_max_tokens=int(os.getenv('LLM_MAX_TOKENS', "1024")),
            func_search_name=os.getenv('FUNC_SEARCH_NAME', "search_api"),
            func_weather_name=os.getenv('FUNC_WEATHER_NAME', "openweathermap_api"),
        ).__dict__
        return kwargs

    @staticmethod
    def get_tts_cosy_voice_args() -> dict:
        kwargs = CosyVoiceTTSArgs().__dict__
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
    def get_tts_chat_args() -> dict:
        kwargs = {}
        kwargs["local_path"] = os.getenv('LOCAL_PATH', os.path.join(
            MODELS_DIR, "2Noise/ChatTTS"))
        kwargs["source"] = os.getenv('TTS_CHAT_SOURCE', "local")
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

    @staticmethod
    def get_silero_vad_args() -> dict:
        kwargs = SileroVADArgs(
            repo_or_dir=os.getenv('REPO_OR_DIR', "snakers4/silero-vad"),
            model=os.getenv('SILERO_MODEL', "silero_vad"),
            check_frames_mode=int(os.getenv('CHECK_FRAMES_MODE', "1")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_webrtc_vad_args() -> dict:
        kwargs = WebRTCVADArgs(
            aggressiveness=int(os.getenv('AGGRESSIVENESS', "1")),
            check_frames_mode=int(os.getenv('CHECK_FRAMES_MODE', "1")),
        ).__dict__
        return kwargs

    @staticmethod
    def get_webrtc_silero_vad_args() -> dict:
        kwargs = WebRTCSileroVADArgs(
            aggressiveness=int(os.getenv('AGGRESSIVENESS', "1")),
            check_frames_mode=int(os.getenv('CHECK_FRAMES_MODE', "1")),
            repo_or_dir=os.getenv('REPO_OR_DIR', "snakers4/silero-vad"),
            model=os.getenv('SILERO_MODEL', "silero_vad"),
        ).__dict__
        return kwargs

    @staticmethod
    def get_pyannote_vad_args() -> dict:
        model_type = os.getenv(
            'VAD_MODEL_TYPE', 'segmentation-3.0')
        model_ckpt_path = os.path.join(
            MODELS_DIR, 'pyannote', model_type, "pytorch_model.bin")
        kwargs = {}
        kwargs["path_or_hf_repo"] = os.getenv(
            'VAD_PATH_OR_HF_REPO', model_ckpt_path)
        kwargs["model_type"] = model_type
        return kwargs

    @staticmethod
    def get_rms_recorder_args() -> dict:
        input_device_index = os.getenv('INPUT_DEVICE_INDEX', None)
        if input_device_index is not None:
            input_device_index = int(input_device_index)
        kwargs = AudioRecoderArgs(
            is_stream_callback=bool(os.getenv('IS_STREAM_CALLBACK', "True")),
            input_device_index=input_device_index,
        ).__dict__
        return kwargs

    @staticmethod
    def get_vad_recorder_args() -> dict:
        input_device_index = os.getenv('INPUT_DEVICE_INDEX', None)
        if input_device_index is not None:
            input_device_index = int(input_device_index)
        kwargs = VADRecoderArgs(
            is_stream_callback=bool(os.getenv('IS_STREAM_CALLBACK', "True")),
            input_device_index=input_device_index,
        ).__dict__
        return kwargs

    @staticmethod
    def get_pyaudio_in_stream_args() -> dict:
        input_device_index = os.getenv('INPUT_DEVICE_INDEX', None)
        if input_device_index is not None:
            input_device_index = int(input_device_index)
        kwargs = PyAudioStreamArgs(
            input=True,
            output=False,
            input_device_index=input_device_index,
            channels=int(os.getenv('IN_CHANNELS', "1")),
            rate=int(os.getenv('IN_SAMPLE_RATE', "16000")),
            sample_width=int(os.getenv('IN_SAMPLE_WIDTH', "2")),
        ).__dict__
        return kwargs

    def get_pyaudio_out_stream_args() -> dict:
        info = Env.get_stream_info()
        info['input'] = False
        info['output'] = True
        output_device_index = os.getenv('OUTPUT_DEVICE_INDEX', None)
        if output_device_index is not None:
            output_device_index = int(output_device_index)
        info['output_device_index'] = output_device_index

        return info

    @staticmethod
    def get_daily_room_audio_in_stream_args() -> dict:
        kwargs = DailyAudioStreamArgs(
            bot_name=os.getenv('BOT_NAME', "chat-bot"),
            input=True,
            output=False,
            in_channels=int(os.getenv('IN_CHANNELS', "1")),
            in_sample_rate=int(os.getenv('IN_SAMPLE_RATE', "16000")),
            in_sample_width=int(os.getenv('IN_SAMPLE_WIDTH', "2")),
            meeting_room_token=os.getenv('MEETING_ROOM_TOKEN', ""),
            meeting_room_url=os.getenv('MEETING_ROOM_URL', ""),
        ).__dict__
        return kwargs

    @staticmethod
    def get_daily_room_audio_out_stream_args() -> dict:
        kwargs = DailyAudioStreamArgs(
            bot_name=os.getenv('BOT_NAME', "chat-bot"),
            input=False,
            output=True,
            out_channels=int(os.getenv('OUT_CHANNELS', "1")),
            out_sample_rate=int(os.getenv('OUT_SAMPLE_RATE', "16000")),
            out_sample_width=int(os.getenv('OUT_SAMPLE_WIDTH', "2")),
            meeting_room_token=os.getenv('MEETING_ROOM_TOKEN', ""),
            meeting_room_url=os.getenv('MEETING_ROOM_URL', ""),
        ).__dict__
        return kwargs

    # TAG : config
    map_config_func = {
        'pyaudio_in_stream': get_pyaudio_in_stream_args,
        'pyaudio_out_stream': get_pyaudio_out_stream_args,
        'daily_room_audio_in_stream': get_daily_room_audio_in_stream_args,
        'daily_room_audio_out_stream': get_daily_room_audio_out_stream_args,
        'tts_coqui': get_tts_coqui_args,
        'tts_cosy_voice': get_tts_cosy_voice_args,
        'tts_chat': get_tts_chat_args,
        'tts_edge': get_tts_edge_args,
        'tts_g': get_tts_g_args,
        'silero_vad': get_silero_vad_args,
        'webrtc_vad': get_webrtc_vad_args,
        'webrtc_silero_vad': get_webrtc_silero_vad_args,
        'pyannote_vad': get_pyannote_vad_args,
        'rms_recorder': get_rms_recorder_args,
        'wakeword_rms_recorder': get_rms_recorder_args,
        'vad_recorder': get_vad_recorder_args,
        'wakeword_vad_recorder': get_vad_recorder_args,
        'llm_llamacpp': get_llm_llamacpp_args,
        'llm_personalai_proxy': get_llm_personal_ai_proxy_args,
    }

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        # tts
        tag = os.getenv('TTS_TAG', "tts_chat")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initTTSEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer | EngineClass:
        # player
        tag = os.getenv('PLAYER_TAG', "stream_player")
        info = Env.get_stream_info()
        if tts:
            info = tts.get_stream_info()
        info["frames_per_buffer"] = CHUNK * 10
        output_device_index = os.getenv('OUTPUT_DEVICE_INDEX', None)
        if output_device_index is not None:
            info["output_device_index"] = int(output_device_index)
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **info)
        logging.info(f"initPlayerEngine: {tag},  {engine}")
        return engine

    @classmethod
    async def save_to_yamls(cls, tag=None):
        return await Conf.save_to_yamls(cls, tag)


class YamlConfig(PromptInit):

    @staticmethod
    async def load_engine(key, tag, file_path):
        conf = await Conf.load_from_yaml(file_path)
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **conf)
        return key, engine

    @staticmethod
    async def load(mainifests_path=None, engine_name=None) -> dict:
        if mainifests_path is None:
            env = os.getenv('CONF_ENV', "local")
            mainifests_path = os.path.join(CONFIG_DIR, env, "manifests.yaml")

        conf = await Conf.load_from_yaml(mainifests_path)
        tasks = []
        for key, item in conf.items():
            if engine_name and engine_name != key:
                continue
            task = asyncio.create_task(
                YamlConfig.load_engine(key, item.tag, item.file_path))
            tasks.append(task)
        res = await asyncio.gather(*tasks)

        engines = {}
        for key, engine in res:
            logging.info(f"{key} engine: {engine} args: {engine.args}")
            engines[key] = engine
        return engines

    @staticmethod
    def initWakerEngine() -> interface.IDetector | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="waker"))["waker"]

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="recorder"))["recorder"]

    @staticmethod
    def initVADEngine() -> interface.IDetector | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="vad"))["vad"]

    @staticmethod
    def initASREngine() -> interface.IAsr | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="asr"))["asr"]

    @staticmethod
    def initLLMEngine() -> interface.ILlm | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="llm"))["llm"]

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="tts"))["tts"]

    @staticmethod
    def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="player"))["player"]


def env2yaml():
    res = asyncio.run(Env.save_to_yamls())
    for file_path in res:
        logging.info(file_path)


def get_engines(init_type="env"):
    if init_type == "config":
        return EngineFactory.get_init_engines(YamlConfig)
    return EngineFactory.get_init_engines(Env)


r"""
CONF_ENV=local python -m src.cmd.init
CONF_ENV=local python -m src.cmd.init -o init_engine -i env

CONF_ENV=local python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_coqui python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_chat python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_g python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_edge python -m src.cmd.init -o env2yaml
CONF_ENV=local \
    TTS_TAG=tts_edge \
    RECORDER_TAG=rms_recorder \
    ASR_TAG=whisper_groq_asr \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m src.cmd.init -o env2yaml
CONF_ENV=local \
    TTS_TAG=tts_edge \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=whisper_groq_asr \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m src.cmd.init -o env2yaml
CONF_ENV=local \
    TTS_TAG=tts_edge \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=whisper_groq_asr \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m src.cmd.init -o env2yaml

CONF_ENV=local FUNC_SEARCH_TAG=search_api python -m src.cmd.init -o env2yaml
CONF_ENV=local FUNC_SEARCH_TAG=search1_api python -m src.cmd.init -o env2yaml
CONF_ENV=local FUNC_SEARCH_TAG=serper_api python -m src.cmd.init -o env2yaml

CONF_ENV=local FUNC_WEATHER_TAG=openweathermap_api python -m src.cmd.init -o env2yaml

CONF_ENV=local LLM_TAG=llm_personalai_proxy  python -m src.cmd.init -o env2yaml

CONF_ENV=local python -m src.cmd.init -o init_engine -i config
CONF_ENV=local python -m src.cmd.init -o gather_load_configs
"""
if __name__ == "__main__":
    # os.environ['CONF_ENV'] = 'local'
    # os.environ['RECORDER_TAG'] = 'wakeword_rms_recorder'

    Logger.init(logging.INFO, is_file=False)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', "-o", type=str,
                        default="load_engine", help='op method')
    parser.add_argument('--init_type', "-i", type=str,
                        default="env",
                        help='init type from env or config')
    args = parser.parse_args()
    if args.op == "load_engine":
        engines = asyncio.run(YamlConfig.load())
        print(engines)
    elif args.op == "init_engine":
        engines = get_engines(args.init_type)
        print(engines)
    elif args.op == "env2yaml":
        env2yaml()
    elif args.op == "gather_load_configs":
        res = asyncio.run(YamlConfig.load())
        print(res)
    else:
        print("unsupport op")
