import os
import logging
import asyncio

from src.common.logger import Logger
from src.common import interface
from src.common.config import Conf
from src.common.factory import EngineFactory, EngineClass
from src.common.types import CONFIG_DIR
from src.core.llm import LLMEnvInit
from src.core.llm.llamacpp import PromptInit
from src.modules.functions.search.api import SearchFuncEnvInit
from src.modules.functions.weather.api import WeatherFuncEnvInit
from src.modules.speech.asr import ASREnvInit
from src.modules.speech.audio_stream import AudioStreamEnvInit
from src.modules.speech.detector import VADEnvInit, WakerEnvInit
from src.modules.speech.player import PlayerEnvInit
from src.modules.speech.recorder import RecorderEnvInit
from src.modules.speech.tts import TTSEnvInit
from src.modules.speech.vad_analyzer import VADAnalyzerEnvInit
from src.modules.vision.detector import VisionDetectorEnvInit

from dotenv import load_dotenv

load_dotenv(override=True)


class Env(
    PromptInit,
    AudioStreamEnvInit,
    VADAnalyzerEnvInit,
    VADEnvInit,
    WakerEnvInit,
    RecorderEnvInit,
    ASREnvInit,
    LLMEnvInit,
    SearchFuncEnvInit,
    WeatherFuncEnvInit,
    TTSEnvInit,
    PlayerEnvInit,
    VisionDetectorEnvInit,
):
    @classmethod
    async def save_to_yamls(cls, tag=None):
        return await Conf.save_to_yamls(cls, tag)


class YamlConfig(PromptInit):
    @staticmethod
    async def load_engine(key, tag, file_path):
        #!TODO: dispatch load_engine to sub modules @weedge

        import src.core.llm
        import src.modules.functions.search
        import src.modules.functions.weather
        import src.modules.speech

        conf = await Conf.load_from_yaml(file_path)
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **conf)
        return key, engine

    @staticmethod
    async def load(mainifests_path=None, engine_name=None) -> dict:
        if mainifests_path is None:
            env = os.getenv("CONF_ENV", "local")
            mainifests_path = os.path.join(CONFIG_DIR, env, "manifests.yaml")

        conf = await Conf.load_from_yaml(mainifests_path)
        tasks = []
        for key, item in conf.items():
            if engine_name and engine_name != key:
                continue
            task = asyncio.create_task(YamlConfig.load_engine(key, item.tag, item.file_path))
            tasks.append(task)
        res = await asyncio.gather(*tasks)

        engines = {}
        for key, engine in res:
            logging.info(f"{key} engine: {engine} args: {engine.args}")
            engines[key] = engine
        return engines

    @staticmethod
    def initAudioInStreamEngine() -> interface.IAudioStream | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="audioinstream"))["audioinstream"]

    @staticmethod
    def initAudioOutStreamEngine() -> interface.IAudioStream | EngineClass:
        return asyncio.run(YamlConfig.load(engine_name="audiooutstream"))["audiooutstream"]

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

CONF_ENV=local \
    AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    python -m src.cmd.init -o env2yaml
CONF_ENV=local \
    AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    python -m src.cmd.init -o env2yaml
CONF_ENV=local \
    AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    python -m src.cmd.init -o env2yaml

CONF_ENV=local python -m src.cmd.init -o init_engine -i config
CONF_ENV=local python -m src.cmd.init -o gather_load_configs
"""
if __name__ == "__main__":
    # os.environ['CONF_ENV'] = 'local'
    # os.environ['RECORDER_TAG'] = 'wakeword_rms_recorder'

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", "-o", type=str, default="load_engine", help="op method")
    parser.add_argument(
        "--init_type", "-i", type=str, default="env", help="init type from env or config"
    )
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
