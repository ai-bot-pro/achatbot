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
from src.common.types import MODELS_DIR, RECORDS_DIR, CHUNK, CONFIG_DIR
# need import engine class -> EngineClass.__subclasses__
import src.modules.speech
import src.core.llm


DEFAULT_SYSTEM_PROMPT = "你是一个中国人,请用中文回答。回答限制在1-5句话内。"


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


class Env(PromptInit):

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
        kwargs["access_key"] = os.getenv('PORCUPINE_ACCESS_KEY', "")
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
        kwargs = {}
        kwargs["input_device_index"] = os.getenv('INPUT_DEVICE_INDEX', None)
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initRecorderEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initVADEngine() -> interface.IDetector | EngineClass:
        # vad detector
        tag = os.getenv('VAD_DETECTOR_TAG', "pyannote_vad")
        model_type = os.getenv(
            'VAD_MODEL_TYPE', 'segmentation-3.0')
        model_ckpt_path = os.path.join(
            MODELS_DIR, 'pyannote', model_type, "pytorch_model.bin")
        kwargs = {}
        kwargs["path_or_hf_repo"] = os.getenv(
            'VAD_PATH_OR_HF_REPO', model_ckpt_path)
        kwargs["model_type"] = model_type
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
        kwargs["verbose"] = True
        kwargs["language"] = "zh"
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initASREngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initLLMEngine() -> interface.ILlm | EngineClass:
        # llm
        tag = os.getenv('LLM_TAG', "llm_llamacpp")
        kwargs = {}
        kwargs["model_name"] = os.getenv('LLM_MODEL_NAME', 'phi-3')
        kwargs["model_path"] = os.getenv('LLM_MODEL_PATH', os.path.join(
            MODELS_DIR, "Phi-3-mini-4k-instruct-q4.gguf"))
        kwargs["model_type"] = os.getenv('LLM_MODEL_TYPE', "generation")
        kwargs["n_threads"] = os.cpu_count()
        kwargs["verbose"] = False
        kwargs["llm_stream"] = True
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs['llm_stop'] = [
            "<|end|>", "<|im_end|>", "<|endoftext|>", "<{/end}>",
            "</s>", "/s>", "</s", "<s>",
            "<|user|>", "<|assistant|>", "<|system|>",
        ]
        kwargs["llm_chat_system"] = DEFAULT_SYSTEM_PROMPT
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initLLMEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def get_tts_coqui_args() -> dict:
        kwargs = {}
        kwargs["model_path"] = os.getenv('TTS_MODEL_PATH', os.path.join(
            MODELS_DIR, "coqui/XTTS-v2"))
        kwargs["conf_file"] = os.getenv(
            'TTS_CONF_FILE', os.path.join(MODELS_DIR, "coqui/XTTS-v2/config.json"))
        kwargs["reference_audio_path"] = os.getenv('TTS_REFERENCE_AUDIO_PATH', os.path.join(
            RECORDS_DIR, "me.wav"))
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

    # TAG : config
    map_config_func = {
        'tts_coqui': get_tts_coqui_args,
        'tts_chat': get_tts_chat_args,
        'tts_edge': get_tts_edge_args,
        'tts_g': get_tts_g_args,
    }

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        # tts
        tag = os.getenv('TTS_TAG', "tts_chat")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initTTSEngine: {tag}, {engine}")
        return engine

    # TTS_TAG : stream_info
    map_tts_player_stream_info = {
        'tts_coqui': {
            "format_": pyaudio.paFloat32,
            "channels": 1,
            "rate": 24000,
        },
        'tts_chat': {
            "format_": pyaudio.paFloat32,
            "channels": 1,
            "rate": 24000,
        },
        'tts_edge': {
            "format_": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
        },
        'tts_g': {
            "format_": pyaudio.paInt16,
            "channels": 1,
            "rate": 22050,
        },
    }

    @staticmethod
    def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer | EngineClass:
        # player
        tag = os.getenv('PLAYER_TAG', "stream_player")
        tts_tag = os.getenv('TTS_TAG', "tts_chat")
        info = Env.map_tts_player_stream_info[tts_tag]
        if tts:
            info = tts.get_stream_info()
        info["sub_chunk_size"] = CHUNK * 10
        info["output_device_index"] = os.getenv('OUTPUT_DEVICE_INDEX', None)
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


def init_engines(object) -> dict:
    engines = {}
    for name, obj in inspect.getmembers(object, inspect.isfunction):
        if "init" not in name and "Engine" not in name:
            continue
        engines[name] = obj()
    return engines


r"""
CONF_ENV=local python -m src.cmd.init
CONF_ENV=local python -m src.cmd.init -o init_engine -i env

CONF_ENV=local python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_coqui python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_chat python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_g python -m src.cmd.init -o env2yaml
CONF_ENV=local TTS_TAG=tts_edge python -m src.cmd.init -o env2yaml

CONF_ENV=local python -m src.cmd.init -o init_engine -i config
CONF_ENV=local python -m src.cmd.init -o gather_load_configs
"""
if __name__ == "__main__":
    os.environ['CONF_ENV'] = 'local'
    os.environ['RECORDER_TAG'] = 'wakeword_rms_recorder'

    Logger.init(logging.INFO, is_file=False)

    def get_engines(init_type="env"):
        if init_type == "config":
            return init_engines(YamlConfig)
        return init_engines(Env)

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
