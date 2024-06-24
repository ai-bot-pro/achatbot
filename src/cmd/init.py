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


DEFAULT_SYSTEM_PROMPT = "你是一个中国人,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"


class Env:

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
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
        return engine

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        # recorder
        tag = os.getenv('RECORDER_TAG', "rms_recorder")
        kwargs = {}
        input_device_index = os.getenv('MIC_IDX', None)
        kwargs["input_device_index"] = None if input_device_index is None else int(
            input_device_index)
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
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
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
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
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
        logging.info(f"initASREngine: {tag}, {engine}")
        return engine

    @staticmethod
    def create_phi3_prompt(history: list[str],
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                           init_message: str = None):
        prompt = f'<|system|>\n{system_prompt}</s>\n'
        if init_message:
            prompt += f"<|assistant|>\n{init_message}</s>\n"

        return prompt + "".join(history) + "<|assistant|>"

    @staticmethod
    def create_prompt(history: list[str],  init_message: str = None):
        system_prompt: str = os.getenv(
            'LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
        if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
            return Env.create_phi3_prompt(history, system_prompt, init_message)

        return ""

    @staticmethod
    def get_user_prompt(text):
        if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
            return (f"<|user|>\n{text}</s>\n")
        return ""

    @staticmethod
    def get_assistant_prompt(text):
        if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
            return (f"<|assistant|>\n{text}</s>\n")
        return ""

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
        kwargs["verbose"] = True
        kwargs["llm_stream"] = False
        # if logger.getEffectiveLevel() != logging.DEBUG:
        #    kwargs["verbose"] = False
        kwargs['llm_stop'] = ["<|end|>", "</s>", "/s>",
                              "</s", "<s>", "<|user|>", "<|assistant|>", "<|system|>"]
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
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
            RECORDS_DIR, "tmp.wav"))
        return kwargs

    def get_tts_chat_args() -> dict:
        kwargs = {}
        kwargs["local_path"] = os.getenv('LOCAL_PATH', os.path.join(
            MODELS_DIR, "2Noise/ChatTTS"))
        kwargs["source"] = os.getenv('TTS_CHAT_SOURCE', "local")
        return kwargs

    # TAG : config
    map_config_func = {
        'tts_coqui': get_tts_coqui_args,
        'tts_chat': get_tts_chat_args,
    }

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        # tts
        tag = os.getenv('TTS_TAG', "tts_chat")
        kwargs = Env.map_config_func[tag]()
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, tag, **kwargs)
        logging.info(f"initTTSEngine: {tag}, {engine}")
        return engine

    @staticmethod
    def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer | EngineClass:
        # player
        tag = os.getenv('PLAYER_TAG', "stream_player")
        info = {
            "format_": pyaudio.paFloat32,
            "channels": 1,
            "rate": 24000,
        }
        if tts:
            info = tts.get_stream_info()
        info["chunk_size"] = CHUNK * 10
        engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **info)
        logging.info(
            f"stream_info: {info}, initPlayerEngine: {tag},  {engine}")
        return engine

    @classmethod
    async def save_obj_to_yaml(cls, name, obj, tag=None):
        if "init" not in name and "Engine" not in name:
            return
        engine = obj()
        if tag and tag not in engine.TAG:
            return
        # u can use local, dev, test, stress, gray, online
        env = os.getenv('CONF_ENV', "env")
        file_dir = os.path.join(CONFIG_DIR, env)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = os.path.join(file_dir, f"{engine.TAG}.yaml")
        logging.debug(f"obj: {obj}, engine: {engine}, file_path: {file_path}")
        await Conf.save_to_yaml(engine.args.__dict__, file_path)
        return name[4:len(name)-6].lower(), file_path, engine.TAG

    @classmethod
    async def save_to_yaml(cls, tag=None):
        async_tasks = []
        for name, obj in inspect.getmembers(cls, inspect.isfunction):
            async_task = asyncio.create_task(
                cls.save_obj_to_yaml(name, obj, tag))
            async_tasks.append(async_task)
            # await async_task
        logging.debug(f"{async_tasks}, len(async_tasks):{len(async_tasks)}")
        res = await asyncio.gather(*async_tasks, return_exceptions=True)
        manifests = {}
        items = []
        for item in res:
            if item is None:
                continue
            items.append(item)
            name, file_path, _tag = item
            manifests[name] = {"file_path": file_path, "tag": _tag}

        env = os.getenv('CONF_ENV', "env")
        file_path = os.path.join(CONFIG_DIR, env, "manifests.yaml")
        await Conf.save_to_yaml(manifests, file_path)
        items.append(("manifests", file_path, "manifests"))
        return items


class Config:

    @staticmethod
    def initWakerEngine() -> interface.IDetector | EngineClass:
        pass

    @staticmethod
    def initRecorderEngine() -> interface.IRecorder | EngineClass:
        pass

    @staticmethod
    def initVADEngine() -> interface.IDetector | EngineClass:
        pass

    @staticmethod
    def initASREngine() -> interface.IAsr | EngineClass:
        pass

    @staticmethod
    def initLLMEngine() -> interface.ILlm | EngineClass:
        pass

    @staticmethod
    def initTTSEngine() -> interface.ITts | EngineClass:
        pass

    @staticmethod
    def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer | EngineClass:
        pass


r"""
CONF_ENV=local python -m src.cmd.init
"""
if __name__ == "__main__":
    os.environ['RECORDER_TAG'] = 'wakeword_rms_recorder'
    os.environ['CONF_ENV'] = 'local'
    Logger.init(logging.INFO, is_file=False)
    res = asyncio.run(Env.save_to_yaml())
    for file_path in res:
        print(file_path)
