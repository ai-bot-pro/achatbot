import multiprocessing.connection
import os
import logging
import sys

import pyaudio

from src.common import interface
from src.common.factory import EngineFactory, EngineClass
from src.common.types import MODELS_DIR, RECORDS_DIR, CHUNK
# need import engine class -> EngineClass.__subclasses__
import src.modules.speech
import src.core.llm

DEFAULT_SYSTEM_PROMPT = "你是一个中国人,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码以及数学公式。"


def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')


def on_wakeword_detected(session, data):
    if "bot_name" in session.ctx.state:
        print(f"{session.ctx.state['bot_name']}~ ",
              end="", flush=True, file=sys.stderr)


def initWakerEngine() -> interface.IDetector:
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
    kwargs["on_wakeword_detected"] = on_wakeword_detected
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    return engine


def initRecorderEngine() -> interface.IRecorder:
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


def initVADEngine() -> interface.IDetector:
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


def initASREngine() -> interface.IAsr:
    # asr
    tag = os.getenv('ASR_TAG', "whisper_timestamped_asr")
    kwargs = {}
    kwargs["model_name_or_path"] = os.getenv('ASR_MODEL_NAME_OR_PATH', 'base')
    kwargs["download_path"] = MODELS_DIR
    kwargs["verbose"] = True
    kwargs["language"] = "zh"
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    logging.info(f"initASREngine: {tag}, {engine}")
    return engine


def create_phi3_prompt(history: list[str],
                       system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                       init_message: str = None):
    prompt = f'<|system|>\n{system_prompt}</s>\n'
    if init_message:
        prompt += f"<|assistant|>\n{init_message}</s>\n"

    return prompt + "".join(history) + "<|assistant|>"


def create_prompt(history: list[str],  init_message: str = None):
    system_prompt: str = os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
    if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
        return create_phi3_prompt(history, system_prompt, init_message)

    return ""


def get_user_prompt(text):
    if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
        return (f"<|user|>\n{text}</s>\n")
    return ""


def get_assistant_prompt(text):
    if "phi-3" in os.getenv('LLM_MODEL_NAME', 'phi-3').lower():
        return (f"<|assistant|>\n{text}</s>\n")
    return ""


def initLLMEngine() -> interface.ILlm:
    # llm
    tag = os.getenv('LLM_TAG', "llm_llamacpp")
    kwargs = {}
    kwargs["model_name"] = os.getenv('LLM_MODEL_NAME', 'phi-3')
    kwargs["model_path"] = os.getenv('LLM_MODEL_PATH', os.path.join(
        MODELS_DIR, "Phi-3-mini-4k-instruct-q4.gguf"))
    kwargs["model_type"] = os.getenv('LLM_MODEL_TYPE', "chat")
    kwargs["n_threads"] = os.cpu_count()
    kwargs["verbose"] = True
    kwargs["llm_stream"] = False
    # if logger.getEffectiveLevel() != logging.DEBUG:
    #    kwargs["verbose"] = False
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    logging.info(f"initLLMEngine: {tag}, {engine}")
    return engine


def get_tts_coqui_config() -> dict:
    kwargs = {}
    kwargs["model_path"] = os.getenv('TTS_MODEL_PATH', os.path.join(
        MODELS_DIR, "coqui/XTTS-v2"))
    kwargs["conf_file"] = os.getenv(
        'TTS_CONF_FILE', os.path.join(MODELS_DIR, "coqui/XTTS-v2/config.json"))
    kwargs["reference_audio_path"] = os.getenv('TTS_REFERENCE_AUDIO_PATH', os.path.join(
        RECORDS_DIR, "tmp.wav"))
    return kwargs


def get_tts_chat_config() -> dict:
    kwargs = {}
    kwargs["local_path"] = os.getenv('LOCAL_PATH', os.path.join(
        MODELS_DIR, "2Noise/ChatTTS"))
    kwargs["source"] = os.getenv('TTS_CHAT_SOURCE', "local")
    return kwargs


# TAG : config
map_config_func = {
    'tts_coqui': get_tts_coqui_config,
    'tts_chat': get_tts_chat_config,
}


def initTTSEngine() -> interface.ITts:
    # tts
    tag = os.getenv('TTS_TAG', "tts_chat")
    kwargs = map_config_func[tag]()
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    logging.info(f"initTTSEngine: {tag}, {engine}")
    return engine


def initPlayerEngine(tts: interface.ITts = None) -> interface.IPlayer:
    # player
    tag = os.getenv('PLAYER_TAG', "stream_player")
    # info = tts.get_stream_info()
    info = {
        "format_": pyaudio.paFloat32,
        "channels": 1,
        "rate": 24000,
    }
    info["chunk_size"] = CHUNK * 10
    engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **info)
    logging.info(f"stream_info: {info}, initPlayerEngine: {tag},  {engine}")
    return engine
