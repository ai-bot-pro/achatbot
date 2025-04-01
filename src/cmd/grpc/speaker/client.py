import os
import random
import sys
from time import perf_counter
import uuid
import json
import logging
import traceback

import grpc
from dotenv import load_dotenv
import numpy as np
import soundfile
import librosa

try:
    cur_dir = os.path.dirname(__file__)
    sys.path.insert(1, os.path.join(cur_dir, "../../../common/grpc/idl"))
    from src.common.grpc.idl.tts_pb2_grpc import TTSStub
    from src.common.grpc.idl.tts_pb2 import (
        LoadModelRequest,
        LoadModelResponse,
        SynthesizeRequest,
        SynthesizeResponse,
        GetStreamInfoRequest,
        GetStreamInfoReponse,
        GetVoicesRequest,
        GetVoicesResponse,
        SetVoiceRequest,
        SetVoiceResponse,
    )
except ModuleNotFoundError as e:
    raise Exception(f"grpc import error: {e}")

from src.common.factory import EngineClass
from src.common.interface import IAudioStream
from src.modules.speech.audio_stream import AudioStreamEnvInit
from src.modules.speech.player import PlayerEnvInit
from src.modules.speech.tts import TTSEnvInit
from src.common.grpc.interceptors.authentication_client import add_authentication
from src.common.logger import Logger
from src.common.types import RECORDS_DIR, SessionCtx, PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.session import Session


load_dotenv(override=True)

Logger.init(
    os.getenv("LOG_LEVEL", "info").upper(),
    app_name="chat-bot-tts-client",
    is_file=False,
    is_console=True,
)

TTS_TEXT_LIST = [
    "hello, 你好，我是机器人",
    "万物之始,大道至简,衍化至繁。",
    "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。",
]


def load_model(tts_stub: TTSStub):
    tag = os.getenv("TTS_TAG", "tts_edge")
    is_reload = bool(os.getenv("IS_RELOAD", None))
    kwargs = TTSEnvInit.map_config_func[tag]()
    request = LoadModelRequest(tts_tag=tag, is_reload=is_reload, json_kwargs=json.dumps(kwargs))
    logging.debug(request)
    response = tts_stub.LoadModel(request)
    logging.debug(response)


def synthesize_us(tts_stub: TTSStub, tts_text="hello,你好，我是机器人"):
    tag = os.getenv("TTS_TAG", "tts_edge")
    if tag not in TTSEnvInit.map_synthesize_config_func:
        logging.warning(f"{tag} not in map_synthesize_config_func, use default config")
        kwargs = TTSEnvInit.get_tts_synth_args()
    else:
        kwargs = TTSEnvInit.map_synthesize_config_func[tag]()
    request_data = SynthesizeRequest(tts_text=tts_text, json_kwargs=json.dumps(kwargs))
    logging.debug(request_data)
    response_iterator = tts_stub.SynthesizeUS(request_data)
    for response in response_iterator:
        yield response.tts_audio


def get_stream_info(tts_stub: TTSStub):
    response: GetStreamInfoReponse = tts_stub.GetStreamInfo(GetStreamInfoRequest())
    return response


def get_voices(tts_stub: TTSStub):
    response: GetVoicesResponse = tts_stub.GetVoices(GetVoicesRequest())
    return response.voices


def set_voice(tts_stub: TTSStub, voice: str):
    response = tts_stub.SetVoice(SetVoiceRequest(voice=voice))
    return response


logging.basicConfig(level=logging.DEBUG)

"""
TTS_TAG=tts_edge python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_g python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_coqui python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_chat python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_cosy_voice python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_fishspeech python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_f5 python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_openvoicev2 python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_kokoro python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_onnx_kokoro KOKORO_ESPEAK_NG_LIB_PATH=/usr/local/lib/libespeak-ng.1.dylib KOKORO_LANGUAGE=cmn python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_cosy_voice2 \
    COSY_VOICE_MODELS_DIR=./models/FunAudioLLM/CosyVoice2-0.5B \
    COSY_VOICE_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_llasa IS_SAVE=1 python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_step IS_SAVE=1 python -m src.cmd.grpc.speaker.client

# instruct2speech
TTS_TAG=tts_minicpmo \
    LLM_DEVICE=cuda LLM_GEN_TEMPERATURE=0.5 LLM_TORCH_DTYPE=bfloat16 \
    LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6 \
    TTS_TASK=instruct2speech \
    IS_SAVE=1 python -m src.cmd.grpc.speaker.client

# voice_cloning
TTS_TAG=tts_minicpmo \
    LLM_DEVICE=cuda LLM_GEN_TEMPERATURE=0.5 LLM_TORCH_DTYPE=bfloat16 \
    LLM_MODEL_NAME_OR_PATH=./models/openbmb/MiniCPM-o-2_6 \
    TTS_TASK=voice_cloning \
    REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    IS_SAVE=1 python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_zonos \
    ZONOS_LM_CHECKPOINT_DIR=./models/Zyphra/Zonos-v0.1-transformer \
    ZONOS_DAC_MODEL_DIR=./models/descript/dac_44khz \
    SPEAKER_EMBEDDING_MODEL_DIR=./models/Zyphra/Zonos-v0.1-speaker-embedding
    ZONOS_REF_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    IS_SAVE=1 python -m src.cmd.grpc.speaker.client

# tts lm gen
TTS_TAG=tts_step IS_SAVE=1 \
    TTS_WARMUP_STEPS=2 TTS_LM_MODEL_PATH=./models/stepfun-ai/Step-Audio-TTS-3B \
    TTS_TOKENIZER_MODEL_PATH=./models/stepfun-ai/Step-Audio-Tokenizer \
    python -m src.cmd.grpc.speaker.client
# tts voice clone
TTS_TAG=tts_step IS_SAVE=1 \
    TTS_WARMUP_STEPS=2 TTS_LM_MODEL_PATH=/root/.achatbot/models/stepfun-ai/Step-Audio-TTS-3B \
    TTS_TOKENIZER_MODEL_PATH=/root/.achatbot/models/stepfun-ai/Step-Audio-Tokenizer \
    TTS_STREAM_FACTOR=2 \
    TTS_MODE=voice_clone \
    SRC_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_spark IS_SAVE=1 \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    TTS_LM_MODEL_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q2_K.gguf \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q3_K_L.gguf \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q4_K_M.gguf \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.Q8_0.gguf \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_llamacpp_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_PATH=./models/SparkTTS-LLM.f16.gguf \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_generator_spark IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_transformers_generator \
    TTS_MODEL_DIR=./models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_NAME_OR_PATH=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_LM_TOKENIZER_DIR=./models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_generator_spark SERVE_ADDR=r21.modal.host:43783 IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_vllm_generator \
    LLM_TORCH_DTYPE=bfloat16 \
    TTS_MODEL_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_NAME_OR_PATH=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_LM_TOKENIZER_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_generator_spark SERVE_ADDR=r21.modal.host:43783 IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_sglang_generator \
    LLM_TORCH_DTYPE=bfloat16 \
    TTS_MODEL_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_NAME_OR_PATH=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_LM_TOKENIZER_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_generator_spark SERVE_ADDR=r21.modal.host:43783 IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_trtllm_generator \
    TTS_MODEL_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_NAME_OR_PATH=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    TTS_LM_TOKENIZER_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client
TTS_TAG=tts_generator_spark SERVE_ADDR=r21.modal.host:43783 IS_SAVE=1 \
    TTS_LM_GENERATOR_TAG=llm_trtllm_runner_generator \
    TTS_MODEL_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B \
    LLM_MODEL_NAME_OR_PATH=/root/.achatbot/trt_models/tts-spark/trt_engines_bfloat16 \
    TTS_LM_TOKENIZER_DIR=/root/.achatbot/models/SparkAudio/Spark-TTS-0.5B/LLM \
    python -m src.cmd.grpc.speaker.client

TTS_TAG=tts_orpheus IS_SAVE=1 \
    LM_MODEL_PATH=./models/canopylabs/orpheus-3b-0.1-ft \
    CODEC_MODEL_PATH=./models/hubertsiuzdak/snac_24khz \
    python -m src.cmd.grpc.speaker.client
"""
if __name__ == "__main__":
    player = None
    channel = None
    is_save = bool(os.getenv("IS_SAVE", ""))
    tts_tag = os.getenv("TTS_TAG")
    try:
        client_id = str(uuid.uuid4())
        session = Session(**SessionCtx(client_id).__dict__)
        # todo: up to the rpc gateway to auth
        token = "oligei-tts"
        authentication = add_authentication("authorization", token)
        serve_addr = os.getenv("SERVE_ADDR", "localhost:50052")
        channel = grpc.insecure_channel(serve_addr)
        channel = grpc.intercept_channel(channel, authentication)
        tts_stub = TTSStub(channel)

        load_model(tts_stub)

        stream_info = get_stream_info(tts_stub)
        logging.debug(f"stream_info:{stream_info}")

        voices = get_voices(tts_stub)
        logging.debug(f"voices:{voices}")

        # voice must match language to select
        # voice = random.choice(voices)
        # logging.debug(f"set voice:{voice}")
        # set_voice(tts_stub, voice)

        if is_save is False:
            audio_out_stream: IAudioStream | EngineClass = (
                AudioStreamEnvInit.initAudioOutStreamEngine()
            )
            player = PlayerEnvInit.initPlayerEngine()
            player.set_out_stream(audio_out_stream)
            player.open()
            player.start(session)

        for idx, tts_text in enumerate(TTS_TEXT_LIST):
            tts_audio_iter = synthesize_us(tts_stub, tts_text=tts_text)
            times = []
            res = bytearray()
            start_time = perf_counter()
            for i, tts_audio in enumerate(tts_audio_iter):
                times.append(perf_counter() - start_time)
                start_time = perf_counter()
                logging.info(f"{i} play tts_chunk len:{len(tts_audio)}")
                session.ctx.state["tts_chunk"] = tts_audio
                if is_save is False:
                    player.play_audio(session)
                else:
                    res.extend(tts_audio)

            if is_save is False:
                player.stop(session)

            if len(res) > 0:
                file_name = f"grpc_{tts_tag}_{idx}.wav"
                os.makedirs(RECORDS_DIR, exist_ok=True)
                file_path = os.path.join(RECORDS_DIR, file_name)
                np_dtype = np.float32
                if stream_info.format == PYAUDIO_PAINT16:
                    np_dtype = np.int16
                data = np.frombuffer(res, dtype=np_dtype)
                data, _ = librosa.effects.trim(data, top_db=60)
                soundfile.write(file_path, data, stream_info.rate)
                logging.info(f"[ {tts_text} ] save audio stream to {file_path}")
                info = soundfile.info(file_path, verbose=True)
                print(
                    f"tts cost time {sum(times)} s, wav duration {info.duration} s, RTF: {sum(times)/info.duration}"
                )

            if is_save is False:
                channel and channel.close()
                player and player.close()

    except grpc.RpcError as e:
        logging.error(f"grpc.RpcError: {e}")
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Exception: {e}; traceback: {tb_str}")
