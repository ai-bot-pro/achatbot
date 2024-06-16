import multiprocessing
import multiprocessing.connection
import os
import logging
import asyncio
import threading

import uuid

from src.common.session import Session
from src.common.logger import Logger
from src.common import interface
from src.common.utils.audio_utils import save_audio_to_file
from src.common.factory import EngineFactory, EngineClass
from src.common.types import SessionCtx,  MODELS_DIR, RECORDS_DIR, CHUNK
# need import engine class -> EngineClass.__subclasses__
import src.modules.speech
import src.core.llm

# global logging for fork processes
Logger.init(logging.DEBUG)


def initRecorderEngine() -> interface.IRecorder:
    # recorder
    tag = os.getenv('RECODER_TAG', "rms_recorder")
    kwargs = {}
    kwargs["input_device_index"] = int(os.getenv('MIC_IDX', "1"))
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    logging.info(f"initRecorderEngine: {tag}, {engine}")
    return engine


def initVADEngine() -> interface.IDetector:
    # vad detector
    tag = os.getenv('DETECTOR_TAG', "pyannote_vad")
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
    engine = EngineFactory.get_engine_by_tag(
        EngineClass, tag, **kwargs)
    logging.info(f"initASREngine: {tag}, {engine}")
    return engine


def initLLMEngine() -> interface.ILlm:
    # llm
    tag = os.getenv('LLM_TAG', "llm_llamacpp")
    kwargs = {}
    kwargs["model_path"] = os.getenv('LLM_MODEL_PATH', os.path.join(
        MODELS_DIR, "Phi-3-mini-4k-instruct-q4.gguf"))
    kwargs["model_type"] = os.getenv('LLM_MODEL_TYPE', "chat")
    kwargs["n_threads"] = os.cpu_count()
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


def initPlayerEngine(tts: interface.ITts) -> interface.IPlayer:
    # player
    tag = os.getenv('PLAYER_TAG', "stream_player")
    info = tts.get_stream_info()
    info["chunk_size"] = CHUNK * 10
    engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **info)
    logging.info(f"stream_info: {info}, initPlayerEngine: {tag},  {engine}")
    return engine


def record_audio(recorder: interface.IRecorder, session: Session, conn: multiprocessing.connection.Connection):
    frames = recorder.record_audio(session)
    data = b''.join(frames)
    conn.send(("RECORD_FRAMES", data, session))
    asyncio.run(save_audio_to_file(
        data, f"{session.ctx.client_id}.wav", audio_dir=RECORDS_DIR))


def loop_record(conn: multiprocessing.connection.Connection, e: multiprocessing.Event):
    recorder = initRecorderEngine()
    sid = uuid.uuid4()
    session = Session(**SessionCtx(sid).__dict__)
    logging.info(f"loop_record starting with session: {session}")
    while True:
        e.clear()
        record_audio(recorder, session, conn)
        e.wait()


def loop_asr(conn: multiprocessing.connection.Connection, q: multiprocessing.Queue):
    # vad_detector = initVADEngine()
    asr = initASREngine()
    logging.info(f"loop_asr starting with asr: {asr}")
    while True:
        msg, frames, session = conn.recv()
        if msg is None or msg.lower() == "stop":
            break
        logging.info(f'Received: {msg}')

        session.ctx.language = "zh"
        asr.set_audio_data(frames)
        res = asyncio.run(asr.transcribe(session))
        if len(res['text']) > 0:
            q.put_nowait(("LLM_TEXT", res['text'], session))


def loop_llm_generate(
        llm: interface.ILlm,
        tts: interface.ITts,
        player: interface.IPlayer,
        q: multiprocessing.Queue, e: multiprocessing.Event):
    logging.info(f"loop_llm_generate starting with {q}")
    while True:
        if q:
            msg, content, session = q.get()
            if msg is None or msg.lower() == "stop":
                break
            logging.info(f"content {content}")
            llm.args.llm_stream = True
            session.ctx.state["prompt"] = content
            logging.debug(f"session.ctx {session.ctx}")
            text_iter = llm.generate(session)
            for text in text_iter:
                logging.info(f"tts_text: {text}")
                session.ctx.state["tts_text"] = text
                tts.args.tts_stream = False
                audio_iter = tts.synthesize(session)
                for i, chunk in enumerate(audio_iter):
                    logging.info(f"synthesize audio {i} chunk {len(chunk)}")
                    session.ctx.state["tts_chunk"] = chunk
                    player.play_audio(session)

            e.set()


def main():
    llm = initLLMEngine()
    tts = initTTSEngine()
    player = initPlayerEngine(tts)

    mp_queue = multiprocessing.Queue()
    start_record_event = multiprocessing.Event()
    parent_conn, child_conn = multiprocessing.Pipe()

    # FE
    p = multiprocessing.Process(
        target=loop_record, args=(child_conn, start_record_event))
    p.start()

    # BE
    c = multiprocessing.Process(
        target=loop_asr, args=(parent_conn, mp_queue))
    c.start()
    t = threading.Thread(target=loop_llm_generate,
                         args=(llm, tts, player, mp_queue, start_record_event))
    t.start()

    t.join()
    c.join()
    p.join()

    mp_queue.close()
    c.close()
    p.close()


if __name__ == "__main__":
    main()
