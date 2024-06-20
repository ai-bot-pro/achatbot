r"""
TQDM_DISABLE=True python -m src.cmd.local-terminal-chat.generate_audio2audio > std_out.log
"""
import multiprocessing
import multiprocessing.connection
import os
import logging
import asyncio
import threading
import queue
import sys

import uuid
import pyaudio

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
logger = Logger.init(logging.INFO, is_file=True, is_console=False)


def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')


def initWakerEngine() -> interface.IDetector:
    # waker
    recorder_tag = os.getenv('RECORDER_TAG', "rms_recorder")
    if "waker" not in recorder_tag:
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


def initLLMEngine() -> interface.ILlm:
    # llm
    tag = os.getenv('LLM_TAG', "llm_llamacpp")
    kwargs = {}
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


def loop_record(conn: multiprocessing.connection.Connection, e: threading.Event):
    recorder = initRecorderEngine()
    sid = uuid.uuid4()
    session = Session(**SessionCtx(sid).__dict__)
    session.ctx.waker = initWakerEngine()
    logging.info(f"loop_record starting with session ctx: {session.ctx}")
    print("start loop_record...", flush=True, file=sys.stderr)
    while True:
        try:
            print(f"-- chat round {session.chat_round} --",
                  flush=True, file=sys.stdout)
            print("\nme >> ", end="", flush=True, file=sys.stderr)
            e.clear()

            frames = recorder.record_audio(session)
            data = b''.join(frames)
            conn.send(("RECORD_FRAMES", data, session))
            asyncio.run(save_audio_to_file(
                data, session.get_file_name(), audio_dir=RECORDS_DIR))
            session.increment_file_counter()
            session.increment_chat_round()

            e.wait()
        except Exception as ex:
            logging.warning(
                f"loop_record Exception {ex} sid:{session.ctx.client_id}")


def loop_play(conn: multiprocessing.connection.Connection, e: threading.Event):
    player = initPlayerEngine()
    print("start loop_play...", flush=True, file=sys.stderr)
    llm_gen_segments = 0
    while True:
        try:
            msg, recv_data, session = conn.recv()
            if msg is None or msg.lower() == "stop":
                break
            if msg == "PLAY_FRAMES":
                session.ctx.state["tts_chunk"] = recv_data
                player.play_audio(session)
                e.set()
                llm_gen_segments = 0
            elif msg == "LLM_GENERATE_TEXT":
                if llm_gen_segments == 0:
                    bot_name = session.ctx.state["bot_name"] if "bot_name" in session.ctx.state else "bot"
                    print(f"\n{bot_name} >> ", end="",
                          flush=True, file=sys.stderr)
                print(recv_data.strip(), end="", flush=True, file=sys.stderr)
                llm_gen_segments += 1
            elif msg == "LLM_GENERATE_DONE":
                print("\n", end="", flush=True, file=sys.stderr)
                llm_gen_segments = 0
            elif msg == "ASR_TEXT":
                print(recv_data.strip(), end="", flush=True, file=sys.stderr)
            elif msg == "ASR_TEXT_DONE":
                print("\n", end="", flush=True, file=sys.stderr)
            elif msg == "BE_EXCEPTION":
                print(f"\nBE exception: {recv_data.strip()}",
                      end="", flush=True, file=sys.stderr)
                e.set()
                llm_gen_segments = 0
            else:
                logging.warning(f"unsupport msg {msg}")
        except Exception as ex:
            logging.warning(f"loop_play Exception {ex}")
            e.set()
            llm_gen_segments = 0


def run_be(conn: multiprocessing.connection.Connection, e: multiprocessing.Event):
    # vad_detector = initVADEngine()
    asr = initASREngine()
    llm = initLLMEngine()
    tts = initTTSEngine()

    th_q = queue.Queue()

    asr_llm_gen_t = threading.Thread(
        target=loop_asr_llm_generate, args=(asr, llm, conn, th_q))
    asr_llm_gen_t.start()
    tts_synthesize_t = threading.Thread(
        target=loop_tts_synthesize, args=(tts, conn, th_q))
    tts_synthesize_t.start()

    logging.info(f"init BE is ok")
    e.set()

    asr_llm_gen_t.join()
    tts_synthesize_t.join()


def loop_asr_llm_generate(asr: interface.IAsr, llm: interface.ILlm,
                          conn: multiprocessing.connection.Connection,
                          text_buffer: queue.Queue):
    logging.info(f"loop_asr starting with asr: {asr}")
    print("start loop_asr_llm_generate...", flush=True, file=sys.stderr)
    while True:
        try:
            msg, frames, session = conn.recv()
            if msg is None or msg.lower() == "stop":
                break
            logging.info(f'Received: {msg} len(frames): {len(frames)}')
            asr.set_audio_data(frames)
            res = asyncio.run(asr.transcribe(session))
            logging.info(f'transcribe res: {res}')
            if len(res['text'].strip()) == 0:
                raise Exception(
                    f"asr.transcribe res['text'] is empty sid: {session.ctx.client_id}")
            conn.send(("ASR_TEXT", res['text'], session))
            conn.send(("ASR_TEXT_DONE", "", session))

            session.ctx.state["prompt"] = res['text']
            text_iter = llm.generate(session)
            for text in text_iter:
                conn.send(("LLM_GENERATE_TEXT", text, session))
                text_buffer.put_nowait(
                    ("LLM_GENERATE_TEXT", text, session))
            conn.send(("LLM_GENERATE_DONE", "", session))
        except Exception as ex:
            conn.send(
                ("BE_EXCEPTION", f"asr_llm_generate's exception: {ex}", session))


def loop_tts_synthesize(
        tts: interface.ITts,
        conn: multiprocessing.connection.Connection,
        text_buffer: queue.Queue):
    print("start loop_tts_synthesize...", flush=True, file=sys.stderr)
    q_get_timeout = 1
    while True:
        try:
            msg, text, session = text_buffer.get(timeout=q_get_timeout)
            if msg is None or msg.lower() == "stop":
                break
            if len(text.strip()) == 0:
                raise Exception(
                    f"tts_synthesize text is empty sid:{session.ctx.client_id}")

            logging.info(f"tts_text: {text}")
            session.ctx.state["tts_text"] = text
            tts.args.tts_stream = False
            audio_iter = tts.synthesize(session)
            for i, chunk in enumerate(audio_iter):
                logging.info(f"synthesize audio {i} chunk {len(chunk)}")
                if len(chunk) > 0:
                    conn.send(("PLAY_FRAMES", chunk, session))
        except queue.Empty:
            logging.debug(
                f"tts_synthesize's consumption queue is empty after block {q_get_timeout}s")
            continue
        except Exception as ex:
            conn.send(
                ("BE_EXCEPTION", f"tts_synthesize's exception: {ex}", session))


def main():
    parent_conn, child_conn = multiprocessing.Pipe()

    # BE
    be_init_event = multiprocessing.Event()
    c = multiprocessing.Process(
        target=run_be, args=(child_conn, be_init_event))
    c.start()
    be_init_event.wait()

    # FE
    start_record_event = threading.Event()
    play_t = threading.Thread(target=loop_play,
                              args=(parent_conn, start_record_event))
    play_t.start()
    record_t = threading.Thread(target=loop_record,
                                args=(parent_conn, start_record_event))
    record_t.start()

    record_t.join()
    play_t.join()
    c.join()
    c.close()


if __name__ == "__main__":
    main()
