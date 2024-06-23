import multiprocessing
import multiprocessing.connection
import logging
import asyncio
import threading
import queue
import sys


from src.common.session import Session
from src.common import interface
from src.cmd import init


HISTORY_LIMIT = 10240


class Audio2AudioChatWorker:

    @classmethod
    def run(cls, conn: interface.IConnector, e: multiprocessing.Event = None):
        # vad_detector = init.initVADEngine()
        asr = init.initASREngine()
        llm = init.initLLMEngine()
        tts = init.initTTSEngine()

        th_q = queue.Queue()

        asr_llm_gen_t = threading.Thread(
            target=cls.loop_asr_llm_generate, args=(asr, llm, conn, th_q))
        asr_llm_gen_t.start()
        tts_synthesize_t = threading.Thread(
            target=cls.loop_tts_synthesize, args=(tts, conn, th_q))
        tts_synthesize_t.start()

        logging.info(f"init BE is ok")
        e and e.set()

        asr_llm_gen_t.join()
        tts_synthesize_t.join()

    @classmethod
    def loop_asr_llm_generate(
            cls,
            asr: interface.IAsr, llm: interface.ILlm,
            conn: interface.IConnector,
            text_buffer: queue.Queue):
        logging.info(f"loop_asr starting with asr: {asr}")
        print("start loop_asr_llm_generate...", flush=True, file=sys.stderr)
        while True:
            try:
                res = conn.recv('be')
                if res is None:
                    continue
                msg, frames, session = res
                if msg is None or msg.lower() == "stop":
                    break
                logging.info(f'Received: {msg} len(frames): {len(frames)}')
                asr.set_audio_data(frames)
                res = asyncio.run(asr.transcribe(session))
                logging.info(f'transcribe res: {res}')
                if len(res['text'].strip()) == 0:
                    raise Exception(
                        f"ASR transcribed text is empty sid: {session.ctx.client_id}")
                conn.send(("ASR_TEXT", res['text'], session), 'be')
                conn.send(("ASR_TEXT_DONE", "", session), 'be')

                logging.info(
                    f"recv session.chat_history: {session.chat_history}")
                session.chat_history.append(init.get_user_prompt(res['text']))
                while llm.count_tokens(init.create_prompt(session.chat_history)) > HISTORY_LIMIT:
                    session.chat_history.pop(0)
                    session.chat_history.pop(0)

                session.ctx.state["prompt"] = init.create_prompt(
                    session.chat_history)
                logging.info(
                    f"llm.generate prompt: {session.ctx.state['prompt']}")
                print(f"me: {session.ctx.state['prompt']}",
                      flush=True, file=sys.stdout)
                assistant_text = ""
                text_iter = llm.generate(session)
                for text in text_iter:
                    assistant_text += text
                    conn.send(("LLM_GENERATE_TEXT", text, session), 'be')
                    text_buffer.put_nowait(
                        ("LLM_GENERATE_TEXT", text, session))
                out = init.get_assistant_prompt(assistant_text)
                session.chat_history.append(out)
                bot_name = session.ctx.state["bot_name"] if "bot_name" in session.ctx.state else "bot"
                print(f"{bot_name}: {out}", flush=True, file=sys.stdout)
                logging.info(
                    f"send session.chat_history: {session.chat_history}")
                conn.send(("LLM_GENERATE_DONE", "", session), 'be')
            except Exception as ex:
                conn.send(
                    ("BE_EXCEPTION", f"asr_llm_generate's exception: {ex}", session), 'be')

    @classmethod
    def loop_tts_synthesize(
            cls,
            tts: interface.ITts,
            conn: interface.IConnector,
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
                        conn.send(("PLAY_FRAMES", chunk, session), 'be')
            except queue.Empty:
                logging.debug(
                    f"tts_synthesize's consumption queue is empty after block {q_get_timeout}s")
                continue
            except Exception as ex:
                conn.send(
                    ("BE_EXCEPTION", f"tts_synthesize's exception: {ex}", session), 'be')
