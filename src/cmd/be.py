import multiprocessing
import multiprocessing.connection
import logging
import asyncio
import threading
import queue
import sys


from src.common import interface
import src.cmd.init


HISTORY_LIMIT = 10240


def run_be(conn: interface.IConnector, e: multiprocessing.Event):
    # vad_detector = src.cmd.init.initVADEngine()
    asr = src.cmd.init.initASREngine()
    llm = src.cmd.init.initLLMEngine()
    tts = src.cmd.init.initTTSEngine()

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
                          conn: interface.IConnector,
                          text_buffer: queue.Queue):
    history = []
    logging.info(f"loop_asr starting with asr: {asr}")
    print("start loop_asr_llm_generate...", flush=True, file=sys.stderr)
    while True:
        try:
            msg, frames, session = conn.recv('be')
            if msg is None or msg.lower() == "stop":
                break
            logging.info(f'Received: {msg} len(frames): {len(frames)}')
            asr.set_audio_data(frames)
            res = asyncio.run(asr.transcribe(session))
            logging.info(f'transcribe res: {res}')
            if len(res['text'].strip()) == 0:
                raise Exception(
                    f"asr.transcribe res['text'] is empty sid: {session.ctx.client_id}")
            conn.send(("ASR_TEXT", res['text'], session), 'be')
            conn.send(("ASR_TEXT_DONE", "", session), 'be')

            history.append(src.cmd.init.get_user_prompt(res['text']))
            while llm.count_tokens(src.cmd.init.create_prompt(history)) > HISTORY_LIMIT:
                history.pop(0)
                history.pop(0)

            session.ctx.state["prompt"] = src.cmd.init.create_prompt(history)
            logging.info(f"llm.generate prompt: {session.ctx.state['prompt']}")
            print(f"me: {session.ctx.state['prompt']}",
                  flush=True, file=sys.stdout)
            assistant_text = ""
            text_iter = llm.generate(session)
            for text in text_iter:
                assistant_text += text
                conn.send(("LLM_GENERATE_TEXT", text, session), 'be')
                text_buffer.put_nowait(
                    ("LLM_GENERATE_TEXT", text, session))
            out = src.cmd.init.get_assistant_prompt(assistant_text)
            history.append(out)
            bot_name = session.ctx.state["bot_name"] if "bot_name" in session.ctx.state else "bot"
            print(f"{bot_name}: {out}", flush=True, file=sys.stdout)
            conn.send(("LLM_GENERATE_DONE", "", session), 'be')
        except Exception as ex:
            conn.send(
                ("BE_EXCEPTION", f"asr_llm_generate's exception: {ex}", session), 'be')


def loop_tts_synthesize(
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
