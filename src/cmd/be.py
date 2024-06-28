import multiprocessing.connection
import multiprocessing
import threading
import logging
import asyncio
import queue
import sys
import os


from src.common.session import Session
from src.common.factory import EngineClass
from src.common import interface
if os.getenv("INIT_TYPE", 'env') == 'yaml_config':
    from src.cmd.init import YamlConfig as init
else:
    from src.cmd.init import Env as init


HISTORY_LIMIT = 10240


class Audio2AudioChatWorker:

    def run(self, conn: interface.IConnector, e: multiprocessing.Event = None):
        # self.vad_detector: interface.IDetector | EngineClass = init.initVADEngine()
        self.asr: interface.IAsr | EngineClass = init.initASREngine()
        self.llm: interface.ILlm | EngineClass = init.initLLMEngine()
        self.tts: interface.ITts | EngineClass = init.initTTSEngine()
        self.model_name = self.llm.model_name()

        th_q = queue.Queue()

        asr_llm_gen_t = threading.Thread(
            target=self.loop_asr_llm_generate, args=(conn, th_q))
        asr_llm_gen_t.start()
        tts_synthesize_t = threading.Thread(
            target=self.loop_tts_synthesize, args=(conn, th_q))
        tts_synthesize_t.start()

        logging.info(f"init BE is ok")
        e and e.set()

        asr_llm_gen_t.join()
        tts_synthesize_t.join()

    def loop_asr_llm_generate(
            self,
            conn: interface.IConnector,
            text_buffer: queue.Queue):
        logging.info(f"loop_asr starting with asr: {self.asr}")
        print(f"start loop_asr_llm_generate with {self.asr.TAG} {self.llm.TAG} {self.model_name} ...",
              flush=True, file=sys.stderr)
        while True:
            try:
                res = conn.recv('be')
                if res is None:
                    continue
                msg, frames, session = res
                if msg is None or msg.lower() == "stop":
                    break
                logging.info(f'Received: {msg} len(frames): {len(frames)}')
                self.asr.set_audio_data(frames)
                res = asyncio.run(self.asr.transcribe(session))
                logging.info(f'transcribe res: {res}')
                if len(res['text'].strip()) == 0:
                    raise Exception(
                        f"ASR transcribed text is empty sid: {session.ctx.client_id}")
                conn.send(("ASR_TEXT", res['text'], session), 'be')
                conn.send(("ASR_TEXT_DONE", "", session), 'be')

                self.llm_generate(res['text'], session, conn, text_buffer)

            except Exception as ex:
                conn.send(
                    ("BE_EXCEPTION", f"asr_llm_generate's exception: {ex}", session), 'be')

    def loop_tts_synthesize(
            self,
            conn: interface.IConnector,
            text_buffer: queue.Queue):
        print(
            f"start loop_tts_synthesize with {self.tts.TAG} ...", flush=True, file=sys.stderr)
        q_get_timeout = 0.1
        while True:
            try:
                msg, text, session = text_buffer.get(timeout=q_get_timeout)
                if msg is None or msg.lower() == "stop":
                    break
                if msg == "LLM_GENERATE_DONE":
                    conn.send(("PLAY_FRAMES_DONE", "", session), 'be')
                    logging.info(f"PLAY_FRAMES_DONE")
                    continue

                if len(text.strip()) == 0:
                    raise Exception(
                        f"tts_synthesize msg:{msg} text is empty sid:{session.ctx.client_id}")
                logging.info(f"tts_text: {text}")
                session.ctx.state["tts_text"] = text
                audio_iter = self.tts.synthesize_sync(session)
                for i, chunk in enumerate(audio_iter):
                    logging.info(
                        f"synthesize audio {i} chunk {len(chunk)}")
                    if len(chunk) > 0:
                        conn.send(("PLAY_FRAMES", chunk, session), 'be')
            except queue.Empty:
                logging.debug(
                    f"tts_synthesize's consumption queue is empty after block {q_get_timeout}s")
                continue
            except Exception as ex:
                conn.send(
                    ("BE_EXCEPTION", f"tts_synthesize's exception: {ex}", session), 'be')

    def llm_generate(self, text: str,
                     session: Session,
                     conn: interface.IConnector,
                     text_buffer: queue.Queue):
        logging.info(f"recv session.chat_history: {session.chat_history}")

        user_prompt = init.get_user_prompt(self.model_name, text)
        user_prompt and session.chat_history.append(user_prompt)
        prompt = init.create_prompt(self.model_name, session.chat_history)
        while prompt and self.llm.count_tokens(prompt) > HISTORY_LIMIT:
            session.chat_history.pop(0)
            session.chat_history.pop(0)
            prompt = init.create_prompt(self.model_name, session.chat_history)

        session.ctx.state["prompt"] = prompt if prompt else text
        logging.info(f"llm.generate prompt: {session.ctx.state['prompt']}")
        print(f"me: {session.ctx.state['prompt']}",
              flush=True, file=sys.stdout)

        assistant_text = ""
        text_iter = self.llm.generate(session)
        for text in text_iter:
            assistant_text += text
            conn.send(("LLM_GENERATE_TEXT", text, session), 'be')
            text_buffer.put_nowait(("LLM_GENERATE_TEXT", text, session))

        logging.info(f"llm generate assistant_text: {assistant_text}")
        assistant_prompt = init.get_assistant_prompt(
            self.model_name, assistant_text)
        out = assistant_prompt if assistant_prompt else assistant_text
        if assistant_prompt:
            session.chat_history.append(assistant_prompt)
        bot_name = session.ctx.state["bot_name"] if "bot_name" in session.ctx.state else "bot"
        print(f"{bot_name}: {out}", flush=True, file=sys.stdout)

        conn.send(("LLM_GENERATE_DONE", "", session), 'be')
        text_buffer.put_nowait(("LLM_GENERATE_DONE", "", session))
        logging.info(f"send session.chat_history: {session.chat_history}")

    def llm_chat(self, text, session: Session,
                 conn: interface.IConnector,
                 text_buffer: queue.Queue):
        pass
