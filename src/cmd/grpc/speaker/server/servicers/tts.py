import logging
import json
import os
import threading
import queue

import grpc
import uuid

from src.common.grpc.idl.tts_pb2_grpc import TTSServicer
from src.common.grpc.idl.tts_pb2 import (
    GetStreamInfoReponse,
    GetStreamInfoRequest,
    GetVoicesRequest,
    GetVoicesResponse,
    LoadModelRequest,
    LoadModelResponse,
    SetVoiceRequest,
    SetVoiceResponse,
    SynthesizeRequest,
    SynthesizeResponse,
)
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, ITts
from src.modules.speech.tts import TTSEnvInit


def get_session_id(context: grpc.ServicerContext):
    for item in context.invocation_metadata():
        if item.key == "client_id":
            return item.value
    return str(uuid.uuid4())


class TTS(TTSServicer):
    def __init__(self) -> None:
        super().__init__()
        self._tts_instance: EngineClass | ITts = None
        self._is_initializing = False
        self._lock = threading.Lock()
        self.init_queue = queue.Queue()

    def get_tts_instance(self):
        """获取当前的 TTS 实例"""
        with self._lock:
            return self._tts_instance

    def LoadModel(self, request: LoadModelRequest, context: grpc.ServicerContext):
        logging.debug(f"LoadModel request: {request}")
        kwargs = json.loads(request.json_kwargs)
        logging.debug(f"LoadModel kwargs: {kwargs}")

        # print("--->", self.get_tts_instance(), request.is_reload)
        if self.get_tts_instance() and request.is_reload is False:
            logging.info("TTS is already initializing un reload, ignoring this signal.")
            return LoadModelResponse()

        with self._lock:
            if self._is_initializing:
                logging.info("TTS is already initializing, ignoring this signal.")
                return LoadModelResponse()

        init_event = threading.Event()
        init_result = {"success": False, "error": None}

        init_signal = {
            "tts_tag": request.tts_tag,
            "kwargs": kwargs,
            "event": init_event,
            "result": init_result,
        }
        self.init_queue.put(init_signal)
        # logging.info(f"Initialization signal added to queue: {init_signal}")

        if not init_event.wait(timeout=int(os.getenv("INIT_TIMEOUT", "600"))):
            # default wait 10 minutes
            logging.error("Initialization timed out.")
            raise Exception("TTS initialization timed out.")

        if not init_result["success"]:
            logging.error(f"Initialization failed: {init_result['error']}")
            raise Exception(f"TTS initialization failed: {init_result['error']}")

        logging.info("TTS initialization completed successfully.")
        return LoadModelResponse()

    def GetVoices(self, request: GetVoicesRequest, context: grpc.ServicerContext):
        voices = self.get_tts_instance().get_voices()
        return GetVoicesResponse(voices=voices)

    def SetVoice(self, request: SetVoiceRequest, context: grpc.ServicerContext):
        self.get_tts_instance().set_voice(request.voice)
        return SetVoiceResponse()

    def GetStreamInfo(self, request: GetStreamInfoRequest, context: grpc.ServicerContext):
        info = self.get_tts_instance().get_stream_info()
        return GetStreamInfoReponse(
            format=info["format"],
            rate=info["rate"],
            channels=info["channels"],
            sample_width=info["sample_width"],
        )

    def SynthesizeUS(self, request: SynthesizeRequest, context: grpc.ServicerContext):
        logging.debug(f"SynthesizeUS request: {request} json.kwargs:{request.json_kwargs}")
        session = Session(**SessionCtx(get_session_id(context)).__dict__)
        if request.json_kwargs:
            kwargs = json.loads(request.json_kwargs)
            logging.debug(f"SynthesizeUS kwargs: {kwargs}")
            session.ctx.state = kwargs
        session.ctx.state["tts_text"] = request.tts_text
        iter = self.get_tts_instance().synthesize_sync(session)
        for i, chunk in enumerate(iter):
            logging.debug(f"get {i} chunk {len(chunk)}")
            yield SynthesizeResponse(tts_audio=chunk)


def main_thread_init(tts_service: TTS):
    logging.info("Main thread initializing TTS engine.")
    while True:
        try:
            init_signal = tts_service.init_queue.get()
            logging.debug(f"Main thread received initialization signal: {init_signal}")
            tts_tag = init_signal["tts_tag"]
            kwargs = init_signal["kwargs"]
            init_event = init_signal["event"]
            init_result = init_signal["result"]

            with tts_service._lock:
                if tts_service._is_initializing:
                    logging.info("Duplicate initialization signal detected, skipping.")
                    continue
                tts_service._is_initializing = True

            try:
                tts_instance = TTSEnvInit.getEngine(tts_tag, **kwargs)
                logging.info(f"Main thread initialized TTS engine: {tts_instance.TAG}")
                with tts_service._lock:
                    tts_service._tts_instance = tts_instance
                init_result["success"] = True
            except Exception as e:
                logging.error(f"Failed to initialize TTS engine: {e}")
                init_result["success"] = False
                init_result["error"] = str(e)
            finally:
                with tts_service._lock:
                    tts_service._is_initializing = False

            init_event.set()

        except Exception as e:
            logging.error(f"Error in main thread initialization loop: {e}")
