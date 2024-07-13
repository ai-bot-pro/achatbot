import logging

import grpc
import uuid

from src.cmd.grpc.idl.tts_pb2_grpc import TTSServicer
from src.cmd.grpc.idl.tts_pb2 import (
    LoadModelRequest, LoadModelResponse,
    SynthesizeRequest, SynthesizeResponse,
)
from src.common.factory import EngineFactory, EngineClass
from src.common.session import Session
from src.common.types import SessionCtx, ITts
import src.modules.speech.tts


def get_session_id(context: grpc.ServicerContext):
    if context.invocation_metadata.key == "client_id":
        return context.invocation_metadata.key
    return uuid.uuid4()


class TTS(TTSServicer):
    def __init__(self,) -> None:
        super().__init__()
        self.tts = None

    def LoadModel(self, request: LoadModelRequest, context: grpc.ServicerContext):
        kwargs = request.kwargs
        if self.tts is not None and request.is_reload is False:
            return LoadModelResponse()
        self.tts: EngineClass | ITts = EngineFactory.get_engine_by_tag(
            EngineClass, self.tts_tag, **kwargs)
        return LoadModelResponse()

    def SynthesizeUS(self,
                     request: SynthesizeRequest,
                     context: grpc.ServicerContext):
        logging.info(f"SynthesizeUS request: {request}")
        self.session = Session(**SessionCtx(get_session_id(context)).__dict__)
        self.session.ctx.state["tts_text"] = request.tts_text
        iter = self.tts.synthesize_sync(self.session)
        for i, chunk in enumerate(iter):
            logging.debug(f"get {i} chunk {len(chunk)}")
            yield SynthesizeResponse(tts_audio=chunk)
