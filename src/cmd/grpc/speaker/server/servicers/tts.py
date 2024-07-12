import logging

import grpc

from src.cmd.grpc.idl.tts_pb2_grpc import TTSServicer
from src.cmd.grpc.idl.tts_pb2 import SynthesizeRequest, SynthesizeResponse


class TTS(TTSServicer):

    def SynthesizeUS(
            self,
            request: SynthesizeRequest,
            context: grpc.ServicerContext):
        logging.info(f"SynthesizeUS request: {request}")
        tts_audio = bytes(b'hello')
        yield SynthesizeResponse(tts_audio=tts_audio)
