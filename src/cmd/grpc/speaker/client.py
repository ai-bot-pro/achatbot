import logging

import grpc

from src.cmd.grpc.idl.tts_pb2 import SynthesizeRequest, SynthesizeResponse
from src.cmd.grpc.idl.tts_pb2_grpc import TTSStub
from src.cmd.grpc.interceptors.authentication_client import add_authentication


def synthesize_us(channel):
    tts_stub = TTSStub(channel)
    request_data = SynthesizeRequest(tts_tag="tts_edge", tts_text="hello world", kawa_params={})
    response_iterator = tts_stub.SynthesizeUS(request_data)
    for response in response_iterator:
        print(len(response.tts_audio))


logging.basicConfig(level=logging.DEBUG)

# python -m src.cmd.grpc.speaker.client 2> ./log/tts_cli_err.log
if __name__ == "__main__":
    try:
        # todo: up to the rpc gateway to auth
        token = "oligei-tts"
        authentication = add_authentication('authorization', token)
        channel = grpc.insecure_channel('localhost:50052')
        channel = grpc.intercept_channel(channel, authentication)

        ops = [synthesize_us]
        for op in ops:
            print(f"--{op}--")
            try:
                op(channel)
            except grpc.RpcError as e:
                logging.error(e)
    except Exception as e:
        logging.error(f"Exception: {e}")
    finally:
        channel and channel.close()
