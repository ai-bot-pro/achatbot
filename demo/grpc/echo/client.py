import logging
import time

import grpc

from demo.grpc.idl.echo_pb2 import EchoRequest
from demo.grpc.idl.echo_pb2_grpc import EchoStub
from demo.grpc.interceptors.authentication_client import add_authentication


def echo_uu(channel):
    print("What is your name?")
    name = input()
    echo_stub = EchoStub(channel)
    request_data = EchoRequest(name=name)
    response = echo_stub.EchoUU(request_data)
    print(response.echo)


def echo_su(channel):
    echo_stub = EchoStub(channel)
    request_iterator = iter([EchoRequest(name="xiaohei"), EchoRequest(name="xiaowu")])
    response = echo_stub.EchoSU(request_iterator)
    print(response.echo)


def echo_us(channel):
    echo_stub = EchoStub(channel)
    request_data = EchoRequest(name="echo_us")
    response_iterator = echo_stub.EchoUS(request_data)
    for response in response_iterator:
        print(response.echo)


def echo_iter():
    cn = 1
    while True:
        yield EchoRequest(name=f"client_echo_iter{cn}")
        cn += 1
        time.sleep(1)


def echo_ss(channel):
    echo_stub = EchoStub(channel)
    response_iterator = echo_stub.EchoSS(echo_iter())
    for response in response_iterator:
        print(response.echo)


logging.basicConfig(level=logging.DEBUG)

# python -m demo.grpc.echo.client 2> err_cli.lo
if __name__ == "__main__":
    try:
        # todo: up to the rpc gateway to auth
        token = "oligei"
        authentication = add_authentication("authorization", token)
        channel = grpc.insecure_channel("localhost:50051")
        channel = grpc.intercept_channel(channel, authentication)
        for op in [echo_uu, echo_su, echo_us, echo_ss]:
            print(f"--{op}--")
            try:
                op(channel)
            except grpc.RpcError as e:
                logging.error(e)
    except Exception as e:
        logging.error(f"Exception: {e}")
    finally:
        channel and channel.close()
