import time
import queue
import threading

import grpc

from demo.grpc.idl.echo_pb2_grpc import EchoServicer
from demo.grpc.idl.echo_pb2 import EchoRequest, EchoResponse


class Echo(EchoServicer):
    def EchoUU(self, request: EchoRequest, context: grpc.ServicerContext) -> EchoResponse:
        return EchoResponse(echo="Hello %s!" % request.name)

    def EchoSU(self, request_iterator, context: grpc.ServicerContext):
        names = []
        for request in request_iterator:
            names.append(request.name)
        return EchoResponse(echo="Hello %s!" % ", ".join(names))

    def EchoUS(self, request, context: grpc.ServicerContext):
        yield EchoResponse(echo=f"stream response: {request.name}!")

    def EchoSS(self, request_iterator, context: grpc.ServicerContext):
        q = queue.Queue()

        def in_yield(request_iterator, q: queue.Queue):
            for request in request_iterator:
                q.put(request.name)

        in_thread = threading.Thread(
            target=in_yield,
            args=(
                request_iterator,
                q,
            ),
        )
        in_thread.start()

        cn = 1
        while True:
            request_name = q.get()
            yield EchoResponse(echo=f"welcome {request_name},serve echo {cn}")
            time.sleep(1)
            cn += 1

        in_thread.join()
