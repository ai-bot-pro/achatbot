import grpc

from demo.grpc.idl.echo_pb2_grpc import EchoServicer
from demo.grpc.idl.echo_pb2 import EchoRequest, EchoResponse


class Echo(EchoServicer):
    def EchoUU(self, request: EchoRequest, context: grpc.ServicerContext) -> EchoResponse:
        return EchoResponse(echo='Hello %s!' % request.name)

    def EchoSU(self, request_iterator, context: grpc.ServicerContext):
        names = []
        for request in request_iterator:
            names.append(request.name)
        return EchoResponse(echo='Hello %s!' % ', '.join(names))

    def EchoUS(self, request, context: grpc.ServicerContext):
        yield EchoResponse(echo=f"stream response: {request.name}!")

    def EchoSS(self, request_iterator, context: grpc.ServicerContext):
        for request in request_iterator:
            yield EchoResponse(echo=f"Welcome, {request.name}!")
