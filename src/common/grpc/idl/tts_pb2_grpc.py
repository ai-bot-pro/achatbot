# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import grpc
import warnings

from . import tts_pb2 as tts__pb2

GRPC_GENERATED_VERSION = "1.65.0"
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = "1.65.0"
SCHEDULED_RELEASE_DATE = "June 25, 2024"
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + " but the generated code in tts_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
        + f" This warning will become an error in {EXPECTED_ERROR_RELEASE},"
        + f" scheduled for release on {SCHEDULED_RELEASE_DATE}.",
        RuntimeWarning,
    )


class TTSStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.LoadModel = channel.unary_unary(
            "/chat_bot.tts.TTS/LoadModel",
            request_serializer=tts__pb2.LoadModelRequest.SerializeToString,
            response_deserializer=tts__pb2.LoadModelResponse.FromString,
            _registered_method=True,
        )
        self.SynthesizeUS = channel.unary_stream(
            "/chat_bot.tts.TTS/SynthesizeUS",
            request_serializer=tts__pb2.SynthesizeRequest.SerializeToString,
            response_deserializer=tts__pb2.SynthesizeResponse.FromString,
            _registered_method=True,
        )


class TTSServicer(object):
    """Missing associated documentation comment in .proto file."""

    def LoadModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SynthesizeUS(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_TTSServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "LoadModel": grpc.unary_unary_rpc_method_handler(
            servicer.LoadModel,
            request_deserializer=tts__pb2.LoadModelRequest.FromString,
            response_serializer=tts__pb2.LoadModelResponse.SerializeToString,
        ),
        "SynthesizeUS": grpc.unary_stream_rpc_method_handler(
            servicer.SynthesizeUS,
            request_deserializer=tts__pb2.SynthesizeRequest.FromString,
            response_serializer=tts__pb2.SynthesizeResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("chat_bot.tts.TTS", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers("chat_bot.tts.TTS", rpc_method_handlers)


# This class is part of an EXPERIMENTAL API.


class TTS(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def LoadModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/chat_bot.tts.TTS/LoadModel",
            tts__pb2.LoadModelRequest.SerializeToString,
            tts__pb2.LoadModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def SynthesizeUS(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/chat_bot.tts.TTS/SynthesizeUS",
            tts__pb2.SynthesizeRequest.SerializeToString,
            tts__pb2.SynthesizeResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
