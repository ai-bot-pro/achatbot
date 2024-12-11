import logging

import grpc


def _unary_unary_rpc_terminator(code, details):
    return grpc.unary_unary_rpc_method_handler(
        lambda ignored_request, context: context.abort(code, details + "(uu)")
    )


def _stream_unary_rpc_terminator(code, details):
    return grpc.stream_unary_rpc_method_handler(
        lambda ignored_request, context: context.abort(code, details + "(su)")
    )


def _unary_stream_rpc_terminator(code, details):
    return grpc.unary_stream_rpc_method_handler(
        lambda ignored_request, context: context.abort(code, details + "(us)")
    )


def _stream_stream_rpc_terminator(code, details):
    return grpc.stream_stream_rpc_method_handler(
        lambda ignored_request, context: context.abort(code, details + "(ss)")
    )


class AuthenticationInterceptor(grpc.ServerInterceptor):
    def __init__(self, header, value, code, details):
        self._header = header
        self._value = value
        self._terminators = {
            "uu": _unary_unary_rpc_terminator(code, details),
            "su": _stream_unary_rpc_terminator(code, details),
            "us": _unary_stream_rpc_terminator(code, details),
            "ss": _stream_stream_rpc_terminator(code, details),
            "": _unary_unary_rpc_terminator(code, details),
        }

    def intercept_service(self, continuation, handler_call_details):
        logging.debug(f"{self._header}, {self._value}," f"{handler_call_details}")
        if (self._header, self._value) in handler_call_details.invocation_metadata:
            return continuation(handler_call_details)  # next
        else:
            s_type = ""
            if ("request_streaming", "s") in handler_call_details.invocation_metadata:
                s_type += "s"
            if ("request_streaming", "u") in handler_call_details.invocation_metadata:
                s_type += "u"
            if ("response_streaming", "s") in handler_call_details.invocation_metadata:
                s_type += "s"
            if ("response_streaming", "u") in handler_call_details.invocation_metadata:
                s_type += "u"
            logging.debug(f"s_type:{s_type} terminator: {self._terminators[s_type]}")
            return self._terminators[s_type]  # abort
