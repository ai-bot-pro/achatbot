import collections
import logging

import grpc


class _ClientCallDetails(
    collections.namedtuple("_ClientCallDetails", ("method", "timeout", "metadata", "credentials")),
    grpc.ClientCallDetails,
):
    pass


class AuthenticationInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    def __init__(self, interceptor_function):
        self._fn = interceptor_function

    def intercept_unary_unary(
        self, continuation, client_call_details: grpc.ClientCallDetails, request
    ):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,)), False, False
        )
        response = continuation(new_details, next(new_request_iterator))
        return postprocess(response) if postprocess else response

    def intercept_unary_stream(
        self, continuation, client_call_details: grpc.ClientCallDetails, request
    ):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,)), False, True
        )
        response_iterator = continuation(new_details, next(new_request_iterator))
        return postprocess(response_iterator) if postprocess else response_iterator

    def intercept_stream_unary(
        self, continuation, client_call_details: grpc.ClientCallDetails, request_iterator
    ):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator, True, False
        )
        response = continuation(new_details, new_request_iterator)
        return postprocess(response) if postprocess else response

    def intercept_stream_stream(
        self, continuation, client_call_details: grpc.ClientCallDetails, request_iterator
    ):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator, True, True
        )
        response = continuation(new_details, new_request_iterator)
        return postprocess(response) if postprocess else response


def add_authentication(header, value):
    def postprocess(response):
        logging.debug(f"response: {response}")
        return response

    def intercept_call(
        client_call_details, request_iterator, request_streaming, response_streaming
    ):
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        metadata.append(
            (
                "request_streaming",
                "s" if request_streaming else "u",
            )
        )
        metadata.append(
            (
                "response_streaming",
                "s" if response_streaming else "u",
            )
        )
        metadata.append(
            (
                header,
                value,
            )
        )
        logging.debug(f"client_call_details: {client_call_details}")
        client_call_details = _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials,
        )
        return client_call_details, request_iterator, postprocess

    return AuthenticationInterceptor(intercept_call)
