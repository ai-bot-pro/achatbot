# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import queue
import sys
import os
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import *


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    print(result.get_response())
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def repeat_int32():
    # This client sends a single request to the model with the
    # following tensor data. In compliance with the behavior
    # of repeat_int32 model, it will expect the 4 responses
    # with output: [4], [2], [0] and [1] respectively.
    model_name = "repeat_int32"
    in_value = [4, 2, 0, 1]
    delay_value = [1, 2, 3, 4]
    wait_value = 5

    inputs = []
    inputs.append(grpcclient.InferInput("IN", [len(in_value)], "INT32"))
    inputs.append(grpcclient.InferInput("DELAY", [len(delay_value)], "UINT32"))
    inputs.append(grpcclient.InferInput("WAIT", [1], "UINT32"))

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("OUT"))
    outputs.append(grpcclient.InferRequestedOutput("IDX"))

    user_data = UserData()

    with grpcclient.InferenceServerClient(
        url=os.getenv("GRPC_SERVE_URL", "r15.modal.host:33231"), verbose=False
    ) as triton_client:
        # Establish stream
        triton_client.start_stream(callback=partial(callback, user_data))

        in_data = np.array(in_value, dtype=np.int32)
        inputs[0].set_data_from_numpy(in_data)
        delay_data = np.array(delay_value, dtype=np.uint32)
        inputs[1].set_data_from_numpy(delay_data)
        wait_data = np.array([wait_value], dtype=np.uint32)
        inputs[2].set_data_from_numpy(wait_data)

        request_id = "0"
        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
            enable_empty_final_response=True,
        )

        # Retrieve results...
        recv_count = 0
        result_dict = {}
        while True:
            data_item = user_data._completed_requests.get()
            if isinstance(data_item, InferenceServerException):
                raise data_item

            if data_item.get_response().parameters["triton_final_response"].bool_param is True:
                break

            this_id = data_item.get_response().id
            if this_id not in result_dict.keys():
                result_dict[this_id] = []
            result_dict[this_id].append((recv_count, data_item))
            recv_count += 1

        # Validate results...
        if len(result_dict[request_id]) != len(in_value):
            print(
                "expected {} many responses for request id {}, got {}".format(
                    len(in_value), request_id, len(result_dict[request_id])
                )
            )
            sys.exit(1)

        result_list = result_dict[request_id]
        for i in range(len(result_list)):
            expected_data = np.array([in_value[i]], dtype=np.int32)
            this_data = result_list[i][1].as_numpy("OUT")
            if not np.array_equal(expected_data, this_data):
                print("incorrect data: expected {}, got {}".format(expected_data, this_data))
                sys.exit(1)

        print("PASS: repeat_int32")
        sys.exit(0)


def square_int32():
    # This client sends a 4 requests to the model with the
    # input as: [4], [2], [0] and [1] respectively. In
    # compliance with the behavior of square_int32 model,
    # it will expect the 4 responses for the 1st request
    # each with output [4], 2 responses for 2nd request
    # each with output [2], no response for the 3rd request
    # and finally 1 response for the 4th request with output
    # [1]
    model_name = "square_int32"
    in_values = [4, 2, 0, 1]
    inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.int32))]
    outputs = [grpcclient.InferRequestedOutput("OUT")]

    user_data = UserData()

    with grpcclient.InferenceServerClient(
        url=os.getenv("GRPC_SERVE_URL", "r15.modal.host:33231"), verbose=True
    ) as triton_client:
        # Establish stream
        triton_client.start_stream(callback=partial(callback, user_data))

        # Send specified many requests in parallel
        for i in range(len(in_values)):
            in_data = np.array([in_values[i]], dtype=np.int32)
            inputs[0].set_data_from_numpy(in_data)

            triton_client.async_stream_infer(
                model_name=model_name,
                inputs=inputs,
                request_id=str(i),
                outputs=outputs,
                enable_empty_final_response=True,
            )

        # Retrieve results...
        recv_count = 0
        result_dict = {}
        while True:
            data_item = user_data._completed_requests.get()
            if isinstance(data_item, InferenceServerException):
                raise data_item

            if data_item.get_response().parameters["triton_final_response"].bool_param is True:
                break

            this_id = data_item.get_response().id
            if this_id not in result_dict.keys():
                result_dict[this_id] = []
            result_dict[this_id].append((recv_count, data_item))

            recv_count += 1

        # Validate results...
        for i in range(len(in_values)):
            this_id = str(i)
            if in_values[i] != 0 and this_id not in result_dict.keys():
                print("response for request id {} not received".format(this_id))
                sys.exit(1)
            elif in_values[i] == 0 and this_id in result_dict.keys():
                print("received unexpected response for request id {}".format(this_id))
                sys.exit(1)
            if in_values[i] != 0:
                if len(result_dict[this_id]) != in_values[i]:
                    print(
                        "expected {} many responses for request id {}, got {}".format(
                            in_values[i], this_id, result_dict[this_id]
                        )
                    )
                    sys.exit(1)

            if in_values[i] != 0:
                result_list = result_dict[this_id]
                expected_data = np.array([in_values[i]], dtype=np.int32)
                for j in range(len(result_list)):
                    this_data = result_list[j][1].as_numpy("OUT")
                    if not np.array_equal(expected_data, this_data):
                        print(
                            "incorrect data: expected {}, got {}".format(expected_data, this_data)
                        )
                        sys.exit(1)

        print("PASS: square_int32")
        sys.exit(0)


def bls_decoupled_sync():
    model_name = "bls_decoupled_sync"

    # with httpclient.InferenceServerClient(
    #    os.getenv("HTTP_SERVE_URL", "localhost:8000"), verbose=True
    with grpcclient.InferenceServerClient(
        url=os.getenv("GRPC_SERVE_URL", "r15.modal.host:33231"), verbose=True
    ) as client:
        in_values = [4, 2, 0, 1]

        for in_value in in_values:
            input_data = np.array([in_value], dtype=np.int32)
            inputs = [
                grpcclient.InferInput("IN", input_data.shape, np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput("SUM")]

            response = client.infer(model_name, inputs, request_id="1", outputs=outputs)

            response.get_response()
            output_data = response.as_numpy("SUM")
            print("==========model result==========")
            print("The square value of {} is {}\n".format(input_data, output_data))

            if not np.allclose(input_data * input_data, output_data):
                print(
                    "BLS Decoupled Sync example error: incorrect output value. Expected {}, got {}."
                ).format(input_data * input_data, output_data)
                sys.exit(1)

        print("PASS: BLS Decoupled Sync")
        sys.exit(0)


def bls_decoupled_async():
    model_name = "bls_decoupled_async"

    # with httpclient.InferenceServerClient(
    #    os.getenv("HTTP_SERVE_URL", "localhost:8000"), verbose=True
    with grpcclient.InferenceServerClient(
        url=os.getenv("GRPC_SERVE_URL", "r15.modal.host:33231"), verbose=True
    ) as client:
        in_values = [4, 2, 0, 1]

        for in_value in in_values:
            input_data = np.array([in_value], dtype=np.int32)
            inputs = [
                grpcclient.InferInput("IN", input_data.shape, np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput("SUM")]

            response = client.infer(model_name, inputs, request_id="1", outputs=outputs)

            response.get_response()
            # output_data contains two times of the square value of the input value.
            output_data = response.as_numpy("SUM")
            print("==========model result==========")
            print("Two times the square value of {} is {}\n".format(input_data, output_data))

            if not np.allclose((2 * input_data * input_data), output_data):
                print(
                    "BLS Decoupled Async example error: incorrect output value. Expected {}, got {}.".format(
                        (2 * input_data * input_data), output_data
                    )
                )
                sys.exit(1)

        print("PASS: BLS Decoupled Async")
        sys.exit(0)


funcs = {
    "repeat_int32": repeat_int32,
    "square_int32": square_int32,
    "bls_decoupled_async": bls_decoupled_async,
    "bls_decoupled_sync": bls_decoupled_sync,
}

"""
OP=repeat_int32 GRPC_SERVE_URL=r21.modal.host:40765 python src/llm/trtllm/decoupled/client_grpc.py
OP=square_int32 GRPC_SERVE_URL=r21.modal.host:40765 python src/llm/trtllm/decoupled/client_grpc.py
OP=bls_decoupled_sync GRPC_SERVE_URL=r21.modal.host:40765 python src/llm/trtllm/decoupled/client_grpc.py
OP=bls_decoupled_async GRPC_SERVE_URL=r21.modal.host:40765 python src/llm/trtllm/decoupled/client_grpc.py
"""

if __name__ == "__main__":
    op = os.getenv("OP", "repeat_int32")
    funcs[op]()
