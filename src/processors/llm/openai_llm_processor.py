import base64
import json
import os
import logging
from typing import List


try:
    from openai import AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionFunctionMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use OpenAI, you need to `pip install achatbot[openai_llm_processor]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
import httpx
from apipeline.frames.data_frames import TextFrame, Frame
from apipeline.pipeline.pipeline import FrameDirection

from src.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from src.processors.llm.base import LLMProcessor, UnhandledFunctionException
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMModelUpdateFrame,
)
from src.types.frames.data_frames import LLMMessagesFrame, VisionImageRawFrame


class BaseOpenAILLMProcessor(LLMProcessor):
    """This is the base for all processors that use the AsyncOpenAI client.

    This processor consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    def __init__(self, *, model: str, api_key="", base_url="", **kwargs):
        super().__init__(**kwargs)
        # api_key = os.environ.get("OPENAI_API_KEY", api_key)
        self._model: str = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                    keepalive_expiry=None,
                )
            ),
        )

    async def call_function(
        self, *, context: OpenAILLMContext, tool_call_id: str, function_name: str, arguments: dict
    ) -> None:
        f = None
        if function_name in self._callbacks.keys():
            f = self._callbacks[function_name]
        elif None in self._callbacks.keys():
            f = self._callbacks[None]
        else:
            return None
        await context.call_function(
            f, function_name=function_name, tool_call_id=tool_call_id, arguments=arguments, llm=self
        )

    # QUESTION FOR CB: maybe this isn't needed anymore?
    async def call_start_function(self, context: OpenAILLMContext, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](function_name, self, context)
        elif None in self._start_callbacks.keys():
            return await self._start_callbacks[None](function_name, self, context)

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        chunks = await self._client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
            # stream Structured Outputs json mode !TODO @weedge -> partial support
            # response_format={ "type": "json-object" }
        )
        return chunks

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logging.info(f"Generating chat context messages: {context.get_messages_json()}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)

        return chunks

    async def record_llm_usage_tokens(self, chunk_dict: dict):
        tokens = {
            "processor": self.name,
            "model": self._model,
        }
        if chunk_dict["usage"]:
            tokens["prompt_tokens"] = chunk_dict["usage"]["prompt_tokens"]
            tokens["completion_tokens"] = chunk_dict["usage"]["completion_tokens"]
            tokens["total_tokens"] = chunk_dict["usage"]["total_tokens"]
            await self.start_llm_usage_metrics(tokens)

    async def _process_context(self, context: OpenAILLMContext):
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        async for chunk in chunk_stream:
            logging.debug(f"chunk:{chunk.model_dump_json()}")
            await self.record_llm_usage_tokens(chunk_dict=chunk.model_dump())

            if len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.tool_calls:
                # We're streaming the LLM response to enable the fastest response times.
                # For text, we just yield each chunk as we receive it and count on consumers
                # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                #
                # If the LLM is a function call, we'll do some coalescing here.
                # If the response contains a function name, we'll yield a frame to tell consumers
                # that they can start preparing to call the function with that name.
                # We accumulate all the arguments for the rest of the streamed response, then when
                # the response is done, we package up all the arguments and the function name and
                # yield a frame containing the function name and the arguments.

                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                    await self.call_start_function(context, function_name)
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                # TODO @weedge:
                # if use json response shot system prompt
                # need support llm assistant response json to exract tool name and arguments
                await self.push_frame(TextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            if self.has_function(function_name):
                await self._handle_function_call(context, tool_call_id, function_name, arguments)
            else:
                raise UnhandledFunctionException(
                    f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                )

    async def _handle_function_call(
        self, context: OpenAILLMContext, tool_call_id: str, function_name: str, arguments: str
    ):
        arguments = json.loads(arguments)
        await self.call_function(
            context=context,
            tool_call_id=tool_call_id,
            function_name=function_name,
            arguments=arguments,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logging.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())


class OpenAILLMProcessor(BaseOpenAILLMProcessor):
    """
    use OpenAI's client lib
    """

    TAG = "openai_llm_processor"

    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model=model, **kwargs)


class OpenAIGroqLLMProcessor(BaseOpenAILLMProcessor):
    """
    Groq API to be mostly compatible with OpenAI's client lib
    detail see: https://console.groq.com/docs/openai
    """

    TAG = "openai_groq_llm_processor"

    def __init__(self, model: str = "llama-3.2-11b-text-preview", **kwargs):
        super().__init__(model=model, **kwargs)

    async def record_llm_usage_tokens(self, chunk_dict: dict):
        if "x_groq" in chunk_dict and "usage" in chunk_dict["x_groq"]:
            tokens = chunk_dict["x_groq"]["usage"]
            tokens["processor"] = self.name
            tokens["model"] = self._model
            await self.start_llm_usage_metrics(tokens)
