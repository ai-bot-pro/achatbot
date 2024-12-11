import base64
import json
import os
import logging
from typing import List


try:
    from openai import AsyncStream
    import litellm
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionFunctionMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionToolParam,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use OpenAI, you need to `pip install achatbot[litellm_processor]`. "
        "Also, set environment variable such as `OPENAI_API_KEY`."
        "see: https://docs.litellm.ai/docs/"
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


class LiteLLMProcessor(LLMProcessor):
    """
    llm proxy to use the OpenAI Input/Output Format
    see: https://docs.litellm.ai/docs/
    """

    TAG = "litellm_processor"

    def __init__(self, model: str = "github/gpt-4o", set_verbose: bool = False, **kwargs):
        super().__init__(model=model, **kwargs)
        super().__init__(**kwargs)
        self._model: str = model
        litellm.set_verbose = set_verbose
        # self._response_format = {"type": "json_object"}

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
        chunks = await litellm.acompletion(
            model=self._model,
            stream=True,
            # trim_messages ensures tokens(messages) < max_tokens(model)
            # need HF_TOKEN to be set
            # issue: https://github.com/BerriAI/litellm/issues/6505
            # messages=litellm.utils.trim_messages(messages, self._model),
            messages=messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
            stream_options={"include_usage": True},
            # stream Structured Outputs json mode !TODO @weedge -> partial support
            # https://docs.litellm.ai/docs/completion/json_mode
            # response_format={"type": "json_object"}  # 👈 KEY CHANGE
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
        if hasattr(chunk_dict, "usage") and chunk_dict["usage"]:
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
            # logging.info(f"chunk:{chunk.model_dump_json()}")
            await self.record_llm_usage_tokens(chunk_dict=chunk.model_dump())

            if len(chunk.choices) == 0:
                continue
            await self.stop_ttfb_metrics()

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
