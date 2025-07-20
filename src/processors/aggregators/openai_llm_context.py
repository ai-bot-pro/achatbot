#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List
import io
import json
import logging

from PIL import Image

from src.schemas.tools_schema import ToolsSchema

try:
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import (
        ChatCompletionToolParam,
        ChatCompletionToolChoiceOptionParam,
        ChatCompletionMessageParam,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use OpenAI types, you need to `pip install achatbot[openai]`. ")
    raise Exception(f"Missing module: {e}")
from apipeline.processors.frame_processor import FrameProcessor
from apipeline.frames.sys_frames import StartInterruptionFrame

from src.processors.aggregators.llm_response import LLMResponseAggregator
from src.types.frames.data_frames import (
    Frame,
    InterimTranscriptionFrame,
    TextFrame,
    FunctionCallResultFrame,
    TranscriptionFrame,
    VisionImageRawFrame,
)
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.types.frames.sys_frames import FunctionCallInProgressFrame


# JSON custom encoder to handle bytes arrays so that we can log contexts
# with images to the console.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return f"{obj.getbuffer()[0:8].hex()}..."
        return super().default(obj)


class OpenAILLMContext:
    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ):
        self._messages: List[ChatCompletionMessageParam] = messages if messages else []
        self._tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self._tools: List[ChatCompletionToolParam] | NotGiven = tools

    def __str__(self):
        return f"messages:{self._messages}, tools:{self._tools}, tool_choice:{self._tool_choice}"

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()

        for message in messages:
            if "name" not in message:
                message["name"] = message["role"]
            context.add_message(message)
        return context

    @staticmethod
    def from_image_frame(frame: VisionImageRawFrame) -> "OpenAILLMContext":
        """
        For images, we are deviating from the OpenAI messages shape. OpenAI
        expects images to be base64 encoded, but other vision models may not.
        So we'll store the image as bytes and do the base64 encoding as needed
        in the LLM service.
        """
        context = OpenAILLMContext()
        buffer = io.BytesIO()
        Image.frombytes(frame.mode, frame.size, frame.image).save(buffer, format=frame.format)
        context.add_message(
            {
                "content": frame.text,
                "role": "user",
                "data": buffer,
                "mime_type": f"image/{frame.format.lower()}",
            }
        )
        return context

    @property
    def messages(self) -> List[ChatCompletionMessageParam]:
        return self._messages

    @property
    def tools(self) -> List[ChatCompletionToolParam] | NotGiven | List[dict]:
        if isinstance(self._tools, ToolsSchema):
            functions_schema = self._tools.standard_tools
            self._tools = [
                {"function": func.to_default_dict(), "type": "function"}
                for func in functions_schema
            ]
        return self._tools

    @property
    def tool_choice(self) -> ChatCompletionToolChoiceOptionParam | NotGiven:
        return self._tool_choice

    def add_message(self, message: ChatCompletionMessageParam):
        self._messages.append(message)

    def add_messages(self, messages: List[ChatCompletionMessageParam]):
        self._messages.extend(messages)

    def set_messages(self, messages: List[ChatCompletionMessageParam]):
        self._messages[:] = messages

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self._messages

    def get_messages_json(self) -> str:
        return json.dumps(self._messages, cls=CustomEncoder)

    def set_tool_choice(self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven):
        self._tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN):
        if tools != NOT_GIVEN and isinstance(tools, list) and len(tools) == 0:
            tools = NOT_GIVEN
        self._tools = tools

    async def call_function(
        self,
        f: Callable[
            [str, str, Any, FrameProcessor, "OpenAILLMContext", Callable[[Any], Awaitable[None]]],
            Awaitable[None],
        ],
        *,
        function_name: str,
        tool_call_id: str,
        arguments: dict,
        llm: FrameProcessor,
    ) -> None:
        # Push a SystemFrame downstream. This frame will let our assistant context aggregator
        # know that we are in the middle of a function call. Some contexts/aggregators may
        # not need this. But some definitely do (Anthropic, for example).
        await llm.push_frame(
            FunctionCallInProgressFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
            )
        )

        # Define a callback function that pushes a FunctionCallResultFrame downstream.
        async def function_call_result_callback(result):
            await llm.push_frame(
                FunctionCallResultFrame(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    result=result,
                )
            )

        await f(function_name, tool_call_id, arguments, llm, self, function_call_result_callback)


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """

    context: OpenAILLMContext

    def __str__(self):
        return f"{self.name}(context: {self.context})"


class LLMContextAggregator(LLMResponseAggregator):
    def __init__(self, *, context: OpenAILLMContext, **kwargs):
        self._context = context
        super().__init__(**kwargs)

    @property
    def context(self):
        return self._context

    def get_context_frame(self) -> OpenAILLMContextFrame:
        return OpenAILLMContextFrame(context=self._context)

    async def push_context_frame(self):
        frame = self.get_context_frame()
        await self.push_frame(frame)

    def _add_messages(self, messages):
        self._context.add_messages(messages)

    def _set_messages(self, messages):
        self._context.set_messages(messages)

    def _set_tools(self, tools: List):
        self._context.set_tools(tools)

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self._reset()


class LLMUserContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            messages=[],
            context=context,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame,
        )


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            messages=[],
            context=context,
            role="assistant",
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
        )


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(context=context)


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: OpenAIUserContextAggregator):
        super().__init__(context=user_context_aggregator._context)
        self._user_context_aggregator = user_context_aggregator
        self._function_call_in_progress = None
        self._function_call_result = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_call_in_progress = None
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = frame
        elif isinstance(frame, FunctionCallResultFrame):
            if (
                self._function_call_in_progress
                and self._function_call_in_progress.tool_call_id == frame.tool_call_id
            ):
                self._function_call_in_progress = None
                self._function_call_result = frame
                await self._push_aggregation()
            else:
                logging.warning(
                    "FunctionCallResultFrame tool_call_id does not match FunctionCallInProgressFrame tool_call_id"
                )
                self._function_call_in_progress = None
                self._function_call_result = None

    async def _push_aggregation(self):
        if not (self._aggregation or self._function_call_result):
            return

        run_llm = False

        aggregation = self._aggregation
        self._aggregation = ""

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": frame.tool_call_id,
                                    "function": {
                                        "name": frame.function_name,
                                        "arguments": json.dumps(frame.arguments),
                                    },
                                    "type": "function",
                                }
                            ],
                        }
                    )
                    self._context.add_message(
                        {
                            "role": "tool",
                            "content": json.dumps(frame.result),
                            "tool_call_id": frame.tool_call_id,
                        }
                    )
                    run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
