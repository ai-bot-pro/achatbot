#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List
import logging
import json

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import StartInterruptionFrame

from src.processors.aggregators.openai_llm_context import OpenAILLMContextFrame, OpenAILLMContext
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.types.frames.data_frames import (
    Frame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    TranscriptionFrame,
    TextFrame,
)
from src.types.frames.sys_frames import FunctionCallInProgressFrame


class LLMResponseAggregator(FrameProcessor):

    def __init__(
        self,
        *,
        messages: List[dict],
        role: str,
        start_frame,
        end_frame,
        accumulator_frame: TextFrame,
        interim_accumulator_frame: TextFrame | None = None,
        handle_interruptions: bool = False
    ):
        super().__init__()

        self._messages = messages
        self._role = role
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._interim_accumulator_frame = interim_accumulator_frame
        self._handle_interruptions = handle_interruptions

        # Reset our accumulator state.
        self._reset()

    @property
    def messages(self):
        return self._messages

    @property
    def role(self):
        return self._role

    #
    # Frame processor
    #

    # Use cases implemented:
    #
    # S: Start, E: End, T: Transcription(ASR gen) / Text(LLM gen), I: Interim, X: Text (aggregated)
    #
    #        S E -> None
    #      S T E -> X
    #    S I T E -> X
    #    S I E T -> X
    #  S I E I T -> X
    #      S E T -> X
    #    S E I T -> X
    #
    # The following case would not be supported:
    #
    #    S I E T1 I T2 -> X
    #
    # and T2 would be dropped.

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        send_aggregation = False
        if isinstance(frame, self._start_frame):
            self._aggregation = ""
            self._aggregating = True
            self._seen_start_frame = True
            self._seen_end_frame = False
            self._seen_interim_results = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, self._end_frame):
            self._seen_end_frame = True
            self._seen_start_frame = False

            # We might have received the end frame but we might still be
            # aggregating (i.e. we have seen interim results but not the final
            # text).
            self._aggregating = self._seen_interim_results or len(self._aggregation) == 0

            # Send the aggregation if we are not aggregating anymore (i.e. no
            # more interim results received).
            send_aggregation = not self._aggregating
            await self.push_frame(frame, direction)
        elif isinstance(frame, self._accumulator_frame):
            if self._aggregating:
                self._aggregation += f" {frame.text}"
                # We have recevied a complete sentence, so if we have seen the
                # end frame and we were still aggregating, it means we should
                # send the aggregation.
                send_aggregation = self._seen_end_frame

            # We just got our final result, so let's reset interim results.
            self._seen_interim_results = False
        elif self._interim_accumulator_frame and isinstance(frame, self._interim_accumulator_frame):
            self._seen_interim_results = True
        elif self._handle_interruptions and isinstance(frame, StartInterruptionFrame):
            await self._push_aggregation()
            # Reset anyways
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            self._add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self._set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self._set_tools(frame.tools)
        else:
            await self.push_frame(frame, direction)

        if send_aggregation:
            await self._push_aggregation()

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            self._messages.append({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            frame = LLMMessagesFrame(self._messages)
            await self.push_frame(frame)

    def _add_messages(self, messages):
        self._messages.extend(messages)

    def _set_messages(self, messages):
        self._reset()
        self._messages.clear()
        self._messages.extend(messages)

    def _set_tools(self, tools):
        pass

    def _reset(self):
        self._aggregation = ""
        self._aggregating = False
        self._seen_start_frame = False
        self._seen_end_frame = False
        self._seen_interim_results = False


class LLMAssistantResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = []):
        super().__init__(
            messages=messages,
            role="assistant",
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True
        )


class LLMUserResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = []):
        super().__init__(
            messages=messages,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame
        )


class LLMFullResponseAggregator(FrameProcessor):
    """This class aggregates Text frames until it receives a
    LLMFullResponseEndFrame, then emits the concatenated text as
    a single text frame.

    given the following frames:

        TextFrame("Hello,")
        TextFrame(" world.")
        TextFrame(" I am")
        TextFrame(" an LLM.")
        LLMFullResponseEndFrame()]

    this processor will yield nothing for the first 4 frames, then

        TextFrame("Hello, world. I am an LLM.")
        LLMFullResponseEndFrame()

    when passed the last frame.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)
    ...         else:
    ...             print(frame.__class__.__name__)

    >>> aggregator = LLMFullResponseAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello,")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" world.")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" I am")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" an LLM.")))
    >>> asyncio.run(print_frames(aggregator, LLMFullResponseEndFrame()))
    Hello, world. I am an LLM.
    LLMFullResponseEndFrame
    """

    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
            self._aggregation = ""
        else:
            await self.push_frame(frame, direction)


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

    # TODO-CB: Types
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


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            messages=[],
            context=context,
            role="assistant",
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True
        )


class LLMUserContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            messages=[],
            context=context,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame
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
            if self._function_call_in_progress and self._function_call_in_progress.tool_call_id == frame.tool_call_id:
                self._function_call_in_progress = None
                self._function_call_result = frame
                await self._push_aggregation()
            else:
                logging.warning(
                    f"FunctionCallResultFrame tool_call_id does not match FunctionCallInProgressFrame tool_call_id")
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
                    self._context.add_message({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": frame.tool_call_id,
                                "function": {
                                    "name": frame.function_name,
                                    "arguments": json.dumps(frame.arguments)
                                },
                                "type": "function"
                            }
                        ]
                    })
                    self._context.add_message({
                        "role": "tool",
                        "content": json.dumps(frame.result),
                        "tool_call_id": frame.tool_call_id
                    })
                    run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
