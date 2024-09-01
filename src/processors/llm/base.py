import logging

from apipeline.processors.frame_processor import FrameDirection

from src.processors.aggregators.openai_llm_context import OpenAILLMContext
from src.processors.ai_processor import AIProcessor
from src.types.frames.control_frames import UserImageRequestFrame


class LLMProcessor(AIProcessor):
    """This class is a no-op but serves as a base class for LLM processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {}
        self._start_callbacks = {}

    # !TODO: use callback function type @weedge
    def register_function(self, function_name: str | None, callback, start_callback=None):

        # Registering a function with the function_name set to None will run that callback
        # for all functions
        self._callbacks[function_name] = callback
        # QUESTION FOR CB: maybe this isn't needed anymore?
        if start_callback:
            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: str | None):
        del self._callbacks[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        if None in self._callbacks.keys():
            return True
        return function_name in self._callbacks.keys()

    async def call_function(
            self,
            *,
            context: OpenAILLMContext,
            tool_call_id: str,
            function_name: str,
            arguments: str) -> None:
        f = None
        if function_name in self._callbacks.keys():
            f = self._callbacks[function_name]
        elif None in self._callbacks.keys():
            f = self._callbacks[None]
        else:
            return None
        await context.call_function(
            f,
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            llm=self)

    # QUESTION FOR CB: maybe this isn't needed anymore?
    async def call_start_function(self, context: OpenAILLMContext, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](function_name, self, context)
        elif None in self._start_callbacks.keys():
            return await self._start_callbacks[None](function_name, self, context)

    async def request_image_frame(self, user_id: str, *, text_content: str | None = None):
        await self.push_frame(
            UserImageRequestFrame(user_id=user_id, context=text_content),
            FrameDirection.UPSTREAM)
