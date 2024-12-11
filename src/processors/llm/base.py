import logging

from apipeline.pipeline.pipeline import FrameDirection
from apipeline.processors.frame_processor import FrameProcessorMetrics, MetricsFrame

from src.processors.ai_processor import AIProcessor
from src.types.frames.control_frames import UserImageRequestFrame


class UnhandledFunctionException(Exception):
    pass


class LLMProcessorMetrics(FrameProcessorMetrics):
    async def start_llm_usage_metrics(self, tokens: dict):
        logging.debug(f"{self._name} tokens: {tokens}")
        return MetricsFrame(tokens=[tokens])


class LLMProcessor(AIProcessor):
    """This class is a no-op but serves as a base class for LLM processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {}
        self._start_callbacks = {}
        self._metrics = LLMProcessorMetrics(name=self.name)
        self._model = ""

    def set_model(self, model: str):
        self._model: str = model

    def can_generate_metrics(self) -> bool:
        return True

    def set_llm_args(self, **args):
        pass

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

    async def request_image_frame(self, user_id: str, *, text_content: str | None = None):
        await self.push_frame(
            UserImageRequestFrame(user_id=user_id, context=text_content), FrameDirection.UPSTREAM
        )

    async def start_llm_usage_metrics(self, tokens: dict):
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_llm_usage_metrics(tokens)
            if frame:
                await self.push_frame(frame)
