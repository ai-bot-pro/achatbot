from abc import abstractmethod

from src.processors.ai_processor import AIProcessor


class LLMProcessor(AIProcessor):
    """This class is a no-op but serves as a base class for LLM processors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {}
        self._start_callbacks = {}

    @abstractmethod
    async def set_model(self, model: str):
        pass

    @abstractmethod
    async def set_llm_args(self, **args):
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
