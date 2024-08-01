import logging
import inspect


class EventHandlerManager:

    def __init__(self):
        self._event_handlers: dict = {}

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler
        return decorator

    def _register_event_handler(self, event_name: str):
        if event_name in self._event_handlers:
            raise Exception(f"Event handler {event_name} already registered")
        self._event_handlers[event_name] = []

    def add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(handler)

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in event handler {event_name}: {e}")
            raise e
