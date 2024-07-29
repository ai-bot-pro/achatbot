import logging

from apipeline.processors.frame_processor import FrameProcessor, FrameDirection

from src.processors.rtvi_processor import RTVIJSONCompletion
from src.types.frames.control_frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
from src.types.frames.data_frames import Frame, TextFrame, TransportMessageFrame


class FunctionCallProcessor(FrameProcessor):

    def __init__(self, context):
        super().__init__()
        self._checking = False
        self._aggregating = False
        self._emitted_start = False
        self._aggregation = ""
        self._context = context

        self._callbacks = {}
        self._start_callbacks = {}

    def register_function(self, function_name: str, callback, start_callback=None):
        self._callbacks[function_name] = callback
        if start_callback:
            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: str):
        del self._callbacks[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        return function_name in self._callbacks.keys()

    async def call_function(self, function_name: str, args):
        if function_name in self._callbacks.keys():
            return await self._callbacks[function_name](self, args)
        return None

    async def call_start_function(self, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](self)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._checking = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and self._checking:
            # TODO-CB: should we expand this to any non-text character to start the completion?
            if frame.text.strip().startswith("{") or frame.text.strip().startswith("```"):
                self._emitted_start = False
                self._checking = False
                self._aggregation = frame.text
                self._aggregating = True
            else:
                self._checking = False
                self._aggregating = False
                self._aggregation = ""
                self._emitted_start = False
                await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and self._aggregating:
            self._aggregation += frame.text
            # TODO-CB: We can probably ignore function start I think
            # if not self._emitted_start:
            #     fn = re.search(r'{"function_name":\s*"(.*)",', self._aggregation)
            #     if fn and fn.group(1):
            #         await self.call_start_function(fn.group(1))
            #         self._emitted_start = True
        elif isinstance(frame, LLMFullResponseEndFrame) and self._aggregating:
            try:
                self._aggregation = self._aggregation.replace("```json", "").replace("```", "")
                self._context.add_message({"role": "assistant", "content": self._aggregation})
                message = RTVIJSONCompletion(data=self._aggregation)
                msg = message.model_dump(exclude_none=True)
                await self.push_frame(TransportMessageFrame(message=msg))

            except Exception as e:
                print(f"Error parsing function call json: {e}")
                print(f"aggregation was: {self._aggregation}")

            self._aggregating = False
            self._aggregation = ""
            self._emitted_start = False
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
