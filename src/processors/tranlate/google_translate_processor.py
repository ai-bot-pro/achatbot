import asyncio
import logging

from apipeline.frames.sys_frames import Frame
from apipeline.frames.data_frames import TextFrame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor, FrameDirection
from deep_translator import GoogleTranslator


class GoogleTranslateProcessor(FrameProcessor):
    def __init__(self, src: str = "en", target: str = "zh-CN", is_keep_original=False):
        super().__init__()
        self._src = src
        self._target = target
        self._is_keep_original = is_keep_original
        self.translator = GoogleTranslator(source=src, target=target)
        self._translate_frame = TextFrame

    def set_translate_frame(self, frame: TextFrame):
        self._translate_frame = frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self._translate_frame):
            if self._src == self._target:
                await self.push_frame(frame, direction)
                return
            try_cn = 3
            while try_cn > 0:
                try:
                    translated_text = self.translator.translate(frame.text)
                    break
                except Exception as e:
                    logging.error("An error occurred:", e)
                    await asyncio.sleep(1)
                    try_cn -= 1
            if self._is_keep_original:
                translated_text = f"{frame.text}\n{translated_text}"
            await self.push_frame(self._translate_frame(text=translated_text))
        else:
            await self.push_frame(frame, direction)
