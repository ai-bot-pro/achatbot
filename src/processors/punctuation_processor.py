import logging

import uuid
from apipeline.frames import Frame, TextFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.processors.session_processor import SessionProcessor
from src.common import interface
from src.common.session import Session


class PunctuationProcessor(SessionProcessor):
    def __init__(
        self,
        engine: interface.IPunc | None = None,
        session: Session | None = None,
        punc_cache: dict | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)
        assert engine is not None
        self.engine = engine
        self.punc_cache = punc_cache or {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            # logging.info(f"{self.punc_cache=}")
            self.set_ctx_state(text=frame.text, punc_cache=self.punc_cache)
            frame.text = self.engine.generate(self.session)
        await self.push_frame(frame, direction)
