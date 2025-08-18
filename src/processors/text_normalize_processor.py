from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames import Frame, TextFrame

from src.processors.session_processor import SessionProcessor
from src.common.session import Session
from src.common import interface


class TextNormalizeProcessor(SessionProcessor):
    def __init__(
        self,
        engine: interface.ITextProcessing = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)
        assert engine is not None
        self.engine = engine

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.set_ctx_state(text=frame.text)
            frame.text = self.engine.normalize(self.session)
        await self.push_frame(frame, direction)
