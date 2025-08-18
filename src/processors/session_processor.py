import threading

import uuid
from apipeline.processors.frame_processor import FrameProcessor

from src.common.session import Session
from src.common.types import SessionCtx


class SessionProcessor(FrameProcessor):
    def __init__(
        self,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

    def set_ctx_state(self, **kwargs):
        self.session.ctx.state.update(kwargs)

    def get_ctx_state(self, key):
        return self.session.ctx.state.get(key)
