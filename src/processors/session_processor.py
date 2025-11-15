import threading

import uuid
from apipeline.processors.async_frame_processor import AsyncFrameProcessor

from src.common.session import Session
from src.common.types import SessionCtx


class SessionProcessor(AsyncFrameProcessor):
    def __init__(
        self,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session or Session(**SessionCtx(str(uuid.uuid4())).__dict__)

    def set_ctx_state(self, **kwargs):
        self.session.ctx.state.update(kwargs)

    def get_ctx_state(self, key):
        return self.session.ctx.state.get(key)

    def set_user_id(self, user_id: str):
        pass

    async def create_conversation(self):
        pass