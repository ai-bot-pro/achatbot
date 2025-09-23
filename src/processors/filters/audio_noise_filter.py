from apipeline.frames import Frame, AudioRawFrame
from apipeline.processors.frame_processor import FrameDirection

from src.common.interface import ISpeechEnhancer
from src.processors.session_processor import SessionProcessor
from src.types.frames import VADStateAudioRawFrame
from src.common.session import Session


class AudioNoiseFilter(SessionProcessor):
    def __init__(
        self,
        se: ISpeechEnhancer,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)
        self._se = se

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self.session.ctx.state["audio_chunk"] = frame.audio
            self.session.ctx.state["sample_rate"] = frame.sample_rate
            self.session.ctx.state["is_last"] = False
            if isinstance(frame, VADStateAudioRawFrame):
                self.session.ctx.state["is_last"] = frame.is_final
            filter_audio = self._se.enhance(self.session)  # only read, don't CoW
            # print(f"{len(frame.audio)=} {len(filter_audio)=}")
            frame.audio = filter_audio
        await self.push_frame(frame)
