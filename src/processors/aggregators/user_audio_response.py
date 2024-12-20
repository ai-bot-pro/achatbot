import collections
import io
import logging
from typing import Type

from apipeline.processors.frame_processor import FrameDirection, FrameProcessor

from src.types.frames import *


class AudioResponseAggregator(FrameProcessor):
    def __init__(
        self,
        *,
        start_frame: Type[Frame],
        end_frame: Type[Frame],
        accumulator_frame: Type[AudioRawFrame],
        interim_accumulator_frame: Type[AudioRawFrame] | None = None,
    ):
        super().__init__()

        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._interim_accumulator_frame = interim_accumulator_frame

        # ring buffer save latest 10 frames
        self._audio_buffer = collections.deque(maxlen=10)

        # Reset our accumulator state.
        self._reset()

    def _reset(self):
        self._aggregation = bytearray()
        self._aggregating = False
        self._seen_start_frame = False
        self._seen_end_frame = False
        self._seen_interim_results = False
        self._cur_audio_frame = None
        self._audio_buffer.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        send_aggregation = False

        if isinstance(frame, self._start_frame):
            self._aggregating = True
            self._seen_start_frame = True
            self._seen_end_frame = False
            self._seen_interim_results = False
            for buff_frame in self._audio_buffer:
                self._aggregation.extend(buff_frame.audio)
            await self.push_frame(frame, direction)
        elif isinstance(frame, self._end_frame):
            self._seen_end_frame = True
            self._seen_start_frame = False

            # We might have received the end frame but we might still be
            # aggregating (i.e. we have seen interim results but not the final audio
            self._aggregating = self._seen_interim_results or len(self._aggregation) == 0

            # Send the aggregation if we are not aggregating anymore (i.e. no
            # more interim results received).
            send_aggregation = not self._aggregating
            await self.push_frame(frame, direction)
        elif isinstance(frame, self._accumulator_frame):
            if self._aggregating:
                self._cur_audio_frame = frame
                self._aggregation.extend(frame.audio)
                # We have recevied a complete sentence, so if we have seen the
                # end frame and we were still aggregating, it means we should
                # send the aggregation.
                send_aggregation = self._seen_end_frame
            else:
                # save frame to ringbuffer
                self._audio_buffer.append(frame)

            # We just got our final result, so let's reset interim results.
            self._seen_interim_results = False
        elif self._interim_accumulator_frame and isinstance(frame, self._interim_accumulator_frame):
            self._seen_interim_results = True
        else:
            await self.push_frame(frame, direction)

        if send_aggregation:
            await self._push_aggregation()

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            frame = self._cur_audio_frame
            frame.audio = bytes(self._aggregation)

            # Reset the aggregation. Reset it before pushing it down,
            # otherwise if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = bytearray()

            await self.push_frame(frame)

            # Reset our accumulator state.
            self._reset()


class UserAudioResponseAggregator(AudioResponseAggregator):
    """
    user input audio frame aggregator, no history, just aggr audio bytes
    """

    def __init__(self):
        super().__init__(
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=AudioRawFrame,
        )
