from typing import Union
from aiortc import MediaStreamTrack
from av.frame import Frame
from av.packet import Packet


class BaseTrack(MediaStreamTrack):
    """
    Custom media stream Base track
    - from remote peer track as in track for bot peer to process
    - set video/audio kind to process frame
    - default recv is echo back
    """

    def __init__(self, in_track: MediaStreamTrack = None) -> None:
        assert in_track is not None
        super().__init__()

        self.in_track = in_track
        self.kind = in_track.kind

    def set_kind(self, kind: str) -> None:
        self.kind = kind

    def set_in_track(self, in_track: MediaStreamTrack) -> None:
        self.in_track = in_track

    async def recv(self) -> Union[Frame, Packet]:
        """
        echo
        Receive the next :class:`~av.audio.frame.AudioFrame`,
        :class:`~av.video.frame.VideoFrame` or :class:`~av.packet.Packet`
        """
        assert self.in_track is not None
        frame = await self.in_track.recv()
        # print(frame)
        return frame
