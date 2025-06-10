from typing import Union
from aiortc import MediaStreamTrack
from av.frame import Frame
from av.packet import Packet


class BaseTrack(MediaStreamTrack):
    """
    Custom media stream Base track
    """

    def __init__(self, track: MediaStreamTrack = None) -> None:
        super().__init__()

        self.track = track

    def set_in_track(self, track: MediaStreamTrack) -> None:
        self.track = track

    async def recv(self) -> Union[Frame, Packet]:
        """
        echo
        Receive the next :class:`~av.audio.frame.AudioFrame`,
        :class:`~av.video.frame.VideoFrame` or :class:`~av.packet.Packet`
        """
        assert self.track is not None
        frame = await self.track.recv()
        print(frame)
        return frame
