from pathlib import Path
from typing import Union
import numpy as np
import onnxruntime
from aiortc import MediaStreamTrack
from aiortc.contrib.media import VideoFrame
from av.frame import Frame
from av.packet import Packet

from ..yolo import YOLOv10
from .base import BaseTrack


yolo_model = None


def load(cache_path: Path = Path("/cache")):
    global yolo_model
    if yolo_model is not None:
        print(f"{yolo_model} already loaded")
        return

    onnxruntime.preload_dlls()
    yolo_model = YOLOv10(cache_path)


class YOLOTrack(BaseTrack):
    """
    Custom media stream track performs object detection
    on the video stream and passes it back to the source peer
    """

    conf_threshold: float = 0.15

    def __init__(self, in_track: MediaStreamTrack = None) -> None:
        super().__init__(in_track)

        self.yolo_model = yolo_model

    # this is the essential method we need to implement
    # to create a custom MediaStreamTrack
    async def recv(self) -> Union[Frame, Packet]:
        assert self.in_track is not None
        frame = await self.in_track.recv()
        if self.in_track.kind == "audio":
            return frame

        img = frame.to_ndarray(format="bgr24")

        processed_img = self.detection(img)

        # VideoFrames are from a really nice package called av
        # which is a pythonic wrapper around ffmpeg
        # and a dependency of aiortc
        new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

    def detection(self, image: np.ndarray) -> np.ndarray:
        import cv2

        orig_shape = image.shape[:-1]

        image = cv2.resize(
            image,
            (self.yolo_model.input_width, self.yolo_model.input_height),
        )

        image = self.yolo_model.detect_objects(image, self.conf_threshold)

        image = cv2.resize(image, (orig_shape[1], orig_shape[0]))

        return image
