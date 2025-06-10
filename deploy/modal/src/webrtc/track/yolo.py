from pathlib import Path
import numpy as np
import onnxruntime
from aiortc import MediaStreamTrack
from aiortc.contrib.media import VideoFrame

from ..yolo import YOLOv10
from .base import BaseTrack


yolo_model = None


def load(cache_path: Path = Path("/cache")):
    global yolo_model
    if yolo_model is not None:
        return

    onnxruntime.preload_dlls()
    yolo_model = YOLOv10(cache_path)


class YOLOTrack(BaseTrack):
    """
    Custom media stream track performs object detection
    on the video stream and passes it back to the source peer
    """

    kind: str = "video"
    conf_threshold: float = 0.15

    def __init__(self, track: MediaStreamTrack = None) -> None:
        super().__init__(track)

        self.yolo_model = yolo_model
        print(yolo_model)

    # this is the essential method we need to implement
    # to create a custom MediaStreamTrack
    async def recv(self) -> VideoFrame:
        assert self.track is not None
        frame = await self.track.recv()
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
