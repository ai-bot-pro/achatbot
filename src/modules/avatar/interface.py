from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from src.types.avatar import AudioSlice


class IFaceAvatar(ABC):
    """
    this is the interface for face avatar (now just for liteavatar method)
    - load model weights
    - audio2signal -> signal2img -> mouth2full -> audio+video
    """

    @abstractmethod
    def load(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def audio2signal(self, audio_slice: AudioSlice) -> list:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def signal2img(self, signal_data) -> tuple[torch.Tensor, int]:
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def mouth2full(
        self, mouth_image: torch.Tensor, bg_frame_id: int, use_bg: bool = False
    ) -> np.ndarray:
        raise NotImplementedError("must be implemented in the child class")
