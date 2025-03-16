from abc import ABC, abstractmethod
from typing import List

import torch


class ICodec(ABC):
    @abstractmethod
    def load_model(self):
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def encode_code(self, wav_tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """
        Encode the given input tensor to quantized representation.

        Args:
            wav_tensor (torch.Tensor): Float tensor of shape [T]

        Returns:
            vq_codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        raise NotImplementedError("must be implemented in the child class")

    @abstractmethod
    def decode_code(self, vq_codes: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        """Decode the given codes to a reconstructed representation.

        Args:
            vq_codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            waveform_tensor (torch.Tensor): Float tensor of shape [T], the reconstructed audio.
        """
        raise NotImplementedError("must be implemented in the child class")
