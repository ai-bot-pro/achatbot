import os
import sys
import logging
import math
from typing import List

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../WavTokenizer"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/WavTokenizer"))

    import torch

    from deps.WavTokenizer.decoder.pretrained import WavTokenizer

except ModuleNotFoundError as e:
    logging.error(
        "In order to use transformers bicodec, you need to `pip install achatbot[codec_wavtokenizer]`.\n"
    )
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[codec_wavtokenizer]` to install the dependencies."
    )

from src.common.utils.helper import get_device, print_model_params
from src.common.factory import EngineClass
from src.modules.codec.interface import ICodec
from src.types.codec import CodecArgs


class WavTokenizerCodec(EngineClass, ICodec):
    """
    SOTA Discrete Codec Models With Forty Tokens Per Second for Audio Language Modeling
    """

    TAG = "codec_wavtokenizer"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.load_model()

    def load_model(self):
        """Load and initialize the codec and feature extractor."""
        self.model = WavTokenizer.from_pretrained0802(
            self.args.config_path, self.args.model_path
        ).to(self.args.device)
        print_model_params(self.model, self.TAG)

    def encode_code(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        bandwidth_id = torch.tensor([0])
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # [B,T]
        _, discrete_code = self.model.encode_infer(  # batch process
            waveform_tensor.to(self.args.device), bandwidth_id=bandwidth_id
        )
        logging.debug(f"encode waveform to discret vq code: {discrete_code.shape}")

        return discrete_code

    def decode_code(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        features = self.model.codes_to_features(audio_tokens)
        bandwidth_id = torch.tensor([0])
        waveform_tensor = self.model.decode(features, bandwidth_id=bandwidth_id)
        logging.debug(f"decode audio vq_codes to gen waveform: {waveform_tensor.shape}")

        return waveform_tensor[0]
