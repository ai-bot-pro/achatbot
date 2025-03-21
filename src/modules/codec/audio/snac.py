import logging
import os
from typing import List


try:
    import torch
    from snac import SNAC
except ImportError as e:
    logging.error("In order to use SNAC-codec, you need to `pip install achatbot[codec_snac]`.")
    raise ImportError(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.utils.helper import get_device, print_model_params
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


class SNACCodec(EngineClass, ICodec):
    """
    Multi-Scale Neural Audio Codec (SNAC) compresses audio into discrete codes at a low bitrate
    - https://arxiv.org/abs/2410.14411
    - https://github.com/hubertsiuzdak/snac
    - https://huggingface.co/hubertsiuzdak/snac_24khz (for 🗣️ Speech)
    - https://huggingface.co/hubertsiuzdak/snac_32khz (for 🎸 Music / Sound Effects)
    - https://huggingface.co/hubertsiuzdak/snac_44khz (for 🎸 Music / Sound Effects)
    """

    TAG = "codec_snac"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.load_model()

    def load_model(self):
        self.model = SNAC.from_pretrained(self.args.model_dir).to(self.args.device).eval()
        print_model_params(self.model, "snac")

    @torch.inference_mode()
    def encode_code(self, waveform_tensor: torch.Tensor) -> List[torch.Tensor]:
        vq_codes_list = self.model.encode(
            waveform_tensor[None, None, :].to(self.args.device)
        )  # waveform_tensor: [1,1,T]
        logging.debug(f"encode waveform to vq_codes cn: {len(vq_codes_list)}")
        return vq_codes_list

    @torch.inference_mode()
    def decode_code(self, vq_codes: List[torch.Tensor]) -> torch.Tensor:
        if not isinstance(vq_codes, list):
            vq_codes = vq_codes.to(self.args.device)
            vq_codes = [vq_codes.squeeze(0)]  # [[1,T]]
        waveform_tensor = self.model.decode(vq_codes)
        logging.debug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
        return waveform_tensor[0][0]
