import logging

try:
    import torch
    from xcodec2.modeling_xcodec2 import XCodec2Model
except ModuleNotFoundError as e:
    logging.error(
        "In order to use xcodec2-codec, you need to `pip install achatbot[codec_xcodec2]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.utils.helper import print_model_params
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


class XCodec2Codec(EngineClass, ICodec):
    TAG = "codec_xcodec2"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.load_model()

    def load_model(self):
        self.model = XCodec2Model.from_pretrained(self.args.model_dir)
        self.model.eval().cuda()
        print_model_params(self.model, "xcodec2")

    @torch.no_grad
    def encode_code(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # Shape: (C, T) C=1
        vq_codes = self.model.encode_code(input_waveform=waveform_tensor)
        logging.deug(f"encode waveform to vq_codes: {vq_codes}")
        return vq_codes

    @torch.no_grad
    def decode_code(self, vq_codes: torch.Tensor) -> torch.Tensor:
        waveform_tensor = self.model.decode_code(vq_codes)
        logging.deug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
        return waveform_tensor
