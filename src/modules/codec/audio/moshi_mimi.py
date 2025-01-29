import logging
import os


try:
    import torch
    from moshi.models import loaders, MimiModel
except ModuleNotFoundError as e:
    logging.error(
        "In order to use moshi-mimi-codec, you need to `pip install achatbot[codec_moshi_mimi]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.utils.helper import get_device, print_model_params
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


class MoshiMimiCodec(EngineClass, ICodec):
    """
    PS: u can use moshi lib to get_mimi model from audio tokenizer ckpt, the same as from transformers[torch] ckpt.
    """

    TAG = "codec_moshi_mimi"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.load_model()

    def load_model(self):
        ckpt_file = os.path.join(self.args.model_dir, loaders.MIMI_NAME)
        self.model = loaders.get_mimi(ckpt_file, self.args.device)
        print_model_params(self.model, "moshi-mimi")

    def encode_code(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        vq_codes = self.model.encode(
            waveform_tensor[None, None, :].to(self.args.device)
        )  # waveform_tensor: [1,1,T]
        logging.debug(f"encode waveform to vq_codes: {vq_codes.shape}")
        return vq_codes

    def decode_code(self, vq_codes: torch.Tensor) -> torch.Tensor:
        waveform_tensor = self.model.decode(vq_codes.to(self.args.device))
        logging.debug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
        return waveform_tensor[0][0]
