import logging

import torch

try:
    from transformers import MimiModel, AutoFeatureExtractor
except ModuleNotFoundError as e:
    logging.error(
        "In order to use transoformers mimi codec, you need to `pip install achatbot[codec_transformers_mimi]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.utils.helper import get_device, print_model_params
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


class TransformersMimiCodec(EngineClass, ICodec):
    """
    use transoformers[torch] mimi model and config
    https://huggingface.co/kyutai/mimi/blob/main/config.json
    """

    TAG = "codec_transformers_mimi"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.load_model()

    def load_model(self):
        self.model = MimiModel.from_pretrained(self.args.model_dir)
        self.model.eval().to(self.args.device)
        print_model_params(self.model, "mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.args.model_dir)
        logging.debug(f"feature_extractor: {self.feature_extractor}")

    @torch.no_grad
    def encode_code(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.encode(waveform_tensor[None, None, :].to(self.args.device)).audio_codes
        # pre-process the input waveform
        inputs = self.feature_extractor(
            raw_audio=waveform_tensor,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        )

        output = self.model.encode(inputs["input_values"].to(self.args.device))
        vq_codes = output.audio_codes
        logging.debug(f"encode waveform to vq_codes: {vq_codes.shape}")
        return vq_codes

    @torch.no_grad
    def decode_code(self, vq_codes: torch.Tensor) -> torch.Tensor:
        waveform_tensor = self.model.decode(vq_codes.to(self.args.device))[0]
        logging.debug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
        return waveform_tensor[0][0]
