import logging
import math

import torch
import torchaudio

try:
    from transformers import DacModel, AutoFeatureExtractor
except ModuleNotFoundError as e:
    logging.error(
        "In order to use transoformers mimi codec, you need to `pip install achatbot[codec_transformers_dac]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.factory import EngineClass
from src.common.utils.helper import get_device, print_model_params
from src.types.codec import CodecArgs
from src.modules.codec.interface import ICodec


class TransformersDescriptAudioCodec(EngineClass, ICodec):
    """
    use transoformers[torch] dac model and config
    https://huggingface.co/descript/dac_44khz/blob/main/config.json
    """

    TAG = "codec_transformers_dac"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.load_model()

    def load_model(self):
        self.model = DacModel.from_pretrained(self.args.model_dir)
        self.model.eval().requires_grad_(False).to(self.args.device)
        print_model_params(self.model, "dac")
        self.sampling_rate = self.model.config.sampling_rate
        logging.info(f"dac config: {self.model.config}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.args.model_dir)
        logging.info(f"feature_extractor: {self.feature_extractor}")

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, right_pad))

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
        with torch.autocast(
            self.model.device.type, torch.float16, enabled=self.model.device.type != "cpu"
        ):
            waveform_tensor = self.model.decode(
                quantized_representation=None, audio_codes=vq_codes
            ).audio_values
            logging.debug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
            return waveform_tensor[0].float()
