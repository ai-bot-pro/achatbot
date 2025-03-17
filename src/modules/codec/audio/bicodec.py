import os
import sys
import logging
import math
from typing import List

import torch
import torchaudio


try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../SparkTTS"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/SparkTTS"))

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from deps.SparkTTS.sparktts.models import bicodec
    from deps.SparkTTS.sparktts.utils.file import load_config

except ModuleNotFoundError as e:
    logging.error(
        "In order to use transformers bicodec, you need to `pip install achatbot[codec_bitokenizer]`.\nPlease install the missing modules: transformers, torchaudio, etc."
    )
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[codec_bitokenizer]` to install the dependencies."
    )

from src.common.utils.helper import get_device, print_model_params
from src.common.factory import EngineClass
from src.modules.codec.interface import ICodec
from src.types.codec import CodecArgs


class BiCodecTokenizer(EngineClass, ICodec):
    """
    BiCodec model  + Wav2Vec2 feature extractor(sematic tokens)
    """

    TAG = "codec_bitokenizer"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.args = CodecArgs(**kwargs)
        self.args.device = self.args.device or get_device()
        logging.info("CodecArgs: %s", self.args)
        self.config = load_config(f"{self.args.model_dir}/config.yaml")
        self.load_model()

    def load_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = bicodec.BiCodec.load_from_checkpoint(f"{self.args.model_dir}/BiCodec").to(
            self.args.device
        )
        print_model_params(self.model, self.TAG)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.args.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            f"{self.args.model_dir}/wav2vec2-large-xlsr-53"
        ).to(self.args.device)
        print_model_params(self.feature_extractor, f"{self.TAG}_feature_extractor")
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: torch.Tensor) -> torch.Tensor:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = torch.tile(wav, [ref_segment_length // wav_length + 1])

        return wav[:ref_segment_length]

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        inputs = self.processor(
            wavs,
            sampling_rate=self.config["sample_rate"],
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device))
        feats_mix = (feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]) / 3

        return feats_mix

    def encode_code(self, waveform_tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        ref_waveform_tensor = self.get_ref_clip(waveform_tensor)
        feat = self.extract_wav2vec2_features(waveform_tensor)
        batch = {
            "wav": waveform_tensor.unsqueeze(0).float().to(self.args.device),
            "ref_wav": ref_waveform_tensor.unsqueeze(0).float().to(self.args.device),
            "feat": feat.to(self.args.device),
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return [semantic_tokens, global_tokens]

    def decode_code(self, tokens: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(tokens, List) and len(tokens) == 2
        semantic_tokens, global_tokens = tokens[0], tokens[1]
        print(semantic_tokens.shape, global_tokens.shape)
        waveform_tensor = self.model.detokenize(semantic_tokens, global_tokens)
        logging.debug(f"decode vq_codes to gen waveform: {waveform_tensor.shape}")
        return waveform_tensor[0][0]
