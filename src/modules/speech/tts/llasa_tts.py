import logging
from pathlib import Path
from typing import AsyncGenerator
import os
import io

import numpy as np
import torch
import torchaudio

try:
    from src.modules.codec.audio.xcodec2 import XCodec2Codec
    from src.core.llm.transformers.manual_speech_llasa import TransformersManualSpeechLlasa
except ModuleNotFoundError as e:
    logging.error("In order to use llasa-tts, you need to `pip install achatbot[tts_llasa]`.")
    raise Exception(f"Missing module: {e}")

from src.common.utils.audio_utils import AUDIO_EXTENSIONS
from src.common.utils.helper import file_md5_hash
from src.common.types import PYAUDIO_PAFLOAT32, RATE
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.llasa import LlasaTTSArgs
from .base import BaseTTS


class LlasaTTS(BaseTTS, ITts):
    TAG = "tts_llasa"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**LlasaTTSArgs().__dict__, **kwargs}

    def __init__(self, **kwargs) -> None:
        self.args = LlasaTTSArgs(**kwargs)
        self.lm_model = TransformersManualSpeechLlasa(**self.args.lm_args.__dict__)
        self.codec_model = XCodec2Codec(**self.args.xcode2_args.__dict__)

        self.ref_encode_codebook_indices_dir = self.ref_codebook_indices_dir()
        self.voices = self.load_ref_voices()
        self.ref_encode_codebook_indices: torch.Tensor = None
        self.args.ref_audio_file_path and self.set_voice(self.args.ref_audio_file_path)

    def ref_codebook_indices_dir(self):
        """
        mkdir -p $ref_dir
        """
        os.makedirs(self.args.output_codebook_indices_dir, exist_ok=True)

        return self.args.output_codebook_indices_dir

    def set_voice(self, ref_audio_path: str):
        """
        - if ref_audio_path is audio path, encode audio to ref_encode_codebook_indices (vq mel spec)
        - if ref_audio_path is encode_codebook_indices path(.npy), load it
        """
        if os.path.exists(ref_audio_path) is False:
            raise FileNotFoundError(f"reference_audio_path: {ref_audio_path}")
        input_path = Path(ref_audio_path)

        if input_path.suffix == ".npy":
            self.ref_encode_codebook_indices = torch.from_numpy(np.load(input_path)).to(
                self.args.device
            )
            md5_hash = file_md5_hash(ref_audio_path)
            self.voices[md5_hash] = self.ref_encode_codebook_indices
            return
        if input_path.suffix not in AUDIO_EXTENSIONS:
            raise ValueError(f"Invalid audio file: {ref_audio_path}")

        logging.info(f"Processing in-place reconstruction of {input_path}")

        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != RATE:
            # only 16khz speech support!
            audio = torchaudio.functional.resample(audio, sr, RATE)

        audios = audio[None].to(self.args.device)
        logging.info(f"Loaded audio with {audios.shape[2] / RATE:.2f} seconds")

        indices = self.codec_model.encode_code(audios[0][0])
        logging.info(f"Generated indices of shape {indices.shape}")

        # Save indices (.npy store numpy array)
        if self.args.is_save is True:
            output_path = self.ref_encode_codebook_indices_dir / input_path.name
            output_path = output_path.with_suffix(".npy")
            np.save(output_path, indices.cpu().numpy())
            logging.info(f"Save indices numpy array to {output_path}")

        self.ref_encode_codebook_indices = indices
        md5_hash = file_md5_hash(ref_audio_path)
        self.voices[md5_hash] = self.ref_encode_codebook_indices

    def get_voices(self) -> list:
        return list(self.voices.keys())

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": RATE,
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        pass
