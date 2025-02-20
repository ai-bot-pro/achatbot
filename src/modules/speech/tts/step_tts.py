import logging
from pathlib import Path
import sys
from typing import AsyncGenerator
import os
import io

import numpy as np
import torch
import torchaudio


try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../StepAudio"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/StepAudio"))
    from src.core.llm.transformers.manual_speech_step import TransformersManualSpeechStep
    from deps.StepAudio.tokenizer import StepAudioTokenizer
except ModuleNotFoundError as e:
    logging.error("In order to use step-tts, you need to `pip install achatbot[tts_step]`.")
    raise Exception(f"Missing module: {e}")

from deps.StepAudio.cosyvoice.cli.cosyvoice import CosyVoice
from src.common.utils.audio_utils import AUDIO_EXTENSIONS
from src.common.utils.helper import file_md5_hash
from src.common.types import PYAUDIO_PAFLOAT32, RATE
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.step import StepTTSArgs
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import BaseTTS


class StepTTS(BaseTTS, ITts):
    TAG = "tts_step"

    def __init__(self, **kwargs) -> None:
        self.args = StepTTSArgs(**kwargs)
        self.lm_args = TransformersSpeechLMArgs(**self.args.lm_args)
        self.lm_model = TransformersManualSpeechStep(**self.lm_args.__dict__)
        assert (
            self.args.stream_factor >= 2
        ), "stream_factor must >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)"

        self.common_cosy_model = CosyVoice(os.path.join(self.lm_args.lm_model_name_or_path, "CosyVoice-300M-25Hz"))
        self.music_cosy_model = CosyVoice(
            os.path.join(self.lm_args.lm_model_name_or_path, "CosyVoice-300M-25Hz-Music")
        )
        self.encoder = StepAudioTokenizer(self.args.tokenizer_model_name_or_path)

        self.ref_encode_codebook_indices_dir = self.ref_codebook_indices_dir()
        self.voices = self.load_ref_voices()
        self.ref_encode_codebook_indices: torch.Tensor = None
        self.args.ref_audio_file_path and self.set_voice(self.args.ref_audio_file_path)

        # lm model gen warmup, codec model decode(flow + hifi) don't to warmup

    def ref_codebook_indices_dir(self):
        """
        mkdir -p $ref_dir
        """
        os.makedirs(self.args.output_codebook_indices_dir, exist_ok=True)

        return self.args.output_codebook_indices_dir

    def set_voice(self, ref_audio_path: str):
        """
        - if ref_audio_path is audio path, encode audio to ref_encode_codebook_indices (audio vq codes)
        - if ref_audio_path is encode_codebook_indices path(.npy), load it
        """
        if os.path.exists(ref_audio_path) is False:
            raise FileNotFoundError(f"reference_audio_path: {ref_audio_path}")
        input_path = Path(ref_audio_path)

        if input_path.suffix == ".npy":
            self.ref_encode_codebook_indices = torch.from_numpy(np.load(input_path)).to(
                self.args.lm_args.lm_device
            )
            md5_hash = file_md5_hash(ref_audio_path)
            self.voices[md5_hash] = self.ref_encode_codebook_indices
            return
        if input_path.suffix not in AUDIO_EXTENSIONS:
            raise ValueError(f"Invalid audio file: {ref_audio_path}")

        logging.info(f"Processing in-place reconstruction of {input_path}")

        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)  # multi-channel to mono
        if sr != RATE:
            # only 16khz speech support!
            audio = torchaudio.functional.resample(audio, sr, RATE)

        audios = audio[None].to(self.args.xcode2_args.device)
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
        input_text = text
        if self.args.prompt_text:
            input_text += " " + self.args.prompt_text
        session.ctx.state["prompt"] = input_text
        session.ctx.state["vq_code_prompt"] = self.ref_encode_codebook_indices
        speech_vq_tokens = self.lm_model.generate(session, **kwargs)

        for tokens in speech_vq_tokens:
            if tokens.shape[0] > 0:
                # Decode the speech tokens to speech waveform
                gen_wav = self.codec_model.decode_code(tokens[None, None, :])  # shape [T]
                yield gen_wav.float().detach().cpu().numpy().tobytes()
