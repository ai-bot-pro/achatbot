import logging
import random
import os
import sys
from typing import AsyncGenerator
from importlib.resources import files

from dotenv import load_dotenv
import numpy as np
import torch
import torchaudio

from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAINT16, F5TTSArgs
from src.common.utils.audio_utils import postprocess_tts_wave_int16
from .base import BaseTTS

from deps.F5TTS.src.f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    infer_batch_process,
)
from deps.F5TTS.src.f5_tts.model.backbones.dit import DiT
from deps.F5TTS.src.f5_tts.model.backbones.unett import UNetT
from deps.F5TTS.src.f5_tts.model.utils import seed_everything

cur_dir = os.path.dirname(__file__)
if bool(os.getenv("ACHATBOT_PKG", "")):
    sys.path.insert(1, os.path.join(cur_dir, "../../../F5TTS/src"))
else:
    sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/F5TTS/src"))

load_dotenv(override=True)


class F5TTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/101
    """

    TAG = "tts_f5"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**F5TTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = F5TTSArgs(**args)
        self.args.device = self.args.device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Load the vocoder model
        self.vocoder_model = load_vocoder(
            self.args.vocoder_name,
            self.args.vocoder_ckpt_path is not None,
            self.args.vocoder_ckpt_path,
            self.args.device,
        )

        if self.args.model_type == "E2-TTS":
            model_cls = UNetT
        else:
            model_cls = DiT

        # Load the CFM model using the provided checkpoint and vocab files
        self.cfm_model = load_model(
            model_cls=model_cls,
            model_cfg=self.args.model_cfg,
            ckpt_path=self.args.model_ckpt_path,
            mel_spec_type=self.args.vocoder_name,
            vocab_file=self.args.vocab_file,
            ode_method=self.args.ode_method,
            use_ema=self.args.use_EMA,
            device=self.args.device,
        )

        # torch manual seed
        if self.args.seed == -1:
            self.args.seed = random.randint(0, sys.maxsize)
        seed_everything(self.args.seed)

        # inference ref
        if self.args.ref_audio_file == "":
            self.args.ref_audio_file = str(
                files("deps.F5TTS.src.f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")
            )

        self.ref_file, self.ref_text = preprocess_ref_audio_text(
            self.args.ref_audio_file, self.args.ref_text, device=self.args.device
        )

    def _warm_up(self):
        """
        Warm up the model with a dummy input to ensure it's ready for real-time processing.
        """
        logging.info("Warming up the f5-tts model...")
        audio, sr = torchaudio.load(self.args.ref_audio_file)
        gen_text = "Warm-up text for the model."

        # Pass the vocoder as an argument here
        infer_batch_process(
            (audio, sr),
            self.args.ref_text,
            [gen_text],
            self.cfm_model,
            self.vocoder_model,
            device=self.args.device,
        )
        print("Warm-up completed.")

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            "channels": 1,
            "rate": 24000,  # target_sample_rate
            "sample_width": 2,
            "np_dtype": np.int16,
        }

    def set_voice(self, ref_file: str):
        # use ASR transcribe ref audio to text
        self.args.ref_file, self.args.ref_text = preprocess_ref_audio_text(
            ref_file, "", device=self.args.device
        )

    def get_voices(self):
        return [self.args.ref_text]

    async def _inference(self, session: Session, text: str) -> AsyncGenerator[bytes, None]:
        wav, _, _ = infer_process(
            self.args.ref_audio_file,
            self.args.ref_text,
            text,
            self.cfm_model,
            self.vocoder_model,
            self.args.vocoder_name,
            target_rms=self.args.target_rms,
            cross_fade_duration=self.args.cross_fade_duration,
            nfe_step=self.args.nfe_step,
            cfg_strength=self.args.cfg_strength,
            sway_sampling_coef=self.args.sway_sampling_coef,
            speed=self.args.speed,
            fix_duration=self.args.fix_duration,
            device=self.args.device,
        )
        yield wav.tobytes()
