import logging
import os
from pathlib import Path
import sys
import time
from typing import AsyncGenerator

import numpy as np
import torch
import torchaudio

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../Zonos"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/Zonos"))
    from deps.Zonos.zonos.model import Zonos
    from deps.Zonos.zonos.conditioning import make_cond_dict
except ModuleNotFoundError as e:
    logging.error(
        "In order to use zonos-tts use transformer, you need to `apt install -y espeak-ng`,  `pip install achatbot[tts_zonos]`."
        + "if use transformer + mabama2, need `pip install achatbot[tts_zonos_hybrid]`"
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.audio_utils import AUDIO_EXTENSIONS
from src.common.random import set_all_random_seed
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import file_md5_hash, get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.zonos import ZonosTTSArgs
from .base import BaseTTS


class ZonosSpeechTTS(BaseTTS, ITts):
    TAG = "tts_zonos"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**ZonosTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.voices = {}
        self.args = ZonosTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        logging.debug(f"args:{self.args}")
        repo_id = "/".join(self.args.lm_checkpoint_dir.split("/")[-2:])
        self.model = Zonos.from_pretrained(
            repo_id,
            device=self.args.device,
            local_dir=self.args.lm_checkpoint_dir,
            dac_model_path=self.args.dac_model_dir,
        )
        self.model.requires_grad_(False).eval()

        self.speaker: torch.Tensor = None
        if self.args.ref_audio_file_path and os.path.exists(self.args.ref_audio_file_path):
            wav, sr = torchaudio.load(self.args.ref_audio_file_path)
            self.speaker = self.model.make_speaker_embedding(
                wav, sr, local_dir=self.args.speaker_embedding_model_dir
            )

        self.warm_up()

    def warm_up(self):
        if not self.args.warm_up_text:
            logging.warning(f"No warm_up_text to Warm Up")
            return

        logging.info(f"Warming up {self.__class__.__name__} device: {self.model.device}")
        cond_dict = make_cond_dict(
            text=self.args.warm_up_text, speaker=self.speaker, language=self.args.language
        )
        conditioning = self.model.prepare_conditioning(cond_dict)
        stream_generator = self.model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,  # no audio prefix
            chunk_size=self.args.chunk_size,
        )

        if "cuda" in str(self.model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for step in range(self.args.warmup_steps):
            times = []
            start_time = time.perf_counter()
            for _ in stream_generator:
                times.append(time.perf_counter() - start_time)
                start_time = time.perf_counter()
            logging.info(f"step {step} warnup TTFT(chunk) time: {times[0]} s")

        if "cuda" in str(self.model.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )
        logging.info(f"End Warm Up")

    def set_voice(self, ref_audio_path: str):
        """
        - ref_audio_path is audio path, make speaker embedding
        """
        if os.path.exists(ref_audio_path) is False:
            raise FileNotFoundError(f"reference_audio_path: {ref_audio_path}")

        md5_hash = file_md5_hash(ref_audio_path)
        if md5_hash in self.voices:
            logging.info(f"{ref_audio_path} had set speaker embedding")
            return

        wav, sr = torchaudio.load(ref_audio_path)
        self.speaker = self.model.make_speaker_embedding(
            wav, sr, local_dir=self.args.speaker_embedding_model_dir
        )
        self.voices[md5_hash] = self.speaker

    def get_voices(self) -> list:
        return list(self.voices.keys())

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            # "rate": self.gan_model.spec_transform.sample_rate,
            # https://huggingface.co/descript/dac_44khz/blob/main/config.json
            "rate": self.model.autoencoder.sampling_rate,  # 44_100
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        # logging.debug(f"session:{session}, text:{text}")
        if "cuda" in str(self.model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.seed)

        set_all_random_seed(seed)

        # default yield every 40 generated tokens,
        # less for faster streaming but lower quality
        chunk_size = kwargs.get("chunk_size", self.args.chunk_size)
        language = kwargs.get("language", self.args.language)
        emotion = kwargs.get("emotion", self.args.emotion)
        fmax = kwargs.get("fmax", self.args.fmax)
        pitch_std = kwargs.get("pitch_std", self.args.pitch_std)
        speaking_rate = kwargs.get("speaking_rate", self.args.speaking_rate)
        vqscore_8 = kwargs.get("vqscore_8", self.args.vqscore_8)
        ctc_loss = kwargs.get("ctc_loss", self.args.ctc_loss)
        dnsmos_ovrl = kwargs.get("dnsmos_ovrl", self.args.dnsmos_ovrl)

        # Create the conditioning dictionary (using text, speaker embedding, language, etc.).
        cond_dict = make_cond_dict(
            text=text,
            speaker=self.speaker,
            language=language,
            emotion=emotion,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            vqscore_8=vqscore_8,
            ctc_loss=ctc_loss,
            dnsmos_ovrl=dnsmos_ovrl,
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        logging.debug("Starting streaming generation...")
        stream_generator = self.model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,  # no audio prefix in this test
            chunk_size=chunk_size,
        )

        for sr_out, codes_chunk in stream_generator:
            logging.debug(f"Received codes chunk of shape: {codes_chunk.shape}, rate: {sr_out}")
            # [1,1,T]->[T]
            tensor_audio_chunk = self.model.autoencoder.decode(codes_chunk)[0][0]
            # Convert the audio to numpy(float32)
            nparr_audio_chunk = tensor_audio_chunk.detach().cpu().numpy()

            yield nparr_audio_chunk.tobytes()
