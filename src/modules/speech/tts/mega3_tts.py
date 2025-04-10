from dataclasses import dataclass, field
import logging
import os
import sys
import time
from typing import AsyncGenerator, List

from dotenv import load_dotenv
import numpy as np

from src.common.utils.helper import get_device, print_model_params
from src.common.random import set_all_random_seed
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.types.speech.tts.mega3 import Mega3TTSArgs
from .base import BaseTTS

load_dotenv(override=True)

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../MegaTTS3"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/MegaTTS3"))

    import torch
    from langdetect import detect as classify_language

    from deps.MegaTTS3.tts.infer_cli import MegaTTS3DiTInfer
    from deps.MegaTTS3.tts.utils.audio_utils.io import to_wav_bytes
    from deps.MegaTTS3.tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english

except ModuleNotFoundError as e:
    logging.error("In order to use mega tts, you need to `pip install achatbot[tts_mega3]`.")
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[tts_mega3]` to install the dependencies."
    )


@dataclass
class RefAudioCtxInfo:
    ph_ref: torch.Tensor = None
    tone_ref: torch.Tensor = None
    mel2ph_ref: torch.Tensor = None
    vae_latent: torch.Tensor = None
    ctx_dur_tokens: torch.Tensor = None
    incremental_state_dur_prompt: dict = field(default_factory=dict)


@dataclass
class RefAudioCodecInfo:
    ref_speaker: str = ""
    ref_text: str = ""
    ref_path: str = ""

    resource_context: RefAudioCtxInfo = None


class MegaTTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/130
    """

    TAG = "tts_mega3"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**Mega3TTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = Mega3TTSArgs(**args)
        self.args.device = self.args.device or get_device()
        logging.debug(f"{MegaTTS.TAG} args: {self.args}")
        self.model = MegaTTS3DiTInfer(ckpt_root=self.args.ckpt_dir, dict_file=self.args.dict_file)
        print_model_params(self.model.g2p_model, "g2p_llm_ar_decoder_model")  # transformer decoder
        print_model_params(
            self.model.aligner_lm, "aligner_whisper_audioEncoder_textDecoder"
        )  # transformer encoder decoder
        print_model_params(self.model.dur_model, "dur_ar_decoder_model")  # transformer decoder
        print_model_params(
            self.model.dit, "dit"
        )  # txt embedding(phone,tone,dur), speaker embedding(wavevae encode audio vq latent feats), local conditioning(latent feats with mask), TimestepEmbedding (timestep with Sway sampling), transformer decoder
        print_model_params(self.model.wavvae, "wavevae")

        self.voices = {}
        self.set_voice(
            self.args.ref_audio_file, latents_file=self.args.ref_latent_file, ref_speaker="default"
        )

        self._warm_up()

    def _warm_up(self):
        """
        Warm up the model with a dummy input to ensure it's ready for real-time processing.
        """
        logging.info(f"Warming up the {self.TAG} model...")
        gen_text = "Warm-up text for the model."

        ref_voice: RefAudioCodecInfo = self.voices.get("default")
        # reference audio context info(phone tone mel2ph vae_latent, dur_tokens and incremental_state)
        ph_ref = ref_voice.resource_context.ph_ref.to(self.args.device)
        tone_ref = ref_voice.resource_context.tone_ref.to(self.args.device)
        mel2ph_ref = ref_voice.resource_context.mel2ph_ref.to(self.args.device)
        vae_latent = ref_voice.resource_context.vae_latent.to(self.args.device)
        ctx_dur_tokens = ref_voice.resource_context.ctx_dur_tokens.to(self.args.device)
        incremental_state_dur_prompt = ref_voice.resource_context.incremental_state_dur_prompt

        start_time = time.perf_counter()
        _ = self.model.gen(
            gen_text,
            ctx_dur_tokens,
            incremental_state_dur_prompt,
            ph_ref,
            tone_ref,
            mel2ph_ref,
            vae_latent,
            self.args.time_step,
            self.args.p_w,
            self.args.t_w,
            is_first=True,
            is_final=True,
            dur_disturb=self.args.dur_disturb,
            dur_alpha=self.args.dur_alpha,
        )
        del _
        logging.info(f"Warm-up completed. cost: {(time.perf_counter() - start_time):.3f} s")

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            # "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": self.model.sr,  # 24000
            "sample_width": 2,
            "np_dtype": np.int16,
            # "np_dtype": np.float32,
        }

    def set_voice(self, ref_file: str, **kwargs):
        ref_text = kwargs.get("ref_text", "")
        ref_speaker = kwargs.get("ref_speaker", ref_file)
        latent_file = kwargs.get("latent_file", self.args.ref_latent_file)
        assert os.path.exists(ref_file), f"{ref_file} not found"
        # wavevae encoder ckpt don't open so just use existing ref latent file
        # download from https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr
        assert os.path.exists(latent_file), f"{latent_file} not found"

        with open(ref_file, "rb") as file:
            file_content = file.read()

        resource_context = self.model.preprocess(file_content, latent_file=latent_file)
        self.voices[ref_speaker] = RefAudioCodecInfo(
            ref_speaker=ref_speaker,
            ref_path=ref_file,
            ref_text=ref_text,
            resource_context=RefAudioCtxInfo(**resource_context),
        )

    def get_voices(self):
        return list(self.voices.keys())

    @torch.inference_mode()
    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        time_step = kwargs.get("time_step", self.args.time_step)
        p_w = kwargs.get("p_w", self.args.p_w)
        t_w = kwargs.get("t_w", self.args.t_w)
        dur_disturb = kwargs.get("dur_disturb", self.args.dur_disturb)
        dur_alpha = kwargs.get("dur_alpha", self.args.dur_alpha)

        ref_speaker = kwargs.get("ref_speaker", "default")
        ref_voice: RefAudioCodecInfo = self.voices.get(ref_speaker)
        if ref_voice is None:
            raise ValueError(f"Invalid voice: {ref_speaker}")

        # reference audio context info(phone tone mel2ph vae_latent, dur_tokens and incremental_state)
        ph_ref = ref_voice.resource_context.ph_ref.to(self.args.device)
        tone_ref = ref_voice.resource_context.tone_ref.to(self.args.device)
        mel2ph_ref = ref_voice.resource_context.mel2ph_ref.to(self.args.device)
        vae_latent = ref_voice.resource_context.vae_latent.to(self.args.device)
        ctx_dur_tokens = ref_voice.resource_context.ctx_dur_tokens.to(self.args.device)
        incremental_state_dur_prompt = ref_voice.resource_context.incremental_state_dur_prompt

        # inference
        with torch.inference_mode():
            language_type = classify_language(text)
            if language_type == "en":
                input_text = self.model.en_normalizer.normalize(text)
                text_segs = chunk_text_english(input_text, max_chars=20)
            else:
                input_text = self.model.zh_normalizer.normalize(text)
                text_segs = chunk_text_chinese(input_text, limit=10)
            logging.debug(f"| Text Segments: {text_segs}")

            for seg_i, text in enumerate(text_segs):
                wav_pred = self.model.gen(
                    text,
                    ctx_dur_tokens,
                    incremental_state_dur_prompt,
                    ph_ref,
                    tone_ref,
                    mel2ph_ref,
                    vae_latent,
                    time_step,
                    p_w,
                    t_w,
                    is_first=seg_i == 0,
                    is_final=seg_i == len(text_segs) - 1,
                    dur_disturb=dur_disturb,
                    dur_alpha=dur_alpha,
                    **kwargs,
                )
                yield to_wav_bytes(wav_pred, self.model.sr)

        torch.cuda.empty_cache()
