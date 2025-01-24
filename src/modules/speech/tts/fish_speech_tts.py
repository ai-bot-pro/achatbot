import logging
import os
from pathlib import Path
import sys
from typing import AsyncGenerator

import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np
import torch
import torchaudio

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../FishSpeech"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/FishSpeech"))
    from deps.FishSpeech.fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
    from deps.FishSpeech.fish_speech.models.text2semantic.inference import load_model, generate_long
    from deps.FishSpeech.fish_speech.utils.utils import set_seed
    from deps.FishSpeech.fish_speech.utils.file import AUDIO_EXTENSIONS
    from deps.FishSpeech.fish_speech.utils.context import autocast_exclude_mps
    from deps.FishSpeech.fish_speech.text.chn_text_norm.text import Text as ChnNormedText
except ModuleNotFoundError as e:
    logging.error(
        "In order to use fishspeech-tts, you need to `pip install achatbot[tts_fishspeech]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import file_md5_hash, get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.fish_speech import FishSpeechTTSArgs
from .base import BaseTTS


class FishSpeechTTS(BaseTTS, ITts):
    TAG = "tts_fishspeech"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**FishSpeechTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = FishSpeechTTSArgs(**args)
        self.args.device = self.args.device or get_device()

        self.precision = torch.half if self.args.half else torch.bfloat16
        self.set_seed()
        self.ref_encode_codebook_indices_dir = self.ref_codebook_indices_dir()
        self.voices = self.load_ref_voices()

        # load Dual AR GLM (Llama2)
        self.dual_ar_model, self.decode_one_token = self.load_dual_ar_model()
        # load Firefly GAN Arch model
        self.gan_model: FireflyArchitecture = self.load_gan_model()

        self.ref_encode_codebook_indices: torch.Tensor = None
        self.args.ref_audio_path and self.set_voice(self.args.ref_audio_path)

        self.warm_up()

    def set_seed(self):
        """
        Set the random seed
        """
        set_seed(self.args.seed)

    def load_ref_voices(self):
        """
        lazy load reference voices
        """
        voices = {}
        return voices

    def ref_codebook_indices_dir(self):
        """
        mkdir -p $ref_dir
        """
        os.makedirs(self.args.output_codebook_indices_dir, exist_ok=True)

        return self.args.output_codebook_indices_dir

    def load_dual_ar_model(self):
        """
        - load Dual-AR LM ckpt;
        - setup kv cache for prefilling and decoding
            - prefilling: prefill_decode(decode first token for TTFT(Time to First Token))
            - decoding: decode_n_tokens for TPOT(Time per Output Token)

        if deploy inference serving: (a/aa,b/bb,c/cc depends on GPU and api session requests)
        - prefilling(Attention(TP_a,SP,DP_b),MLP(TP_c))
        - decoding(Attention(TP_aa,SP,DP_bb),MLP(TP_cc))
        """
        logging.info(f"Loading Dual-AR LM model from {self.args.lm_checkpoint_dir} ...")
        model, decode_one_token = load_model(
            self.args.lm_checkpoint_dir,
            self.args.device,
            self.precision,
            compile=self.args.compile,
        )
        print_model_params(model, "dual_ar_lm")

        if self.args.kv_cache is True:
            with torch.device(self.args.device):
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=model.config.max_seq_len,
                    dtype=next(model.parameters()).dtype,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logging.info(f"Load Dual-AR LM model")
        return model, decode_one_token

    def load_gan_model(self):
        """
        load FireFly-GAN model ckpt;
        """
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=self.args.gan_config_path):
            cfg = compose(config_name=self.args.gan_config_name)

        model = instantiate(cfg)
        print_model_params(model, "firefly_vq_gan")
        state_dict = torch.load(
            self.args.gan_checkpoint_path,
            map_location=self.args.device,
            mmap=True,
            weights_only=True,
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k
            }

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        model.to(self.args.device)

        logging.info(f"Loaded FireFlyGAN model: {result}")
        return model

    def warm_up(self):
        if not self.args.warm_up_text:
            logging.warning(f"No warm_up_text to Warm Up")
            return
        logging.info(f"Start Warm Up")
        generate_codebook_indices_iter = generate_long(
            model=self.dual_ar_model,
            device=self.args.device,
            decode_one_token=self.decode_one_token,
            text=self.args.warm_up_text,
            num_samples=2,  # warm up 2 samples
            max_new_tokens=self.args.max_new_tokens,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            temperature=self.args.temperature,
            compile=self.args.compile,
            iterative_prompt=self.args.iterative_prompt,
            chunk_length=self.args.chunk_length,
            prompt_text=[],  # don't use None
            prompt_tokens=[],  # don't use None
        )
        for chunk in generate_codebook_indices_iter:
            logging.debug(f"Warm Up chunk {chunk}")
            if chunk.action == "sample" and chunk.codes is not None:
                self.decode_vq_tokens(chunk.codes)

        logging.info(f"End Warm Up")

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
        audio = torchaudio.functional.resample(audio, sr, self.gan_model.spec_transform.sample_rate)

        audios = audio[None].to(self.args.device)
        logging.info(
            f"Loaded audio with {audios.shape[2] / self.gan_model.spec_transform.sample_rate:.2f} seconds"
        )

        indices = self.encode_audios(audios)
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

    def get_audio_segment(self, code_indices: torch.Tensor) -> np.ndarray:
        """
        Decode the VQ tokens to audio segment.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.gan_model.device.type,
            dtype=self.precision,
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(code_indices=code_indices)

        # Convert the audio to numpy(float32)
        res = segment.float().detach().cpu().numpy()
        return res

    def encode_audios(self, audios: torch.Tensor) -> torch.Tensor:
        """
        # Firefly-GAN Encoder
        1. LogMelSpectrogram with STFT (torch.stft) input audio waveform transform to Mel spec
        2. ConvNeXt Encoder input Mel spec encode(downsample) to tensor $z_d$
        3. DownsampleFiniteScalarQuantize with grouped FSQ input downsampled tensor $z_d$ encode(downsample) to vq codebook indices (quantized Mel spec)

        return codebook indices, tensor shape is 1D
        """
        audio_lengths = torch.tensor([audios.shape[2]], device=self.args.device, dtype=torch.long)
        indices = self.gan_model.encode(audios, audio_lengths)[0][0]

        return indices

    def decode_vq_tokens(self, code_indices: torch.Tensor) -> torch.Tensor:
        """
        # Firefly-GAN Decoder
        1. DownsampleFiniteScalarQuantize with grouped FSQ input vq codebook indices decode (upsample) to Mel spec
        2. Firefly Generator (firefly.HiFiGANGenerator) input Mel spec to waveform

        return waveform, tensor shape is 1D
        """
        feature_lengths = torch.tensor([code_indices.shape[1]], device=self.gan_model.device)

        waveform = self.gan_model.decode(
            indices=code_indices[None],
            feature_lengths=feature_lengths,
        )[0].squeeze()

        logging.info(
            f"VQ features: {code_indices.shape}, feature_lengths: {feature_lengths}, waveform: {waveform}"
        )
        return waveform

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            # https://github.com/fishaudio/fish-speech/blob/main/fish_speech/configs/firefly_gan_vq.yaml
            # "rate": self.gan_model.spec_transform.sample_rate,
            "rate": 44100,
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if self.args.normalize:
            text = ChnNormedText(raw_text=text).normalize()

        prompt_text = []
        prompt_tokens = []
        if self.ref_encode_codebook_indices is not None and self.args.ref_text is not None:
            prompt_text.append(self.args.ref_text)
            prompt_tokens.append(self.ref_encode_codebook_indices)

        generate_codebook_indices_iter = generate_long(
            model=self.dual_ar_model,
            device=self.args.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=self.args.max_new_tokens,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            temperature=self.args.temperature,
            compile=self.args.compile,
            iterative_prompt=self.args.iterative_prompt,
            chunk_length=self.args.chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )
        for chunk in generate_codebook_indices_iter:
            logging.debug(f"inference GenerateResponse {chunk}")
            if chunk.action == "sample" and chunk.codes is not None:
                audio_segment = self.get_audio_segment(chunk.codes)

                yield audio_segment.tobytes()
                # yield audio_segment.astype(np.float32).tobytes()
