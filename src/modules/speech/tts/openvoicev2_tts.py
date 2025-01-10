import io
import os
import sys
import logging
from typing import AsyncGenerator
from datetime import datetime

import torch
import numpy as np
import soundfile
from dotenv import load_dotenv
from melo.api import TTS
import torchaudio


try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../OpenVoice"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/OpenVoice"))
    from deps.OpenVoice.openvoice import se_extractor
    from deps.OpenVoice.openvoice.api import ToneColorConverter
except ModuleNotFoundError as e:
    logging.error(
        "In order to use openvoice-tts, you need to `pip install achatbot[tts_openvoicev2]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils import audio_utils
from src.common.session import Session
from src.common.interface import ITts
from src.modules.speech.tts.base import BaseTTS
from src.common.types import ASSETS_DIR, MODELS_DIR, PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16, RECORDS_DIR
from src.types.speech.tts.openvoicev2 import OpenVoiceV2TTSArgs

load_dotenv(override=True)


class OpenVoiceV2TTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/103
    """

    TAG = "tts_openvoicev2"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**OpenVoiceV2TTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = OpenVoiceV2TTSArgs(**args)
        self.args.device = self.args.device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logging.debug(f"{OpenVoiceV2TTS.TAG} args: {self.args}")

        # load melo tts model
        self.load_melo_tts()
        # load tone color converter model
        self.load_tone_color_converter()
        # load src/target tone color stats
        self.load_tone_color_stats()

        # set target_sample_rate
        # just use tone_color_converter_model sampling rate
        self.target_sample_rate = self.tone_color_converter_model.hps.data.sampling_rate

        os.makedirs(RECORDS_DIR, exist_ok=True)
        self._warm_up()

    def load_melo_tts(self):
        # download melo-tts model ckpt
        # huggingface-cli download myshell-ai/MeloTTS-English-v3 --local-dir ./models/myshell-ai/MeloTTS-English-v3
        # huggingface-cli download myshell-ai/MeloTTS-Chinese --local-dir ./models/myshell-ai/MeloTTS-Chinese
        self.tts_model = TTS(
            language=self.args.language,
            device=self.args.device,
            config_path=self.args.tts_config_path,
            ckpt_path=self.args.tts_ckpt_path,
        )
        logging.debug(f"melo tts model config: {self.tts_model.hps}")
        logging.debug(f"melo tts model: {self.tts_model}")
        model_million_params = sum(p.numel() for p in self.tts_model.parameters()) / 1e6
        logging.debug(f"melo tts model params: {model_million_params}")

    def load_tone_color_converter(self):
        # download openvoice converter model ckpt
        # huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ./models/myshell-ai/OpenVoiceV2
        self.tone_color_converter_model = ToneColorConverter(
            self.args.converter_conf_path, device=self.args.device
        )
        self.tone_color_converter_model.load_ckpt(self.args.converter_ckpt_path)
        logging.debug(f"tone_color_converter model config: {self.tone_color_converter_model.hps}")
        logging.debug(f"tone_color_converter model: {self.tone_color_converter_model.model}")
        model_million_params = (
            sum(p.numel() for p in self.tone_color_converter_model.model.parameters()) / 1e6
        )
        logging.debug(f"tone_color_converter model params: {model_million_params}")
        logging.debug(
            f"tone_color_converter watermark model: {self.tone_color_converter_model.watermark_model}"
        )
        model_million_params = (
            sum(p.numel() for p in self.tone_color_converter_model.watermark_model.parameters())
            / 1e6
        )
        logging.debug(f"tone_color_converter watermark model params: {model_million_params}")

    def load_tone_color_stats(self):
        self.src_se_stats_tensor = None
        if self.args.src_se_ckpt_path:
            self.src_se_stats_tensor = torch.load(
                self.args.src_se_ckpt_path,
                map_location=self.args.device,
            )
        self.target_se_stats_tensor = None
        if self.args.target_se_ckpt_path:
            self.target_se_stats_tensor = torch.load(
                self.args.target_se_ckpt_path,
                map_location=self.args.device,
            )

    def _warm_up(self):
        """
        Warm up the model with a dummy input to ensure it's ready for real-time processing.
        """
        logging.info("Warming up the Melo-TTS model...")
        gen_text = "Warm-up text for the model."
        np_audio_data = self.tts_model.tts_to_file(
            gen_text,
            self.tts_model.hps.data.spk2id[self.args.language],
            None,
            speed=self.args.speed,
            sdp_ratio=self.args.sdp_ratio,
            noise_scale=self.args.noise_scale,
            noise_scale_w=self.args.noise_scale_w,
            quiet=self.args.quiet,
        )

        if (
            self.args.enable_clone is True
            and self.src_se_stats_tensor is not None
            and self.target_se_stats_tensor is not None
        ):
            logging.info("Warming up the OpenVoiceV2 tone_color_converter model...")
            audio_buf = io.BytesIO()
            soundfile.write(
                audio_buf,
                np_audio_data.astype(dtype=np.float32),
                self.tts_model.hps.data.sampling_rate,
                format="WAV",
            )
            audio_buf.seek(0)
            src_path = soundfile.SoundFile(audio_buf)

            self.tone_color_converter_model.convert(
                audio_src_path=src_path,
                src_se=self.src_se_stats_tensor,
                tgt_se=self.target_se_stats_tensor,
                output_path=None,
                message=self.args.watermark_name,
            )

            src_path.close()
            audio_buf.close()

        logging.info("Warm-up completed.")

    def save(self, data: np.ndarray, rate: int, file_name: str = "melo_tts"):
        output_path = os.path.join(RECORDS_DIR, f"{file_name}_{self.args.language}.wav")
        soundfile.write(output_path, data, rate)
        return output_path

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": self.target_sample_rate,  # target_sample_rate
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    def set_voice(self, target_se_path: str):
        """set target tone color stats"""

        if target_se_path.endswith(".pth"):
            self.target_se_stats_tensor = torch.load(target_se_path, map_location=self.args.device)
            self.tts_model.hps.data.spk2id["custom"] = target_se_path
        else:
            self.target_se_stats_tensor, target_se_path = self.reference_target_se_extractor(
                target_se_path
            )
            self.tts_model.hps.data.spk2id["custom"] = target_se_path

    def get_voices(self):
        return list(self.tts_model.hps.data.spk2id.values())

    def reference_target_se_extractor(
        self,
        reference_speaker_file: str = os.path.join(ASSETS_DIR, "basic_ref_zh.wav"),
    ) -> tuple[torch.Tensor, str]:
        """This is the voice you want to clone"""

        target_dir = os.path.join(RECORDS_DIR, "openvoicev2")
        os.makedirs(target_dir, exist_ok=True)
        vad = True  # False use whisper, True use silero vad
        target_se, audio_name = se_extractor.get_se(
            reference_speaker_file, self.tone_color_converter_model, target_dir=target_dir, vad=vad
        )
        se_path = os.path.join(target_dir, audio_name, "se.pth")
        logging.info(
            f"target tone color shape: {target_se.shape},saved target tone color file: {se_path}"
        )

        return target_se, se_path

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        sample_rate = self.tts_model.hps.data.sampling_rate
        np_audio_data = self.tts_model.tts_to_file(
            text,
            self.tts_model.hps.data.spk2id[self.args.language],
            None,
            speed=self.args.speed,
            sdp_ratio=self.args.sdp_ratio,
            noise_scale=self.args.noise_scale,
            noise_scale_w=self.args.noise_scale_w,
            quiet=self.args.quiet,
        )
        res = np_audio_data

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_"
        file_name_prefix = current_time + f"{session.ctx.client_id}_{session.chat_round}"
        src_path = ""
        if self.args.is_save:
            src_path = self.save(
                np_audio_data,
                self.tts_model.hps.data.sampling_rate,
                file_name=f"{file_name_prefix}_melo_tts",
            )

        if (
            self.args.enable_clone is True
            and self.src_se_stats_tensor is not None
            and self.target_se_stats_tensor is not None
        ):
            if not src_path:
                logging.info("no src_path, use soundfile")
                audio_buf = io.BytesIO()
                soundfile.write(
                    audio_buf,
                    np_audio_data.astype(dtype=np.float32),
                    self.tts_model.hps.data.sampling_rate,
                    format="WAV",
                )
                audio_buf.seek(0)
                src_path = soundfile.SoundFile(audio_buf)

            np_convert_audio_data = self.tone_color_converter_model.convert(
                audio_src_path=src_path,
                src_se=self.src_se_stats_tensor,
                tgt_se=self.target_se_stats_tensor,
                output_path=None,
                message=self.args.watermark_name,
            )
            res = np_convert_audio_data

            if self.args.is_save:
                self.save(
                    np_convert_audio_data,
                    self.tone_color_converter_model.hps.data.sampling_rate,
                    file_name=f"{file_name_prefix}_openvoicev2_tts",
                )

            if isinstance(src_path, soundfile.SoundFile):
                src_path.close()
                audio_buf.close()

            sample_rate = self.tone_color_converter_model.hps.data.sampling_rate

        audio = torch.from_numpy(res)
        if self.target_sample_rate != sample_rate:
            logging.debug(f"Resample {sample_rate} -> {self.target_sample_rate}")
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio = resampler(torch.from_numpy(res))

        # yield audio_utils.postprocess_tts_wave_int16(torch.from_numpy(res))

        # use float32 have high-quality wave
        yield audio_utils.postprocess_tts_wave(audio)
        # yield np.frombuffer(res, dtype=np.float32).tobytes()
