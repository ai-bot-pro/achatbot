from dataclasses import dataclass, field
import logging
import math
import random
import os
import re
import sys
from typing import AsyncGenerator, List

from dotenv import load_dotenv
import numpy as np

from src.common.utils.helper import get_device
from src.common.random import set_all_random_seed
from src.core.llm.transformers.manual_speech_spark import TransformersManualSpeechSpark
from src.common.interface import ITts
from src.common.session import Session
from src.common.types import PYAUDIO_PAFLOAT32
from src.types.speech.tts.spark import SparkTTSArgs
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import BaseTTS

load_dotenv(override=True)

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../SparkTTS"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/SparkTTS"))

    import torch

    from deps.SparkTTS.sparktts.models.audio_tokenizer import BiCodecTokenizer
    from deps.SparkTTS.sparktts.utils.file import load_config

except ModuleNotFoundError as e:
    logging.error(
        "In order to use spark tts, you need to `pip install achatbot[tts_spark]`.\nPlease install the missing modules."
    )
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[tts_spark]` to install the dependencies."
    )


@dataclass
class RefAudioCodecInfo:
    ref_speaker: str = ""
    ref_text: str = ""
    ref_path: str = ""
    # [global_token_ids, semantic_token_ids]
    vq_indices: List[torch.Tensor] = field(default_factory=list)


class SparkTTS(BaseTTS, ITts):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/130
    """

    TAG = "tts_spark"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**SparkTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = SparkTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        # self.args.lm_args["lm_model_name_or_path"] = os.path.join(self.args.model_dir, "/LLM")
        logging.debug(f"{SparkTTS.TAG} args: {self.args}")

        self.lm_args = TransformersSpeechLMArgs(**self.args.lm_args)
        self.lm_model = TransformersManualSpeechSpark(**self.lm_args.__dict__)
        self.lm_tokenizer = self.lm_model.tokenizer

        self.start_global_token_id = self.lm_tokenizer.encode("<|start_global_token|>")[0]
        self.start_semantic_token_id = self.lm_tokenizer.encode("<|start_semantic_token|>")[0]
        logging.debug(
            f"start_global_token_id:{self.start_global_token_id} start_semantic_token_id:{self.start_semantic_token_id}"
        )

        self.configs = load_config(f"{self.args.model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self.audio_tokenizer = BiCodecTokenizer(
            self.args.model_dir, device=torch.device(self.args.device)
        )

        self.voices = {}
        if self.args.ref_audio_path:
            self.set_voice(
                self.args.ref_audio_path, ref_text=self.args.ref_text, ref_speaker="default"
            )
        self.default_gender = "male" if random.random() > 0.5 else "female"

    def get_stream_info(self) -> dict:
        return {
            # "format": PYAUDIO_PAINT16,
            "format": PYAUDIO_PAFLOAT32,
            "channels": 1,
            "rate": 16000,  # target_sample_rate
            "sample_width": 2,
            # "np_dtype": np.int16,
            "np_dtype": np.float32,
        }

    def set_voice(self, ref_file: str, **kwargs):
        ref_text = kwargs["ref_text"] if "ref_text" in kwargs else self.args.ref_text
        ref_speaker = kwargs["ref_speaker"] if "ref_speaker" in kwargs else ref_file
        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(ref_file)
        self.voices[ref_speaker] = RefAudioCodecInfo(
            ref_speaker=ref_speaker,
            ref_path=ref_file,
            ref_text=ref_text,
            vq_indices=[global_token_ids, semantic_token_ids],
        )

    def get_voices(self):
        return list(self.voices.keys())

    def token2wav(self, generated_ids: torch.Tensor, gender: str, global_token_ids: torch.Tensor):
        """
        generated_ids -- tokenizer.decode --> sematic tokens + global tokens  -- audio_tokenizer.detokenize --> waveform
        """
        # print("generated_ids", generated_ids)
        # Decode the generated tokens into text (just a mapping, so quick,don't worry)
        predicts = self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("predicts", predicts)

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        ).to(self.args.device)

        # Check if pred_semantic_ids is empty
        if pred_semantic_ids.numel() == 0:
            logging.warning(f"No semantic tokens found, return empty waveform.")
            return np.array([])

        if pred_semantic_ids.dim() == 2 and pred_semantic_ids.numel() > 0:
            # print(pred_semantic_ids)
            # Check if all generated tokens are the same
            mask = pred_semantic_ids == pred_semantic_ids[0][0].item()
            if mask.all():
                logging.warning(f"All silence tokens, return empty waveform.")
                return np.array([])

        if gender is not None:
            # Tips: generated_id - global_vq_index = 151665
            global_token_ids = (
                torch.tensor(
                    [int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)]
                )
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Check if global_token_ids is empty
        if global_token_ids.numel() == 0:
            logging.warning(f"No global tokens found, return empty waveform.")
            return np.array([])

        # print("global_token_ids", global_token_ids.shape)
        # print("pred_semantic_ids", pred_semantic_ids.shape)
        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.args.device).squeeze(0),
            pred_semantic_ids,
        )

        return wav

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.lm_model._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.lm_args.lm_gen_seed)
        set_all_random_seed(seed)

        is_meet_start_global_token = False
        is_meet_start_semantic_token = False
        controll_gen_global_token_ids = []
        semantic_token_ids = []

        stream_factor = kwargs.get("stream_factor", self.args.stream_factor)
        stream_scale_factor = kwargs.get("stream_scale_factor", self.args.stream_scale_factor)
        max_stream_factor = kwargs.get("max_stream_factor", self.args.max_stream_factor)
        token_overlap_len = kwargs.get("token_overlap_len", self.args.token_overlap_len)
        input_frame_rate = kwargs.get("input_frame_rate", self.args.input_frame_rate)

        max_batch_size = math.ceil(max_stream_factor * input_frame_rate)
        batch_size = math.ceil(stream_factor * input_frame_rate)
        logging.info(f"init batch_size: {batch_size} max_batch_size: {max_batch_size}")

        ref_speaker = kwargs["ref_speaker"] if "ref_speaker" in kwargs else "default"
        ref_voice: RefAudioCodecInfo = self.voices.get(ref_speaker)
        if ref_voice is None:
            if "gender" not in kwargs:
                kwargs["gender"] = self.default_gender
            logging.warning(
                f"Voice {ref_speaker} not found, use Controlled Generation inference with gender:{kwargs['gender']} attribute."
            )

        session.ctx.state["prompt"] = text
        if ref_voice:
            session.ctx.state["global_fsq_indices"] = ref_voice.vq_indices[0]
            session.ctx.state["semantic_vq_indices"] = ref_voice.vq_indices[1]
            session.ctx.state["ref_text"] = ref_voice.ref_text

        streamer = self.lm_model.generate(session, **kwargs)
        gender = kwargs.get("gender", None)

        for token_id in streamer:
            if gender is not None:
                # Inference Overview of Controlled Generation
                if is_meet_start_global_token is False and token_id != self.start_global_token_id:
                    continue
                if is_meet_start_global_token is False and token_id == self.start_global_token_id:
                    is_meet_start_global_token = True
                    controll_gen_global_token_ids.append(token_id)
                    continue
                # append global token until meet start_global_token
                if (
                    is_meet_start_global_token is True
                    and is_meet_start_semantic_token is False
                    and token_id != self.start_global_token_id
                ):
                    controll_gen_global_token_ids.append(token_id)

                if (
                    is_meet_start_semantic_token is False
                    and token_id != self.start_semantic_token_id
                ):
                    continue
                if (
                    is_meet_start_semantic_token is False
                    and token_id == self.start_semantic_token_id
                ):
                    is_meet_start_semantic_token = True
                    continue
                # do batch stream until meet start_semantic_token
                if (
                    is_meet_start_semantic_token is True
                    and token_id != self.start_semantic_token_id
                ):
                    # print(controll_gen_global_token_ids)
                    pass

            semantic_token_ids.append(token_id)
            # if len(semantic_token_ids) % batch_size == 0:
            if len(semantic_token_ids) >= batch_size + token_overlap_len:
                batch = semantic_token_ids[: batch_size + token_overlap_len]
                # Process each batch
                sub_tts_speech = self.token2wav(
                    [controll_gen_global_token_ids + batch],
                    gender,
                    session.ctx.state.get("global_fsq_indices", None),
                )  # one batch
                semantic_token_ids = semantic_token_ids[batch_size:]
                if sub_tts_speech.size == 0:
                    break
                yield np.frombuffer(sub_tts_speech, dtype=float).tobytes()
                # increase token_hop_len for better speech quality
                batch_size = min(max_batch_size, int(batch_size * stream_scale_factor))
                logging.info(
                    f"increase batch_size: {batch_size} token_overlap_len:{token_overlap_len}"
                )

        if len(semantic_token_ids) > 0:  # end to finalize
            # Process each batch
            sub_tts_speech = self.token2wav(
                [controll_gen_global_token_ids + semantic_token_ids],
                gender,
                session.ctx.state.get("global_fsq_indices", None),
            )  # one batch
            yield np.frombuffer(sub_tts_speech, dtype=float).tobytes()
            logging.info(f"last batch len: {len(semantic_token_ids)}")

        torch.cuda.empty_cache()
