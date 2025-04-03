import logging
import math
import random
import os
import re
import sys
from typing import AsyncGenerator, List

import numpy as np

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../SparkTTS"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/SparkTTS"))

    from transformers import AutoTokenizer, GenerationConfig
    import torch

    from deps.SparkTTS.sparktts.models.audio_tokenizer import BiCodecTokenizer
    from deps.SparkTTS.sparktts.utils.file import load_config
    from deps.SparkTTS.sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


except ModuleNotFoundError as e:
    logging.error(
        "In order to use spark tts, you need to `pip install achatbot[tts_spark]`.\nPlease install the missing modules."
    )
    raise Exception(
        f"Missing module: {e}. Please run `pip install achatbot[tts_spark]` to install the dependencies."
    )


from src.common.utils.helper import get_device
from src.common.interface import ILlmGenerator
from src.core.llm import LLMEnvInit
from src.common.random import set_all_random_seed
from src.common.session import Session
from src.types.speech.tts.spark import SparkGeneratorTTSArgs
from src.modules.speech.tts.spark_tts import SparkTTS, RefAudioCodecInfo


class SparkGeneratroTTS(SparkTTS):
    r"""
    https://github.com/ai-bot-pro/achatbot/pull/136
    """

    TAG = "tts_generator_spark"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**SparkGeneratorTTSArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = SparkGeneratorTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        # self.args.lm_args["lm_model_name_or_path"] = os.path.join(self.args.model_dir, "/LLM")
        logging.debug(f"{self.TAG} args: {self.args}")

        self.lm_model: ILlmGenerator = LLMEnvInit.initLLMEngine(
            self.args.lm_generator_tag, self.args.lm_args
        )
        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.args.lm_tokenzier_dir)

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

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    def process_prompt(
        self,
        text: str,
        global_token_ids: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        ref_text: str = None,
    ) -> str:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            global_token_ids, semantic_token_ids (tensor)
            ref_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        # Prepare the input tokens for the model
        if ref_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                ref_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.args.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", random.randint(20, 2**10))
        if seed is not None:
            set_all_random_seed(seed)

        is_meet_start_global_token = False
        is_meet_start_semantic_token = False
        controll_gen_global_token_ids = []
        semantic_token_ids = []

        stream_factor = kwargs.pop("stream_factor", self.args.stream_factor)
        stream_scale_factor = kwargs.pop("stream_scale_factor", self.args.stream_scale_factor)
        max_stream_factor = kwargs.pop("max_stream_factor", self.args.max_stream_factor)
        token_overlap_len = kwargs.pop("token_overlap_len", self.args.token_overlap_len)
        input_frame_rate = kwargs.pop("input_frame_rate", self.args.input_frame_rate)

        max_chunk_size = math.ceil(max_stream_factor * input_frame_rate)
        chunk_size = math.ceil(stream_factor * input_frame_rate)
        logging.info(f"init chunk_size: {chunk_size} max_chunk_size: {max_chunk_size}")

        gender = None
        ref_speaker = kwargs.pop("ref_speaker", "default")
        ref_voice: RefAudioCodecInfo = self.voices.get(ref_speaker)
        if ref_voice is None:
            gender = self.default_gender
            logging.warning(
                f"Voice {ref_speaker} not found, use Controlled Generation inference with gender:{gender} attribute."
            )

        global_fsq_indices = None
        if gender is not None:
            pitch = kwargs.pop("pitch", "moderate")
            speed = kwargs.pop("speed", "moderate")
            prompt = self.process_prompt_control(gender, pitch, speed, text)
        else:
            global_fsq_indices = ref_voice.vq_indices[0]
            semantic_vq_indices = ref_voice.vq_indices[1]
            prompt = self.process_prompt(
                text, global_fsq_indices, semantic_vq_indices, ref_voice.ref_text
            )

        model_inputs = self.lm_tokenizer(prompt)
        session.ctx.state["token_ids"] = model_inputs["input_ids"]
        kwargs["pad_token_id"] = kwargs.get("pad_token_id", None) or self.lm_tokenizer.pad_token_id
        gen_kwargs = {**kwargs, **model_inputs}
        logging.info(f"gen_kwargs: {gen_kwargs}")
        streamer = self.lm_model.generate(session, **gen_kwargs)

        pre_sub_tts_speech_size = 0
        async for token_id in streamer:
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
            # if len(semantic_token_ids) % chunk_size == 0:
            if len(semantic_token_ids) >= chunk_size:
                batch = semantic_token_ids[:chunk_size]
                # Process each batch
                sub_tts_speech = self.token2wav(
                    [controll_gen_global_token_ids + batch],
                    gender,
                    global_fsq_indices,
                )  # one batch
                if sub_tts_speech.size == 0:
                    break
                yield np.frombuffer(sub_tts_speech, dtype=float).tobytes()
                if pre_sub_tts_speech_size > sub_tts_speech.size:  # for llamacpp quants cases
                    logging.info(f"break by empty wav size: {sub_tts_speech.size}")
                    break
                pre_sub_tts_speech_size = sub_tts_speech.size
                semantic_token_ids = semantic_token_ids[chunk_size - token_overlap_len :]
                # increase token_hop_len for better speech quality
                chunk_size = min(max_chunk_size, int(chunk_size * stream_scale_factor))
                logging.info(
                    f"increase chunk_size: {chunk_size} token_overlap_len:{token_overlap_len}"
                )

        if len(semantic_token_ids) > 0:  # end to finalize
            # Process each batch
            sub_tts_speech = self.token2wav(
                [controll_gen_global_token_ids + semantic_token_ids],
                gender,
                global_fsq_indices,
            )  # one batch
            yield np.frombuffer(sub_tts_speech, dtype=float).tobytes()
            logging.info(f"last batch len: {len(semantic_token_ids)}")

        torch.cuda.empty_cache()
