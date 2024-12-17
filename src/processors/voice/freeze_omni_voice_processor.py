from copy import deepcopy
import os
import logging
import sys
from typing import AsyncGenerator
import uuid

import torch
from apipeline.frames import *
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor


from deps.FreezeOmni.bin.inference import audioEncoderProcessor
from deps.FreezeOmni.models.decoder.llm2tts import llm2TTS
from deps.FreezeOmni.models.pipeline import inferencePipeline

from src.common.pool import ClassObjectPool, OneClassObjectPool
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import *

DEFAULT_SYS_PROMPT = "You are a helpful voice assistant.\
Your answer should be coherent, natural, simple, complete.\
Your name is QQ.\
Your inventor is Tencent."


@dataclass
class FreezeOmniVoiceBaseProcessorArgs:
    system_prompt: str = DEFAULT_SYS_PROMPT  # text llm ssystem prompt
    llm_path: str | None = None  # text llm path
    # text llm sampling params
    top_k: int = 20
    top_p: float = 0.8
    temperature: float = 0.8

    # obj pool params
    max_users: int = 1
    llm_exec_nums: int = 1

    # decoder(LLM2TTSCodecAR)
    # NAR llama transformer decoder pre_nn_forward -> NAR llama transformer decoder kv_cache_prefix_forward -> AR llama transformer decoder transformer_infer
    model_path: str | None = None  # audio-llm decoder and codec(decoder) ckpt path
    # llama transformer decoder
    top_k: int = 2  # The number of top-k tokens to consider during inference.
    penalty_window_size: int = 20  # The window size for applying penalties during decoding.
    penalty: float = 1.1  # The penalty factor.
    max_tokens: int = 1000  # The maximum number of tokens to generate.
    # codec decoder
    codec_chunk_size: int = 40  # The size of each chunk to process in the codec model.
    codec_padding_size: int = 10  # The amount of padding to add on each side of the codec chunk.
    # find_min_sum_index
    N: int = 2401  # The size of the sliding window used to calculate the sum.
    seg_threshold: float = 0.01  # Threshold value to determine whether to concatenate buffer and current segment or not


class FreezeOmniVoiceBaseProcessor(VoiceProcessorBase):
    """ """

    def __init__(
        self,
        *,
        args: FreezeOmniVoiceBaseProcessorArgs,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cur_dir = os.path.dirname(__file__)
        if bool(os.getenv("ACHATBOT_PKG", "")):
            sys.path.insert(1, os.path.join(cur_dir, "../../FreezeOmni"))
        else:
            sys.path.insert(1, os.path.join(cur_dir, "../../../deps/FreezeOmni"))

        self._args = args
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

        self.reset()
        self.load_models()

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": 24000,
            "channels": 1,
        }

    def reset(self):
        # input_texts + completion_texts (audio tokenid special tag)
        self._history_texts = ""
        self._stat = ""

    def load_models(self):
        logging.info("loading model weights")
        # stream chunk to encoder
        self.audio_processor = audioEncoderProcessor()
        self.chunk_size = self.audio_processor.get_chunk_size()
        # encoder, adpter and text llm
        self.inference_pipeline_pool = ClassObjectPool(
            size=self._args.llm_exec_nums,
            cls=inferencePipeline,
            multi_thread_init=False,
            configs=self._args,
        )
        # NAR AR decoder and codec decode (vq-vae)
        self.tts_pool = OneClassObjectPool(
            size=self._args.max_users,
            cls=llm2TTS,
            multi_thread_init=True,
            model_path=self._args.model_path,
        )

        logging.info("model weights loaded")

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._create_push_task()
        self.inference_pipeline_obj = self.inference_pipeline_pool.acquire()
        self.inference_pipeline: inferencePipeline = self.inference_pipeline_obj.obj

        self.tts_obj = self.tts_pool.acquire()
        self.llm2tts: llm2TTS = self.tts_obj.obj

        # init default prompt
        self._stat = "pre"
        self.init_outputs = self.inference_pipeline.speech_dialogue(
            None, stat=self._stat, role=DEFAULT_SYS_PROMPT
        )
        # init_outputs dict have stat, if stat change in the out, no only_read need deep copy
        # self.system_role = deepcopy(self.init_outputs)

        self.tts_pool.print_info()
        self.pipeline_pool.print_info()

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self.tts_pool.release(self.tts_obj)
        self.llm2tts = None
        self.inference_pipeline_pool.release(self.inference_pipeline_obj)
        self.inference_pipeline = None
        self.tts_pool.print_info()
        self.pipeline_pool.print_info()
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self.tts_pool.release(self.tts_obj)
        self.llm2tts = None
        self.inference_pipeline_pool.release(self.inference_pipeline_obj)
        self.inference_pipeline = None
        self.tts_pool.print_info()
        self.pipeline_pool.print_info()
        logging.info("cancel done")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        if isinstance(frame, UserStartedSpeakingFrame):
            self._stat = "sl"  # start listenning
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._stat = "el"  # end listenning

        await super().process_frame(frame, direction)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            utt = frame.path
        else:
            audio_tensor = bytes2TorchTensorWith16(frame.audio)
            utt = (audio_tensor, self._voice_in_args.audio_sample_rate)

        # TODO: generate audio

        yield None
