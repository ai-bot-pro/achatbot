# python 2 -> python3 print
# from __future__ import print_function

import asyncio
import builtins
import datetime
import uuid
import math
import logging
from typing import AsyncGenerator, Generator

import torch
import torchaudio
from apipeline.frames import *
from apipeline.processors.frame_processor import FrameDirection

from deps.FreezeOmni.bin.inference import audioEncoderProcessor
from deps.FreezeOmni.models.decoder.llm2tts import llm2TTS
from deps.FreezeOmni.models.pipeline import inferencePipeline

from src.common.pool import ClassObjectPool, OneClassObjectPool
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16, postprocess_tts_wave_int16
from src.types.frames import *

DEFAULT_SYS_PROMPT = "You are a helpful voice assistant.\
Your answer should be coherent, natural, simple, complete.\
Your name is QQ.\
Your inventor is Tencent."


def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    original_print(f"[{current_time}] [{__name__}]", *args, **kwargs)


# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print


@dataclass
class FreezeOmniVoiceProcessorArgs:
    system_prompt: str = DEFAULT_SYS_PROMPT  # text llm ssystem prompt
    llm_path: str | None = None  # text llm path
    # text llm sampling params
    top_k: int = 20
    top_p: float = 0.8
    temperature: float = 0.8

    # obj pool params
    max_use_nums: int = 1
    llm_exec_nums: int = 1

    # decoder(LLM2TTSCodecAR)
    # NAR llama transformer decoder pre_nn_forward -> NAR llama transformer decoder kv_cache_prefix_forward -> AR llama transformer decoder transformer_infer
    model_path: str | None = None  # audio-llm decoder and codec(decoder) ckpt path
    # llama transformer decoder
    decoder_max_tokens = 1000
    decoder_top_k: int = 2  # The number of top-k tokens to consider during inference.
    decoder_penalty_window_size: int = 20  # The window size for applying penalties during decoding.
    decoder_penalty: float = 1.1  # The penalty factor.
    decoder_max_tokens: int = 1000  # The maximum number of tokens to generate.
    # codec decoder
    codec_chunk_size: int = 40  # The size of each chunk to process in the codec model.
    codec_padding_size: int = 10  # The amount of padding to add on each side of the codec chunk.
    # find_min_sum_index
    decoder_N: int = 2401  # The size of the sliding window used to calculate the sum.
    decoder_seg_threshold: float = 0.01  # Threshold value to determine whether to concatenate buffer and current segment or not

    # control params
    max_past_tokens = 512  # ss max past token to stop speak


class FreezeOmniVoiceObjPool:
    """
    create obj pool
    """

    @staticmethod
    def create_inference_pipeline_pool(args: FreezeOmniVoiceProcessorArgs):
        # encoder, adpter and text llm
        return ClassObjectPool(
            size=args.llm_exec_nums,
            cls=inferencePipeline,
            multi_thread_init=False,
            args=args,
        )

    @staticmethod
    def create_tts_pool(args: FreezeOmniVoiceProcessorArgs):
        # NAR AR decoder and codec decode (vq-vae)
        return OneClassObjectPool(
            size=args.max_use_nums,
            cls=llm2TTS,
            multi_thread_init=True,
            model_path=args.model_path,
        )


class FreezeOmniVoiceProcessor(VoiceProcessorBase):
    """ """

    def __init__(
        self,
        *,
        args: FreezeOmniVoiceProcessorArgs | dict = FreezeOmniVoiceProcessorArgs(),
        inference_pipeline_pool: ClassObjectPool | None = None,
        tts_pool: OneClassObjectPool | None = None,
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._args = args
        if isinstance(args, dict):
            self._args = FreezeOmniVoiceProcessorArgs(**args)
        self.inference_pipeline_pool = inference_pipeline_pool
        self.tts_pool = tts_pool
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
        if self.inference_pipeline_pool is None:
            self.inference_pipeline_pool = FreezeOmniVoiceObjPool.create_inference_pipeline_pool(
                self._args
            )
        # NAR AR decoder and codec decode (vq-vae)
        if self.tts_pool is None:
            self.tts_pool = FreezeOmniVoiceObjPool.create_tts_pool(self._args)

        logging.info("model weights loaded")

    async def start(self, frame: StartFrame):
        await super().start(frame)

        # self._create_push_task()
        self.inference_pipeline_obj = self.inference_pipeline_pool.acquire()
        self.inference_pipeline: inferencePipeline = self.inference_pipeline_obj.obj

        self.tts_obj = self.tts_pool.acquire()
        self.llm2tts: llm2TTS = self.tts_obj.obj

        # Satge0: preprocess, init default prompt
        # set system role, stat will be set to 'sl'
        self._stat = "pre"
        self._outputs = self.inference_pipeline.speech_dialogue(
            None, stat=self._stat, role=DEFAULT_SYS_PROMPT
        )
        # init_outputs dict have stat, if stat change in the out, no only_read need deep copy
        # self.system_role = deepcopy(self.init_outputs)

        self.tts_pool.print_info()
        self.inference_pipeline_pool.print_info()

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self.tts_pool.release(self.tts_obj)
        self.llm2tts = None
        self.inference_pipeline_pool.release(self.inference_pipeline_obj)
        self.inference_pipeline = None
        self.tts_pool.print_info()
        self.inference_pipeline_pool.print_info()
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self.tts_pool.release(self.tts_obj)
        self.llm2tts = None
        self.inference_pipeline_pool.release(self.inference_pipeline_obj)
        self.inference_pipeline = None
        self.tts_pool.print_info()
        self.inference_pipeline_pool.print_info()
        logging.info("cancel done")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        if isinstance(frame, UserStartedSpeakingFrame):
            self._stat = "sl"  # start listenning
        if isinstance(frame, AudioRawFrame) and self._stat == "sl":
            self._stat = "cl"  # continue listenning
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._stat = "el"  # end listenning

        await super().process_frame(frame, direction)

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_tensor, sample_rate = torchaudio.load(frame.path)
            if sample_rate != 16000:
                audio_tensor = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )(audio_tensor)
                sample_rate = 16000
        else:
            audio_tensor = bytes2TorchTensorWith16(frame.audio)
        audio_tensor = audio_tensor.reshape(-1)  # [1,len] -> [len]
        if self._stat == "cl" or self._stat == "el":
            # - cl: continue audio to process (NOTE: need audio size >= chunk_size)
            # - el: aggr sl - el  audio to process

            # Satge1: start listen
            # stat will be auto set 'sl' to 'cl' after Stage1
            wav_input = torch.zeros(
                math.ceil(audio_tensor.shape[0] / self.chunk_size) * self.chunk_size
            )
            wav_input[: audio_tensor.shape[0]] = audio_tensor
            for i in range(0, wav_input.shape[0], self.chunk_size):
                if "text" in self._outputs:
                    del self._outputs["text"]
                if "hidden_state" in self._outputs:
                    del self._outputs["hidden_state"]
                fbank = self.audio_processor.process(wav_input[i : i + self.chunk_size])
                self._outputs = self.inference_pipeline.speech_dialogue(fbank, **self._outputs)
                logging.info(f"speech_dialogue self._outputs stat: {self._outputs['stat']}")
                if self._outputs["stat"] == "ss":
                    break
                self._outputs["stat"] = "cl"

        if self._stat == "el":  # end listen -> start speak
            self.audio_processor.reset()

            logging.info(f"outputs keys: {self._outputs.keys()}")
            self._outputs["adapter_cache"] = None
            self._outputs["encoder_cache"] = None
            self._outputs["pe_index"] = 0
            self._outputs["last_id"] = None
            self._outputs["stat"] = "ss"

        if self._outputs["stat"] == "ss":
            async for item in self.start_speak():
                yield item

    async def start_speak(self):
        """
        audioLLM(asr) recognize -> start speak
        """
        if "text" in self._outputs:
            del self._outputs["text"]
        if "hidden_state" in self._outputs:
            del self._outputs["hidden_state"]
        # Stage3: start speak
        self._outputs = self.inference_pipeline.speech_dialogue(None, **self._outputs)
        cur_hidden_state = []
        cur_hidden_state.append(self._outputs["hidden_state"])
        await self.push_frame(BotStartedSpeakingFrame())

        whole_text = ""
        last_text = ""
        cur_text = ""
        # Stage4: contiune speak until stat is set to 'sl'
        # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
        stop = False
        while True:
            await self.push_frame(BotSpeakingFrame())
            if len(self._outputs["past_tokens"]) > self._args.max_past_tokens:
                logging.info(f"trigger max_past_tokens:{self._args.max_past_tokens} to stop tts")
                stop = True
            if stop:
                break
            del self._outputs["text"]
            del self._outputs["hidden_state"]
            # https://superfastpython.com/asyncio-to_thread/
            self._outputs = await asyncio.to_thread(
                self.inference_pipeline.speech_dialogue, None, **self._outputs
            )
            if self._outputs["stat"] == "cs":  # continue speak
                cur_hidden_state.append(self._outputs["hidden_state"])
                whole_text += self._outputs["text"][len(last_text) :]
                cur_text += self._outputs["text"][len(last_text) :]
                suffix_list = ["。", "：", "？", "！", ".", "?", "!", "\n"]
                if self._outputs["text"][len(last_text) :].endswith(tuple(suffix_list)):
                    if (
                        self._outputs["text"][len(last_text) :].endswith(".")
                        and last_text[-1].isdigit()
                    ):
                        pass
                    else:
                        if len(cur_hidden_state) > 0:
                            await self.push_frame(TextFrame(text=cur_text))
                            segs = await asyncio.to_thread(
                                self.decoder,
                                cur_hidden_state,
                                cur_text,
                            )
                            for item in segs:
                                # audio_bytes = postprocess_tts_wave_int16(item)
                                audio_bytes = (
                                    (item.squeeze().float().cpu().numpy() * 32768)
                                    .astype("int16")
                                    .tobytes()
                                )
                                logging.info(
                                    f"seg tensor:{item.shape},push audio len:{len(audio_bytes)}"
                                )
                                audio_frame = AudioRawFrame(audio=audio_bytes, sample_rate=24000)
                                await self.queue_frame(audio_frame)
                                yield None
                                # yield audio_frame
                                # await asyncio.sleep(0.01)
                            cur_hidden_state = []
                        cur_text = ""
            if self._outputs["stat"] == "sl":
                break
            # print(self._outputs['text'])
            last_text = self._outputs["text"]

        if len(cur_hidden_state) != 0:
            await self.push_frame(TextFrame(text=cur_text))
            segs = await asyncio.to_thread(self.decoder, cur_hidden_state, cur_text)
            for item in segs:
                # audio_bytes = postprocess_tts_wave_int16(item)
                audio_bytes = (
                    (item.squeeze().float().cpu().numpy() * 32768).astype("int16").tobytes()
                )
                logging.info(f"seg tensor:{item.shape},push audio len:{len(audio_bytes)}")
                audio_frame = AudioRawFrame(audio=audio_bytes, sample_rate=24000)
                await self.queue_frame(AudioRawFrame(audio=audio_bytes, sample_rate=24000))
                yield None
                # yield audio_frame
                # await asyncio.sleep(0.01)

        # one turn conversation over, set stat is "sl"
        await self.push_frame(BotStoppedSpeakingFrame())
        self._outputs["stat"] = "sl"
        self._outputs["last_id"] = None

    def decoder(self, cur_hidden_state, cur_text) -> list[torch.Tensor]:
        segs = []
        for seg in self.decoder_iter(cur_hidden_state, cur_text):
            segs.append(seg)
        return segs

    def decoder_iter(self, cur_hidden_state, cur_text) -> Generator[torch.Tensor, None, None]:
        """
        Decodes the current hidden state and text to generate audio segments using speech decoder.
        """
        hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
        cur_text_procced = self.inference_pipeline.post_process(cur_text)
        logging.info(f"Synthesis: {cur_text_procced}")
        embeddings = self.inference_pipeline.model.llm_decoder.model.embed_tokens(
            torch.tensor(self.inference_pipeline.model.tokenizer.encode(cur_text_procced)).cuda()
        )

        for seg in self.llm2tts.run(
            embeddings.reshape(-1, 896).unsqueeze(0),
            self._args.decoder_top_k,
            hidden_state_output.reshape(-1, 896).unsqueeze(0),
            codec_chunk_size=self._args.codec_chunk_size,
            codec_padding_size=self._args.codec_padding_size,
            penalty_window_size=self._args.decoder_penalty_window_size,
            penalty=self._args.decoder_penalty,
            N=self._args.decoder_N,
            seg_threshold=self._args.decoder_seg_threshold,
            max_tokens=self._args.decoder_max_tokens,
        ):
            yield seg
