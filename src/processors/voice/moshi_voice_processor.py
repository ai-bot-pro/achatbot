import asyncio
import logging
import random
import time
from typing import AsyncGenerator

from apipeline.frames import Frame, ErrorFrame, StartFrame, EndFrame, CancelFrame, AudioRawFrame, TextFrame
from moshi.models.loaders import DEFAULT_REPO
import numpy as np
import torch

from common.utils.audio_utils import bytes2NpArrayWith16, postprocess_tts_wave_int16
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import LMGenArgs

try:
    from sphn import OpusStreamReader, OpusStreamWriter, resample
    from sentencepiece import SentencePieceProcessor
    from moshi.models import loaders, MimiModel, LMModel, LMGen
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use moshi, you need to `pip install achatbot[moshi_voice_processor]`")
    raise Exception(f"Missing module: {e}")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class MoshiVoiceBaseProcessor(VoiceProcessorBase):
    """
    use mimi speech codec + moshi lm(moshiko/moshika)
    - A1-T2A2: (speech)-to-(speech and text) (mimi(speech encoder)) -> (llm) -- text|speech tokens --> bpe(text decoder)|mimi(speech decoder))

    NOTE:
    - moshi genative lm no system prompt and funciton call.
    """

    def __init__(self,
                 *,
                 lm_gen_args: LMGenArgs | dict = LMGenArgs(),
                 model_name: str = loaders.DEFAULT_REPO,
                 mimi_weight_file: str | None = None,
                 text_tokenizer_file: str | None = None,
                 moshi_weight_file: str | None = None,
                 device: str = "cuda",
                 **kwargs):
        super().__init__(**kwargs)
        seed_all(12345678)
        self._lm_gen_args = lm_gen_args
        if isinstance(lm_gen_args, dict):
            self._lm_gen_args = LMGenArgs(**lm_gen_args)
        self._model_name = model_name
        self._mimi_weight_file = mimi_weight_file
        self._text_tokenizer_file = text_tokenizer_file
        self._moshi_weight_file = moshi_weight_file
        self._device = device

        self._mimi: MimiModel = None
        self._text_tokenizer: SentencePieceProcessor = None
        self._moshi_lm: LMModel = None
        self._lm_gen: LMGen = None

        self.load_models()

        # open streaming with batch_size:1,
        # if not, need use with lm_gen.streaming(1):
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)
        self.warmup()

        self._cur_in_sample_rate = 0

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": self._mimi.sample_rate,
            "channels": self._mimi.channels,
        }

    def load_models(self):
        logging.info("loading mimi")
        if self._mimi_weight_file is None:
            self._mimi_weight_file = hf_hub_download(self._model_name, loaders.MIMI_NAME)
        self._mimi = loaders.get_mimi(self._mimi_weight_file, self._device)
        self._mimi.set_num_codebooks(8)
        self._frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        logging.info("mimi loaded")

        logging.info("loading text tokenizer")
        if self._text_tokenizer_file is None:
            self._text_tokenizer_file = hf_hub_download(
                self._model_name, loaders.TEXT_TOKENIZER_NAME)
        self._text_tokenizer = SentencePieceProcessor(self._text_tokenizer_file)
        logging.info("loaded text tokenizer")

        logging.info("loading moshi")
        if self._moshi_weight_file is None:
            self._moshi_weight_file = hf_hub_download(self._model_name, loaders.MOSHI_NAME)
        self._moshi_lm = loaders.get_moshi_lm(self._moshi_weight_file, self._device)
        self._lm_gen = LMGen(
            self._moshi_lm,
            **self._lm_gen_args.__dict__,
        )
        logging.info("moshi loaded")

    def warmup(self):
        logging.info("start warmup")
        for i in range(4):
            be = time.time()
            chunk = torch.zeros(1, 1, self._frame_size, dtype=torch.float32, device=self._device)
            codes = self._mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self._mimi.decode(tokens[:, 1:])
            logging.info(f"chunk warmup {i} in {1000 * (time.time() - be):.1f}ms")
        torch.cuda.synchronize()
        logging.info("end warmup")

    def resample(self, pcm):
        if self._cur_in_sample_rate != self._mimi.sample_rate:
            resample(pcm, self._cur_in_sample_rate, self._mimi.sample_rate)


class MoshiVoiceOpusStreamProcessor(MoshiVoiceBaseProcessor):
    """
    use mimi speech codec + moshi lm(moshiko/moshika) with opus w/r streamer
    """

    def __init__(
            self,
            *,
            lm_gen_args: LMGenArgs | dict = LMGenArgs(),
            model_name: str = loaders.DEFAULT_REPO,
            mimi_weight_file: str | None = None,
            text_tokenizer_file: str | None = None,
            moshi_weight_file: str | None = None,
            device: str = "cuda",
            **kwargs):
        super().__init__(
            lm_gen_args=lm_gen_args,
            model_name=model_name,
            mimi_weight_file=mimi_weight_file,
            text_tokenizer_file=text_tokenizer_file,
            moshi_weight_file=moshi_weight_file,
            device=device,
            **kwargs)

        self._opus_writer: OpusStreamWriter = None
        self._opus_reader: OpusStreamReader = None

        self._audio_in_task = None
        self._audio_out_task = None

    def reset_state(self):
        # https://opus-codec.org/
        # https://github.com/kyutai-labs/sphn (python binds rust lib so)
        # Easily load various audio file formats (bytes) into numpy arrays.
        # Read/write ogg/opus audio files with streaming support.
        # use Opus format for audio across the websocket,
        # as it can be safely streamed and decoded in real-time
        self._opus_writer = OpusStreamWriter(self._mimi.sample_rate)
        # NOTE:
        # new stream reader thread,
        # in thread event loop have block,
        # https://github.com/kyutai-labs/sphn/blob/main/src/opus.rs#L345
        self._opus_reader = OpusStreamReader(self._mimi.sample_rate)

        # LLM is stateful, maintaining chat history,
        # so reset it on each connection
        self._mimi.reset_streaming()
        self._lm_gen.reset_streaming()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self.reset_state()

        self._audio_in_task = self.get_event_loop().create_task(self._audio_in_task_handler())
        self._audio_out_task = self.get_event_loop().create_task(self._audio_out_task_handler())

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        if self._audio_in_task:
            self._audio_in_task.cancel()
            await self._audio_in_task
        if self._audio_out_task:
            self._audio_out_task.cancel()
            await self._audio_out_task

        self._opus_reader.close()
        self.reset_state()

        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    async def _audio_in_task_handler(self):
        while True:
            try:
                pcm = self._opus_reader.read_pcm()
                pcm = self.resample(pcm)
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                await self.start_processing_metrics()
                ttfb_metric = True
                while all_pcm_data.shape[-1] >= self._frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self._frame_size]
                    all_pcm_data = all_pcm_data[self._frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self._device)[None, None]
                    codes = self._mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        ttfb_metric and await self.start_ttfb_metrics()
                        tokens = self._lm_gen.step(codes[:, :, c: c + 1])
                        ttfb_metric and await self.stop_ttfb_metrics()
                        ttfb_metric = False
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self._lm_gen.lm_model.dep_q + 1
                        main_pcm = self._mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        self._opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self._text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            logging.info(f"text token '{_text}'")
                            await self.queue_frame(TextFrame(text=_text))
                    logging.info(f"frame handled in {1000 * (time.time() - be):.1f}ms")
                await self.stop_processing_metrics()
            except asyncio.CancelledError:
                break

    async def _audio_out_task_handler(self):
        while True:
            try:
                await asyncio.sleep(0.001)
                audio_bytes = self._opus_writer.read_bytes()
                if len(audio_bytes) > 0:
                    await self.queue_frame(AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=self._mimi.sample_rate,
                        num_channels=self._mimi.channels,
                    ))
            except asyncio.CancelledError:
                break

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        """
        StreamReader sample_rate: 48000, 24000
        https://github.com/kyutai-labs/sphn/blob/main/src/opus.rs#L337
        """
        self._cur_in_sample_rate = frame.sample_rate
        self._opus_reader.append_bytes(frame.audio)
        yield None


class MoshiVoiceProcessor(MoshiVoiceBaseProcessor):

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        """
        StreamReader sample_rate: 48000, 24000
        https://github.com/kyutai-labs/sphn/blob/main/src/opus.rs#L337
        """
        self._cur_in_sample_rate = frame.sample_rate
        pcm = bytes2NpArrayWith16(frame.audio)
        pcm = self.resample(pcm)
        if pcm.shape[-1] == 0:
            yield None
            return
        if all_pcm_data is None:
            all_pcm_data = pcm
        else:
            all_pcm_data = np.concatenate((all_pcm_data, pcm))
        await self.start_processing_metrics()
        ttfb_metric = True
        while all_pcm_data.shape[-1] >= self._frame_size:
            be = time.time()
            chunk = all_pcm_data[: self._frame_size]
            all_pcm_data = all_pcm_data[self._frame_size:]
            chunk = torch.from_numpy(chunk)
            chunk = chunk.to(device=self._device)[None, None]
            codes = self._mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                ttfb_metric and await self.start_ttfb_metrics()
                tokens = self._lm_gen.step(codes[:, :, c: c + 1])
                ttfb_metric and await self.stop_ttfb_metrics()
                ttfb_metric = False
                if tokens is None:
                    continue
                assert tokens.shape[1] == self._lm_gen.lm_model.dep_q + 1
                main_pcm = self._mimi.decode(tokens[:, 1:])
                main_pcm = main_pcm.cpu()
                audio_bytes = postprocess_tts_wave_int16(main_pcm[0, 0])
                yield AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self._mimi.sample_rate,
                    num_channels=self._mimi.channels,
                )
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = self._text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    logging.info(f"text token '{_text}'")
                    yield TextFrame(text=_text)
            logging.info(f"frame handled in {1000 * (time.time() - be):.1f}ms")
        await self.stop_processing_metrics()