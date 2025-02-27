import logging
import os
import sys
from threading import Lock
from typing import AsyncGenerator, Generator
import uuid

import torch
from transformers import WhisperFeatureExtractor, AutoTokenizer
from apipeline.frames import *

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../StepAudio"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../deps/StepAudio"))

    from deps.StepAudio.tts import StepAudioTTS
    from deps.StepAudio.tokenizer import StepAudioTokenizer
    from deps.StepAudio.utils import load_audio, speech_adjust, volumn_adjust
except ModuleNotFoundError as e:
    raise Exception(f"Missing module: {e}")

from src.core.llm.transformers.manual_voice_step import TransformersManualVoiceStep
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.types.llm.transformers import TransformersLMArgs
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import *


class StepVoiceBaseProcessor(VoiceProcessorBase):
    """
    use Speech-Tokenizer (linguistic(funasr Paraformer)+semantic(CosyVoice tokenizer)) +  Step1-Audio-LM(Step-Audio-Chat)(text/audio) + Voice-Decoder (Step-Audio-TTS-3B)
    - T1A1-T2A2: (text/audio)-to-(tokens) (Step-Audio-Chat(text)/Speech-Tokenizer(audio)) -> Step1-Audio-LM(Step-Audio-Chat)(130B) -- text|speech tokens --> Step-Audio-Chat(text decoder)|CosyVoice(speech decoder(mel->waveform))

    Model Architecture:
    - Speech-Tokenizer: linguistic(funasr Paraformer)+semantic(CosyVoice tokenizer)
    - Step1-Audio-LM(Step-Audio-Chat): 130B
    - Speech Decoder: Step-Audio-TTS
    """

    def __init__(
        self,
        *,
        voice_in_args: GLMVoiceInArgs | dict = GLMVoiceInArgs(),
        lm_gen_args: dict = {},
        voice_out_args: GLMVoiceOutArgs | dict = GLMVoiceOutArgs(),
        system_prompt: str = "",
        voice_tokenizer_path: str | None = None,  # audio encoder/ft extractor
        voice_decoder_path: str | None = None,  # audio decoder
        device: str = "cuda",
        torch_dtype: str = "auto",  # auto,float16,bfloat16,float32
        session: Session | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice_in_args = voice_in_args
        if isinstance(voice_in_args, dict):
            self._voice_in_args = GLMVoiceInArgs(**voice_in_args)
        self._lm_gen_args = lm_gen_args
        self._voice_out_args = voice_out_args
        if isinstance(voice_out_args, dict):
            self._voice_out_args = GLMVoiceOutArgs(**voice_out_args)

        self._sys_prompt = system_prompt or TransformersManualVoiceStep.DEFAULT_SYS_PROMPT
        self._voice_tokenizer_path = voice_tokenizer_path
        self._voice_decoder_path = voice_decoder_path
        self._torch_dtype = torch_dtype
        self._device = device

        # now just support a single session
        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

        self.reset()
        self.load_models()

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": self._voice_out_args.audio_sample_rate,
            "channels": self._voice_out_args.audio_channels,
        }

    def reset(self):
        pass

    def load_models(self):
        logging.info("loading model weights")

        # Speech tokenizer (linguistic, semantic audio encoder)/feature_extractor
        self.encoder = StepAudioTokenizer(self._voice_tokenizer_path)
        logging.info("speech vq encoder and feature_extractor model state weight load")

        # Flow & Hift decoder with config, fixed sample rate 22050
        self._audio_decoder = StepAudioTTS(
            self._voice_decoder_path, self.encoder, max_stream_factor=4
        )
        logging.info("speech audio Flow & Hift decoder model state weight load")

        # gen lm
        self._glm_model = TransformersManualVoiceStep(**self._lm_gen_args)
        self._glm_model.warmup()
        logging.info("gen lm model state weight load and warnup")

        logging.info("model weights loaded")

    def apply_chat_template(self, messages: list):
        # TODO: function call
        text_with_audio = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio += f"<|BOT|>{role}\n{content}<|EOT|>"
            elif isinstance(content, dict):
                if content["type"] == "text":
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content["type"] == "audio":
                    if "audio" in content:
                        audio_tokens = self.encode_audio(content["audio"])
                        text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
                    elif "audio_tokens" in content:
                        text_with_audio += f"<|BOT|>{role}\n{content['audio_tokens']}<|EOT|>"
            elif content is None:
                text_with_audio += f"<|BOT|>{role}\n"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        if not text_with_audio.endswith("<|BOT|>assistant\n"):
            text_with_audio += "<|BOT|>assistant\n"
        return text_with_audio

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._create_push_task()

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    def encode_audio(self, audio_path):
        audio_wav, sr = load_audio(audio_path)
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio_tokens = self.encode_audio(frame.path)
        else:
            audio_wav_tensor = bytes2TorchTensorWith16(frame.audio)
            audio_tokens = self.encoder(audio_wav_tensor, self._voice_in_args.audio_sample_rate)
        if len(audio_tokens) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return
        message = {
            "role": "user",
            "content": {"type": "audio", "audio_tokens": audio_tokens},
        }
        self._session.chat_history.append(message)
        text_with_audio = self.apply_chat_template(self._session.chat_history)

        self._session.ctx.state["prompt"] = text_with_audio
        iter_tokens = self._glm_model.generate(self._session)
        async for text in self.tokens_decode_out(iter_tokens):
            yield text

    @torch.no_grad()
    async def tokens_decode_out(self, iter_tokens):
        # Note: now just support decode text -> text frame
        # TODO: audio token decode -> flow -> hift -> audio frame
        texts = ""
        for token_id in iter_tokens:
            complete_text = self._glm_model._tokenizer.decode(
                [token_id], spaces_between_special_tokens=False
            )
            yield complete_text
            texts += complete_text
        self._session.chat_history.append({"role": "assistant", "content": texts})


class StepAudioVoiceProcessor(StepVoiceBaseProcessor):
    """
    use Speech-Tokenizer (linguistic(funasr Paraformer)+semantic(CosyVoice tokenizer)) +  Step1-Audio-LM(Step-Audio-Chat)(text/audio) + Voice-Decoder (Step-Audio-TTS-3B)
    - A1-T2A2: (text/audio)-to-(tokens) (Step-Audio-Chat(text)/Speech-Tokenizer(audio)) -> Step1-Audio-LM(Step-Audio-Chat)(130B) -- text|speech tokens --> Step-Audio-Chat(text decoder)|CosyVoice(speech decoder(mel->waveform))
    """

    pass


class StepTextVoiceProcessor(StepVoiceBaseProcessor):
    """
    use Text-Tokenizer (Step1-Audio-LM tokenizer)  +  Step1-Audio-LM(Step-Audio-Chat)(text/audio) + Voice-Decoder (Step-Audio-TTS-3B)
    - T1-T2A2: text-to-(tokens) (Step-Audio-Chat(text)/Speech-Tokenizer(audio)) -> Step1-Audio-LM(Step-Audio-Chat)(130B) -- text|speech tokens --> Step-Audio-Chat(text decoder)|CosyVoice(speech decoder(mel->waveform))
    """

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        # history
        message = {
            "role": "user",
            "content": {"type": "text", "audio_tokens": user_input},
        }
        self._session.chat_history.append(message)
        text_with_audio = self.apply_chat_template(self._session.chat_history)

        self._session.ctx.state["prompt"] = text_with_audio
        iter_tokens = self._glm_model.generate(self._session)
        async for text in self.tokens_decode_out(iter_tokens):
            yield text


class MockStepAudioVoiceProcessor(VoiceProcessorBase):
    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        yield TextFrame(text="hello, 语音, 你好!")


class MockStepTextVoiceProcessor(VoiceProcessorBase):
    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        yield TextFrame(text="hello, 文本, 你好!")
