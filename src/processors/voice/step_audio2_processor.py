import os
import sys
import time
import logging
from typing import AsyncGenerator

import uuid
import torch
from apipeline.frames import *

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../StepAudio2"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../deps/StepAudio2"))

    from deps.StepAudio2.token2wav import Token2wav
except ModuleNotFoundError as e:
    raise Exception(f"Missing module: {e}")

from src.common.interface import ILlm
from src.common.types import ASSETS_DIR
from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import TextQuestionsAudioRawFrame, PathAudioRawFrame


class StepAudio2BaseProcessor(VoiceProcessorBase):
    """ """

    CHUNK_SIZE = 25
    SYS_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        *,
        init_system_prompt: str = "",
        text_stream_out: bool = False,
        prompt_wav: str = "",
        warmup_cn: int = 1,
        chat_history_size: int | None = None,
        session: Session | None = None,
        audio_llm: ILlm | None = None,
        token2wav: Token2wav | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert audio_llm is not None, "audio_llm is None"

        from src.core.llm.transformers.manual_voice_step2 import TransformersManualVoiceStep2

        assert isinstance(audio_llm, TransformersManualVoiceStep2), (
            "audio_llm is not TransformersManualVoiceStep2"
        )

        self._audio_llm = audio_llm
        token2wav_path = os.path.join(audio_llm.args.lm_model_name_or_path, "token2wav")
        self._token2wav = Token2wav(token2wav_path)
        if torch.cuda.is_available():
            self._token2wav.flow.scatter_cuda_graph(True)

        self._prompt_wav = prompt_wav or os.path.join(ASSETS_DIR, "default_female.wav")
        self._token2wav.set_stream_cache(self._prompt_wav)
        if warmup_cn > 0:
            for i in range(warmup_cn):
                start = time.time()
                self._token2wav.warmup(self._prompt_wav)
                logging.info(f"Token2wav warmup {i=} done in {time.time() - start:.3f}s")

        self._system_prompt = init_system_prompt or self.SYS_PROMPT
        self._text_stream_out = text_stream_out

        self._session = session or Session(
            chat_history_size=chat_history_size, **SessionCtx(str(uuid.uuid4())).__dict__
        )
        self._session.chat_history.init({"role": "system", "content": self._system_prompt})

    def reset(self):
        self._token2wav.cache = {}
        self._session.reset()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": 24000,
            "channels": 1,
        }

    async def say(
        self,
        text: str,
        is_out_text: bool = True,
        system_prompt: str = "以自然的语速读出下面的文字。\n",
    ) -> None:
        """
        support: en,zh,ja
        """
        logging.info(f"say: {text}")

        messages = [
            {"role": "system", "content": system_prompt or self._system_prompt},
            {"role": "human", "content": text},
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        ]
        self._session.ctx.state["messages"] = messages
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        await self.process_out_audio_text(token_iter, is_out_text)

    async def process_out_audio_text(self, token_iter, is_out_text: bool = True):
        output_tokens = []
        output_audio_tokens = []
        out_text_tokens = []
        buffer = []
        for token_id in token_iter:
            if token_id in self._audio_llm.eos_token_id:
                break
            output_tokens.append(token_id)
            if token_id < 151688:  # text
                out_text_tokens.append(token_id)
                if is_out_text is True and self._text_stream_out is True:
                    out_text = self._audio_llm.llm_tokenizer.decode(token_id)
                    await self.queue_frame(TextFrame(text=out_text))
            if token_id > 151695:  # audio
                audio_token_id = token_id - 151696
                if audio_token_id < 6561:  # remove audio padding
                    output_audio_tokens.append(audio_token_id)
                    buffer.append(audio_token_id)
                    if len(buffer) >= self.CHUNK_SIZE + self._token2wav.flow.pre_lookahead_len:
                        out_bytes = self._token2wav.stream(
                            buffer[: self.CHUNK_SIZE + self._token2wav.flow.pre_lookahead_len],
                            prompt_wav=self._prompt_wav,
                            last_chunk=False,
                        )
                        await self.queue_frame(
                            AudioRawFrame(
                                audio=out_bytes,
                                sample_rate=24000,
                                num_channels=1,
                            )
                        )
                        buffer = buffer[self.CHUNK_SIZE :]
        if len(buffer) > 0:
            out_bytes = self._token2wav.stream(buffer, prompt_wav=self._prompt_wav, last_chunk=True)
            await self.queue_frame(
                AudioRawFrame(
                    audio=out_bytes,
                    sample_rate=24000,
                    num_channels=1,
                )
            )

        out_text = ""
        if len(out_text_tokens) > 0 and is_out_text is True:
            out_text = self._audio_llm.llm_tokenizer.decode(out_text_tokens)
            if self._text_stream_out is False:
                await self.queue_frame(TextFrame(text=out_text))

        return output_tokens, out_text

    async def process_out_text(self, token_iter):
        out_text_tokens = []
        is_tag = False
        for token_id in token_iter:
            if token_id >= 151688:
                continue
            if token_id == 27:  # <
                is_tag = True
                continue
            if token_id == 29:  # >
                is_tag = False
                continue
            if is_tag:
                continue
            if token_id in self._audio_llm.eos_token_id:
                break
            # text
            out_text_tokens.append(token_id)
            if self._text_stream_out is True:
                out_text = self._audio_llm.llm_tokenizer.decode(token_id)
                await self.queue_frame(TextFrame(text=out_text))

        out_text = ""
        if len(out_text_tokens) > 0:
            out_text = self._audio_llm.llm_tokenizer.decode(out_text_tokens)
            if self._text_stream_out is False:
                await self.queue_frame(TextFrame(text=out_text))

        return out_text


# --------------------------------------------------------------------------------


# A1->T2
class StepAudio2TextProcessor(StepAudio2BaseProcessor):
    """
    audio -> audio_LLM -> text
    - A1->T2 (ASR(trancribe), Audio Understanding, S2TT(support: en,zh,ja))
    - system prompt example:
        - ASR: 请记录下你所听到的语音内容。
        - Audio Understanding: Please briefly explain the important events involved in this audio clip.
        - S2TT: 请仔细聆听这段语音，然后将其内容翻译成中文。
    """

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio = frame.path
        else:
            audio = bytes2TorchTensorWith16(frame.audio)

        if len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "human",
                "content": [{"type": "audio", "audio": audio}],
            },
            {"role": "assistant", "content": None},
        ]
        self._session.ctx.state["messages"] = messages
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        await self.process_out_text(token_iter)


class StepASRProcessor(StepAudio2TextProcessor):
    SYS_PROMPT = "请记录下你所听到的语音内容。"


class StepAudioCaptionProcessor(StepAudio2TextProcessor):
    SYS_PROMPT = "Please briefly explain the important events involved in this audio clip."


class StepS2TTProcessor(StepAudio2TextProcessor):
    SYS_PROMPT = "请仔细聆听这段语音，然后将其内容翻译成中文。"


# --------------------------------------------------------------------------------


# T1->A2
class StepText2AudioProcessor(StepAudio2BaseProcessor):
    """
    text -> audio_LLM -> audio
    - T1-A2 (TTS)
    """


class StepTTSProcessor(StepText2AudioProcessor):
    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        await self.say(
            user_input, is_out_text=False, system_prompt="以自然的语速读出下面的文字。\n"
        )


# --------------------------------------------------------------------------------


# T1-T2A2
class StepText2TextAudioProcessor(StepAudio2BaseProcessor):
    """
    text -> audio_LLM -> audio
    - T1-T2A2 (T2ST)
    """


class StepT2STProcessor(StepText2TextAudioProcessor):
    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        await self.say(
            user_input, is_out_text=True, system_prompt="请将下面的文本翻译成英文，并用语音播报。\n"
        )


# --------------------------------------------------------------------------------


# A1-T2A2
class StepAudio2TextAudioProcessor(StepAudio2BaseProcessor):
    """
    audio -> audio_LLM -> text and audio
    - A1-T2A2 (S2ST, Paralingustic information understanding)
    """

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio = frame.path
        else:
            audio = bytes2TorchTensorWith16(frame.audio)

        if len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "human",
                "content": [{"type": "audio", "audio": audio}],
            },
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        ]
        self._session.ctx.state["messages"] = messages
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        await self.process_out_audio_text(token_iter, is_out_text=True)


class StepS2STProcessor(StepAudio2TextAudioProcessor):
    SYS_PROMPT = "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"


class StepParalingusticInformationUnderstandingProcessor(StepAudio2TextAudioProcessor):
    SYS_PROMPT = "请用语音与我交流。"


# --------------------------------------------------------------------------------


# Chat: multi turn TQTA
class StepText2TextChatProcessor(StepAudio2BaseProcessor):
    """
    text -> audio_LLM -> text
    - T1->T2 (Text Query and Text Answer)
    - system prompt example:
        - TQTA: "You are a helpful assistant."
    """

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()

        self._session.chat_history.append(
            {"role": "human", "content": [{"type": "text", "text": user_input}]}
        )
        self._session.chat_history.append({"role": "assistant", "content": None})
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        out_text = await self.process_out_text(token_iter)
        self._session.chat_history.pop(-1)
        self._session.chat_history.append({"role": "assistant", "content": out_text})


# Chat: multi turn AQTA
class StepAudio2TextChatProcessor(StepAudio2BaseProcessor):
    """
    audio -> audio_LLM -> text
    - A1->T2 (Audio Query and Text Answer)
    - system prompt example:
        - AQTA: "You are a helpful assistant."
    """

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio = frame.path
        else:
            audio = bytes2TorchTensorWith16(frame.audio)

        if audio is None or len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        self._session.chat_history.append(
            {"role": "human", "content": [{"type": "audio", "audio": audio}]}
        )
        self._session.chat_history.append({"role": "assistant", "content": None})
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        out_text = await self.process_out_text(token_iter)
        self._session.chat_history.pop(-1)
        self._session.chat_history.append({"role": "assistant", "content": out_text})


# Chat: multi turn TQAA
class StepText2TextAudioChatProcessor(StepAudio2BaseProcessor):
    """
    text -> audio_LLM -> audio and text
    - T1-T2A2 (Text Query and Audio Answer)
    - system prompt example:
        - TQAA: "You are a helpful assistant."
    """

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()

        self._session.chat_history.append(
            {"role": "human", "content": [{"type": "text", "text": user_input}]}
        )
        self._session.chat_history.append(
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        )
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )

        output_tokens, _ = await self.process_out_audio_text(token_iter, is_out_text=True)

        self._session.chat_history.pop(-1)
        self._session.chat_history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": output_tokens},
                ],
            }
        )


# Chat: multi turn AQAA
class StepAudio2TextAudioChatProcessor(StepAudio2BaseProcessor):
    """
    Audio -> audio_LLM -> audio and text
    - A1-T2A2 (Text Query and Audio Answer)
    - system prompt example:
        - AQAA: "You are a helpful assistant."
    """

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, PathAudioRawFrame):
            audio = frame.path
        else:
            audio = bytes2TorchTensorWith16(frame.audio)

        if audio is None or len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        self._session.chat_history.append(
            {"role": "human", "content": [{"type": "audio", "audio": audio}]}
        )
        self._session.chat_history.append(
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        )
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )

        output_tokens, _ = await self.process_out_audio_text(token_iter, is_out_text=True)

        self._session.chat_history.pop(-1)
        self._session.chat_history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<tts_start>"},
                    {"type": "token", "token": output_tokens},
                ],
            }
        )


# ----------------------------------------------------------------------------------


# A1T1->T2
class StepAudioText2TextProcessor(StepAudio2BaseProcessor):
    """
    audio+text -> audio_LLM -> text
    - A1T1->T2 (Audio Understanding with Text Query)
    """

    async def run_voice(self, frame: TextQuestionsAudioRawFrame) -> AsyncGenerator[Frame, None]:
        audio = bytes2TorchTensorWith16(frame.audio)

        if len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "human",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": frame.text},
                ],
            },
            {"role": "assistant", "content": None},
        ]
        self._session.ctx.state["messages"] = messages
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        await self.process_out_text(token_iter)


# ----------------------------------------------------------------------------------


# A1T1->T2A2
class StepAudioText2TextAudioProcessor(StepAudio2BaseProcessor):
    """
    audio+text -> audio_LLM -> text
    - A1T1->T2A2 (Audio Understanding with Text Query)
    """

    async def run_voice(self, frame: TextQuestionsAudioRawFrame) -> AsyncGenerator[Frame, None]:
        audio = bytes2TorchTensorWith16(frame.audio)

        if len(audio) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "human",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": frame.text},
                ],
            },
            {
                "role": "assistant",
                "content": "<tts_start>",
                "eot": False,
            },  # Insert <tts_start> for speech response
        ]
        self._session.ctx.state["messages"] = messages
        token_iter = self._audio_llm.generate(
            self._session, max_new_tokens=2048, temperature=0.7, do_sample=True
        )
        await self.process_out_audio_text(token_iter, is_out_text=True)


# audio understanding with text query
class StepMMAUProcessor(StepAudioText2TextAudioProcessor):
    SYS_PROMPT = "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."
