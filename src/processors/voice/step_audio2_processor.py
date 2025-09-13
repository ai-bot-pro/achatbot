import os
import sys
import time
import json
import queue
import asyncio
import logging
import threading
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
from src.types.frames import (
    TextQuestionsAudioRawFrame,
    PathAudioRawFrame,
    LLMGenedTokensFrame,
    FunctionCallFrame,
)
import src.modules.functions.search.api
import src.modules.functions.weather.api
from src.modules.functions.function import FunctionManager
from .helper import extract_function_info


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
        no_stream_sleep_time: float = 0.5,
        chunk_size: int = 0,
        session: Session | None = None,
        audio_llm: ILlm | None = None,
        token2wav: Token2wav | None = None,
        is_speaking: bool = True,
        tools: list = [],
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert audio_llm is not None, "audio_llm is None"

        from src.core.llm.transformers.manual_voice_step2 import TransformersManualVoiceStep2

        assert isinstance(audio_llm, TransformersManualVoiceStep2), (
            "audio_llm is not TransformersManualVoiceStep2"
        )

        self._audio_llm = audio_llm
        self._is_speaking = is_speaking
        if is_speaking is True:
            token2wav_path = os.path.join(audio_llm.args.lm_model_name_or_path, "token2wav")
            self._token2wav = token2wav or Token2wav(token2wav_path)
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
        self._chunk_size = chunk_size or self.CHUNK_SIZE

        self._session = session or Session(
            chat_history_size=chat_history_size, **SessionCtx(str(uuid.uuid4())).__dict__
        )
        self._session.chat_history.init({"role": "system", "content": self._system_prompt})
        tool_calls = FunctionManager.get_tool_calls_by_names(tools)
        if len(tool_calls) > 0:
            tool_json_schemas = json.dumps(tool_calls)
            self._session.chat_history.init_tools(
                {"role": "tool_json_schemas", "content": tool_json_schemas}
            )

        # for async generate to yield
        self._queue = queue.Queue()
        self._input_queue = queue.Queue()
        self._generate_thread = None

        self._sleep_time = no_stream_sleep_time
        self._verbose = verbose

    @property
    def chat_history(self):
        return self._session.chat_history

    def reset(self):
        if self._is_speaking is True:
            self._token2wav.cache = {}
        self._session.reset()

    def _generate(self):
        while True:
            try:
                item = self._input_queue.get()
                if item is None:
                    self._queue.put(None)  # Signal the end of the stream
                    break  # Signal to stop the thread
                session, kwargs = item
                token_iter = self._audio_llm.generate(session, **kwargs)
                self.put_out_audio_text(token_iter, is_out_text=True)
                self._queue.put(None)  # Signal the end of the stream
            except Exception as e:
                logging.error(f"Exception generate: {e}", exc_info=True)
                self._queue.put(None)  # Signal the end of the stream
                break

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._generate_thread = threading.Thread(target=self._generate)
        self._generate_thread.start()
        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("cancel done")

    async def gen(self, is_push_frame: bool = False) -> AsyncGenerator[Frame, None]:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    logging.info(f"generate done")
                    break  # End of the stream
                logging.info(f"generate data: {item}")
                if is_push_frame is True:
                    await self.push_frame(item)
                    yield None
                else:
                    yield item
            except queue.Empty:
                # yield asysncio.sleep to allow other tasks to run, e.g.: sink task (write audio)
                await asyncio.sleep(self._sleep_time)
                # logging.info(f"queue empty sleep {self._sleep_time}")
                continue

    def send_input(self, session: Session, **kwargs):
        self._input_queue.put((session, kwargs))

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": self._audio_llm.RATE,
            "channels": 1,
        }

    async def say(
        self,
        text: str,
        system_prompt: str = "以自然的语速读出下面的文字。\n",
        **kwargs,
    ):
        async for item in self.generator_say(
            text,
            system_prompt=system_prompt,
            is_push_frame=True,
            **kwargs,
        ):
            pass

    async def generator_say(
        self,
        text: str,
        system_prompt: str = "以自然的语速读出下面的文字。\n",
        is_push_frame: str = True,
        **kwargs,
    ) -> AsyncGenerator[Frame, None]:
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
        self.send_input(self._session, **kwargs)
        async for item in self.gen(is_push_frame=is_push_frame):
            yield item

    def put_out_audio_text(self, token_iter, is_out_text: bool = True):
        output_token_ids = []
        output_audio_token_ids = []
        out_text_token_ids = []
        is_tool = False
        tool_calls_token_ids = []
        is_tag = False
        buffer = []
        unicode_token_id = []
        for token_id in token_iter:
            if self._verbose is True:
                print(f"{token_id=} {self._audio_llm.llm_tokenizer.decode(token_id)=}")
            output_token_ids.append(token_id)
            if token_id < 151688:  # text
                if token_id == 151657:  # <tool_call>
                    is_tool = True
                    continue
                if token_id == 151658:  # </tool_call>
                    is_tool = False
                    continue
                if is_tool:
                    tool_calls_token_ids.append(token_id)
                    continue

                if token_id == 27:  # <
                    is_tag = True
                    continue
                if token_id == 29:  # >
                    is_tag = False
                    continue
                if is_tag:  # <***>
                    continue

                out_text_token_ids.append(token_id)
                if is_out_text is True and self._text_stream_out is True:
                    out_text = self._audio_llm.llm_tokenizer.decode(unicode_token_id + [token_id])
                    if "�" in out_text:
                        unicode_token_id.append(token_id)
                    else:
                        unicode_token_id = []
                        frame = TextFrame(text=out_text)
                        self._queue.put(frame)
            if token_id > 151695 and self._is_speaking is True:  # audio
                audio_token_id = token_id - 151696
                if audio_token_id < 6561:  # remove audio padding
                    output_audio_token_ids.append(audio_token_id)
                    buffer.append(audio_token_id)
                    if len(buffer) >= self._chunk_size + self._token2wav.flow.pre_lookahead_len:
                        out_bytes = self._token2wav.stream(
                            buffer[: self._chunk_size + self._token2wav.flow.pre_lookahead_len],
                            prompt_wav=self._prompt_wav,
                            last_chunk=False,
                        )
                        frame = AudioRawFrame(
                            audio=out_bytes,
                            sample_rate=self._audio_llm.RATE,
                            num_channels=1,
                        )
                        self._queue.put(frame)
                        buffer = buffer[self._chunk_size :]
        if len(buffer) > 0 and self._is_speaking is True:
            logging.info(f"last chunk size: {len(buffer)}")
            out_bytes = self._token2wav.stream(buffer, prompt_wav=self._prompt_wav, last_chunk=True)
            frame = AudioRawFrame(
                audio=out_bytes,
                sample_rate=self._audio_llm.RATE,
                num_channels=1,
            )
            self._queue.put(frame)

        out_text = ""
        if len(out_text_token_ids) > 0 and is_out_text is True:
            out_text = self._audio_llm.llm_tokenizer.decode(out_text_token_ids)
            if self._text_stream_out is False:
                frame = TextFrame(text=out_text)
                self._queue.put(frame)

        self._queue.put(LLMGenedTokensFrame(token_ids=output_token_ids))

        if len(tool_calls_token_ids) > 0:
            tool_calls_token = self._audio_llm.llm_tokenizer.decode(tool_calls_token_ids)
            # print(f"{tool_calls_token=}")
            function_name, function_args = extract_function_info(tool_calls_token)
            self._queue.put(FunctionCallFrame(function_name=function_name, arguments=function_args))

        return output_token_ids, out_text


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

    def __init__(self, **kwargs):
        kwargs["is_speaking"] = False
        super().__init__(**kwargs)

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
        self.send_input(self._session)
        async for item in self.gen():
            yield item


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
        async for item in self.generator_say(
            user_input, system_prompt="以自然的语速读出下面的文字。\n"
        ):
            yield item


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
        async for item in self.generator_say(
            user_input,
            system_prompt="请将下面的文本翻译成英文，并用语音播报。\n",
            is_push_frame=False,
        ):
            yield item


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
        self.send_input(self._session)
        async for item in self.gen():
            yield item


class StepS2STProcessor(StepAudio2TextAudioProcessor):
    SYS_PROMPT = "请仔细聆听这段语音，然后将其内容翻译成英文并用语音播报。"


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

    def __init__(self, **kwargs):
        kwargs["is_speaking"] = False
        super().__init__(**kwargs)

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()

        self._session.chat_history.append(
            {"role": "human", "content": [{"type": "text", "text": user_input}]}
        )
        self._session.chat_history.append({"role": "assistant", "content": None})
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        self.send_input(self._session)
        out_text = ""
        async for item in self.gen():
            if isinstance(item, TextFrame):
                out_text += item.text
            yield item
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

    def __init__(self, **kwargs):
        kwargs["is_speaking"] = False
        super().__init__(**kwargs)

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
        self.send_input(self._session)
        out_text = ""
        async for item in self.gen():
            if isinstance(item, TextFrame):
                out_text += item.text
            yield item
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
        self.send_input(self._session)
        output_token_ids = []
        async for item in self.gen():
            if isinstance(item, LLMGenedTokensFrame):
                output_token_ids = item.token_ids
                self._session.chat_history.pop(-1)
                self._session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "<tts_start>"},
                            {"type": "token", "token": output_token_ids},
                        ],
                    }
                )
            yield item


# Chat: multi turn AQAA
class StepAudio2TextAudioChatProcessor(StepAudio2BaseProcessor):
    """
    Audio -> audio_LLM -> audio and text
    - A1-T2A2 (Audio Query and Audio+text Answer)
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
        self.send_input(self._session)
        output_token_ids = []
        async for item in self.gen():
            if isinstance(item, LLMGenedTokensFrame):
                output_token_ids = item.token_ids
                self._session.chat_history.pop(-1)
                self._session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "<tts_start>"},
                            {"type": "token", "token": output_token_ids},
                        ],
                    }
                )
            if isinstance(item, FunctionCallFrame):  # send input for function call
                func_res = FunctionManager.execute(
                    item.function_name, self._session, **item.arguments
                )
                self._session.chat_history.append(
                    {
                        "role": "input",
                        "content": [
                            {"type": "text", "text": func_res},
                            {
                                "type": "text",
                                "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                            },
                        ],
                    }
                )
                self._session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "<tts_start>",
                        "eot": False,
                    },  # Insert <tts_start> for speech response
                )
                self._session.ctx.state["messages"] = self._session.chat_history.to_list()
                self.send_input(self._session)
            yield item


# ----------------------------------------------------------------------------------


# A1T1->T2
class StepAudioText2TextProcessor(StepAudio2BaseProcessor):
    """
    audio+text -> audio_LLM -> text
    - A1T1->T2 (Audio Understanding with Text Query)
    """

    def __init__(self, **kwargs):
        kwargs["is_speaking"] = False
        super().__init__(**kwargs)

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
        self.send_input(self._session)
        async for item in self.gen():
            yield item


# ----------------------------------------------------------------------------------


# A1T1->T2A2
class StepAudioText2TextAudioProcessor(StepAudio2BaseProcessor):
    """
    audio+text -> audio_LLM -> audio+text
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
        self.send_input(self._session)
        async for item in self.gen():
            yield item


# audio understanding with text query
class StepMMAUProcessor(StepAudioText2TextAudioProcessor):
    SYS_PROMPT = "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."


"""
python -m src.processors.voice.step_audio2_processor
"""
if __name__ == "__main__":
    from pathlib import Path

    import wave
    from apipeline.frames import AudioRawFrame, StartFrame, EndFrame, CancelFrame

    from src.common.logger import Logger
    from src.types.frames import PathAudioRawFrame
    from src.cmd.bots.voice.step_audio2.helper import (
        get_step_audio2_processor,
    )
    from src.types.ai_conf import AIConfig, LLMConfig

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    async def run_aqaa():
        processor = get_step_audio2_processor(
            LLMConfig(
                processor="StepAudio2TextAudioChatProcessor",
                args={
                    "init_system_prompt": "",
                    "prompt_wav": "./assets/default_male.wav",
                    "warmup_cn": 2,
                    "chat_history_size": None,
                    "text_stream_out": False,
                    "no_stream_sleep_time": 0.5,
                    "lm_model_name_or_path": "./models/stepfun-ai/Step-Audio-2-mini",
                },
            )
        )
        await processor.start(StartFrame())
        for round_idx, audio_path in enumerate(
            [
                "./deps/StepAudio2/assets/multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
                "./deps/StepAudio2/assets/multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
            ]
        ):
            print("round: ", round_idx)
            frame_iter = processor.run_voice(
                PathAudioRawFrame(
                    path=audio_path,
                    audio=b"",
                )
            )
            audio = b""
            async for frame in frame_iter:
                if isinstance(frame, AudioRawFrame):
                    audio += frame.audio
                print(f"{round_idx=} gen_frame-->", frame)
            print(f"{round_idx=} {processor._session.chat_history=}")
            wav_path = Path(
                f"{ASSETS_DIR}/StepAudio2/output-processor-chunks-stream-{round_idx}.wav"
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio)
        await processor.stop(EndFrame())

    asyncio.run(run_aqaa())
