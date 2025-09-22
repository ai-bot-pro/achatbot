import glob
import os
import sys
import time
import json
import queue
import asyncio
import logging
import threading
from typing import AsyncGenerator, List

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

from src.common.utils.time import time_now_iso8601
from src.common.interface import ILlm
from src.common.types import ASSETS_DIR, RECORDS_DIR
from src.processors.voice.base import VoiceProcessorBase
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16
from src.types.frames import (
    Language,
    TextQuestionsAudioRawFrame,
    PathAudioRawFrame,
    LLMGenedTokensFrame,
    FunctionCallFrame,
    ReasoningThinkTextFrame,
    TranscriptionFrame,
    VADStateAudioRawFrame,
)
import src.modules.functions.search.api
import src.modules.functions.weather.api
from src.modules.functions.function import FunctionManager
from src.core.llm.transformers.manual_voice_step2 import TransformersManualVoiceStep2
from src.core.llm.vllm.step_audio2 import VllmClientStepAudio2
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
        set_last_chunk: bool = True,
        is_reasoning_think: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert audio_llm is not None, "audio_llm is None"

        assert isinstance(audio_llm, (VllmClientStepAudio2, TransformersManualVoiceStep2)), (
            "audio_llm is not TransformersManualVoiceStep2 or VllmClientStepAudio2"
        )

        self._audio_llm = audio_llm
        self._is_speaking = is_speaking
        if is_speaking is True:
            token2wav_path = os.path.join(audio_llm.args.lm_model_name_or_path, "token2wav")
            self._prompt_wav = prompt_wav or os.path.join(ASSETS_DIR, "default_female.wav")
            self._token2wav = token2wav or Token2wav(
                token2wav_path,
                prompt_wav=self._prompt_wav,
                warmup_cn=warmup_cn,
                verbose=verbose,
                **kwargs,
            )

        self._system_prompt = init_system_prompt or self.SYS_PROMPT
        self._text_stream_out = text_stream_out
        self._chunk_size = chunk_size or self.CHUNK_SIZE

        self._session = session or Session(**SessionCtx(str(uuid.uuid4())).__dict__)
        self._session.set_chat_history_size(chat_history_size)
        self._session.chat_history.init({"role": "system", "content": self._system_prompt})
        self._tools = FunctionManager.get_tool_calls_by_names(tools)
        if len(self._tools) > 0 and isinstance(self._audio_llm, TransformersManualVoiceStep2):
            tool_json_schemas = json.dumps(self._tools)
            self._session.chat_history.init_tools(
                {"role": "tool_json_schemas", "content": tool_json_schemas}
            )

        # for async generate to yield
        self._queue = queue.Queue()
        self._input_queue = queue.Queue()
        self._generate_thread = None
        self._generate_over = False

        self._sleep_time = no_stream_sleep_time
        self._verbose = verbose
        self._set_last_chunk = set_last_chunk
        # self._is_push_frame = kwargs.get("is_push_frame", False)
        self._is_push_frame = False  # close this param, just debug

        # think
        self._is_reasoning_think = is_reasoning_think

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
                if self._tools:
                    kwargs["tools"] = self._tools

                if kwargs.get("stop") and "</think>" in kwargs.get("stop"):
                    is_think = True
                elif kwargs.get("stop_strings") and "</think>" in kwargs.get("stop_strings"):
                    is_think = True
                else:
                    is_think = False

                if self._verbose:
                    print(f"{session=} {kwargs=} {is_think=}")

                token_iter = self._audio_llm.generate(session, **kwargs)
                _, _, is_tool_call = self.put_out_audio_text(
                    token_iter, is_out_text=True, is_think=is_think
                )
                if (
                    is_tool_call is False and is_think is False
                ):  # no tool call, no think end round generate
                    self._queue.put(None)  # Signal the end of the stream

                # reset token2wav stream cache
                if self._is_speaking is True and is_think is False:
                    start = time.time()
                    self._token2wav.set_stream_cache(self._prompt_wav)
                    print(f"token2wav.set_stream_cache cost: {time.time() - start}")
            except Exception as e:
                logging.error(f"Exception generate: {e}", exc_info=True)
                self._queue.put(None)  # Signal the end of the stream
                break

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._generate_thread = threading.Thread(target=self._generate)
        self._generate_thread.daemon = True
        self._generate_thread.start()
        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_over = True
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._input_queue.put(None)  # Signal the thread to stop
        self._generate_over = True
        self._generate_thread.join()  # Wait for the thread to finish
        logging.info("cancel done")

    async def gen(self, is_push_frame: bool = False) -> AsyncGenerator[Frame, None]:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    logging.info(f"generate done")
                    break  # End of the stream
                # logging.info(f"generate data: {item}")
                if is_push_frame is True or self._is_push_frame is True:
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

    def put_out_audio_text(self, token_iter, is_out_text: bool = True, is_think: bool = False):
        output_token_ids = []
        output_audio_token_ids = []
        out_text_token_ids = []
        is_tool = False
        tool_calls_token_ids = []
        is_tag = False
        buffer = []
        unicode_token_id = []
        tool_calls: List[FunctionCallFrame] = []  # for v1/chat/completions api

        for token_id in token_iter:
            if self._generate_over is True:
                break
            if isinstance(token_id, dict) and "tool_calls" in token_id:
                for tool_call in token_id["tool_calls"]:
                    tool_calls.append(
                        FunctionCallFrame(
                            tool_call_id=tool_call["id"],
                            type=tool_call["type"],
                            index=tool_call["index"],
                            function_name=tool_call["function"]["name"],
                            arguments=tool_call["function"]["arguments"],
                        )
                    )
                continue
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
                    output_audio_token_ids.append(token_id)
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
            out_bytes = self._token2wav.stream(
                buffer, prompt_wav=self._prompt_wav, last_chunk=self._set_last_chunk
            )
            frame = AudioRawFrame(
                audio=out_bytes,
                sample_rate=self._audio_llm.RATE,
                num_channels=1,
            )
            self._queue.put(frame)

        out_text = ""
        if len(out_text_token_ids) > 0 and is_out_text is True:
            out_text = self._audio_llm.llm_tokenizer.decode(out_text_token_ids)
            if "</think>" not in out_text and is_think:  # think mod out text include </think>
                out_text += "</think>"
            if self._is_reasoning_think and is_think:
                self._queue.put(ReasoningThinkTextFrame(text=out_text))
            if self._text_stream_out is False:
                frame = TextFrame(text=out_text)
                self._queue.put(frame)

        out_audio = (
            self._audio_llm.llm_tokenizer.decode(output_audio_token_ids)
            if len(output_audio_token_ids) > 0
            else []
        )

        is_tool_call = False
        if len(tool_calls_token_ids) > 0:
            tool_calls_token = self._audio_llm.llm_tokenizer.decode(tool_calls_token_ids)
            # print(f"{tool_calls_token=}")
            function_name, function_args = extract_function_info(tool_calls_token)
            self._queue.put(FunctionCallFrame(function_name=function_name, arguments=function_args))
            is_tool_call = True
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                self._queue.put(
                    FunctionCallFrame(
                        function_name=tool_call.function_name, arguments=tool_call.arguments
                    )
                )
            is_tool_call = True

        self._queue.put(
            LLMGenedTokensFrame(
                token_ids=output_token_ids,
                text_tokens=out_text,
                audio_tokens=out_audio,
                tool_calls=tool_calls,
            )
        )

        return output_token_ids, out_text, is_tool_call


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
        self.language = kwargs.pop("language", None)
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
            if isinstance(frame, VADStateAudioRawFrame) and isinstance(item, TextFrame):
                language = Language(self.language) if self.language else None
                yield TranscriptionFrame(
                    text=item.text,
                    user_id=self._session.ctx.client_id,
                    timestamp=time_now_iso8601(),
                    language=language,
                    speech_id=frame.speech_id,
                    start_at_s=frame.start_at_s,
                    end_at_s=frame.end_at_s,
                )
            else:
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
        self._session.increment_chat_round()


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
        self._session.increment_chat_round()


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
            }  # Insert <tts_start> for speech response
        )
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        self.send_input(self._session)
        async for item in self.gen():
            if isinstance(item, LLMGenedTokensFrame):
                self._session.chat_history.pop(-1)
                if isinstance(self._audio_llm, TransformersManualVoiceStep2):
                    self._session.chat_history.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "<tts_start>"},
                                {"type": "token", "token": item.token_ids},
                            ],
                        }
                    )
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item = {
                        "role": "assistant",
                        "tts_content": {
                            "tts_text": item.text_tokens,
                            "tts_audio": item.audio_tokens,
                        },
                    }
                    if len(item.tool_calls) > 0:
                        history_item["tool_calls"] = []
                        for tool_call in item.tool_calls:
                            history_item["tool_calls"].append(
                                {
                                    "id": tool_call.tool_call_id,
                                    "type": tool_call.type,
                                    "index": tool_call.index,
                                    "function": {
                                        "name": tool_call.function_name,
                                        "arguments": tool_call.arguments,
                                    },
                                }
                            )
                    self._session.chat_history.append(history_item)
            if isinstance(item, FunctionCallFrame):  # send input for function call
                func_res = await asyncio.to_thread(
                    FunctionManager.execute,
                    item.function_name,
                    self._session,
                    **item.arguments_dict,
                )
                history_item = {
                    "role": "input",
                    "content": [
                        {"type": "text", "text": func_res},
                        {
                            "type": "text",
                            "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                        },
                    ],
                }
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item["tool_call_id"] = item.tool_call_id
                self._session.chat_history.append(history_item)
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
        self._session.increment_chat_round()


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
        async for item in self.gen():
            if isinstance(item, LLMGenedTokensFrame):
                self._session.chat_history.pop(-1)
                if isinstance(self._audio_llm, TransformersManualVoiceStep2):
                    self._session.chat_history.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "<tts_start>"},
                                {"type": "token", "token": item.token_ids},
                            ],
                        }
                    )
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item = {
                        "role": "assistant",
                        "tts_content": {
                            "tts_text": item.text_tokens,
                            "tts_audio": item.audio_tokens,
                        },
                    }
                    if len(item.tool_calls) > 0:
                        history_item["tool_calls"] = []
                        for tool_call in item.tool_calls:
                            history_item["tool_calls"].append(
                                {
                                    "id": tool_call.tool_call_id,
                                    "type": tool_call.type,
                                    "index": tool_call.index,
                                    "function": {
                                        "name": tool_call.function_name,
                                        "arguments": tool_call.arguments,
                                    },
                                }
                            )
                    self._session.chat_history.append(history_item)
            if isinstance(item, FunctionCallFrame):  # send input for function call
                func_res = await asyncio.to_thread(
                    FunctionManager.execute,
                    item.function_name,
                    self._session,
                    **item.arguments_dict,
                )
                history_item = {
                    "role": "input",
                    "content": [
                        {"type": "text", "text": func_res},
                        {
                            "type": "text",
                            "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                        },
                    ],
                }
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item["tool_call_id"] = item.tool_call_id
                self._session.chat_history.append(history_item)
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
        self._session.increment_chat_round()


# Chat: multi turn AQAA with reasoning think
class StepAudio2TextAudioThinkChatProcessor(StepAudio2BaseProcessor):
    SYS_PROMPT = "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"

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
            {"role": "assistant", "content": "\n<think>\n", "eot": False}
        )
        self._session.ctx.state["messages"] = self._session.chat_history.to_list()
        # get think content, stop when "</think>" appears
        # if use hf transformers, stop_strings need text tokenizer
        if isinstance(self._audio_llm, TransformersManualVoiceStep2):
            self.send_input(self._session, stop_strings=["</think>"])
        elif isinstance(self._audio_llm, VllmClientStepAudio2):
            self.send_input(self._session, stop=["</think>"])
        else:
            raise Exception("Unsupported LLM engine type")
        async for item in self.gen():
            if isinstance(item, LLMGenedTokensFrame):
                self._session.chat_history.pop(-1)
                if isinstance(self._audio_llm, TransformersManualVoiceStep2):
                    self._session.chat_history.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "<tts_start>"},
                                {"type": "token", "token": item.token_ids},
                            ],
                        }
                    )
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item = {
                        "role": "assistant",
                        "tts_content": {
                            "tts_text": item.text_tokens,
                            "tts_audio": item.audio_tokens,
                        },
                    }
                    if len(item.tool_calls) > 0:
                        history_item["tool_calls"] = []
                        for tool_call in item.tool_calls:
                            history_item["tool_calls"].append(
                                {
                                    "id": tool_call.tool_call_id,
                                    "type": tool_call.type,
                                    "index": tool_call.index,
                                    "function": {
                                        "name": tool_call.function_name,
                                        "arguments": tool_call.arguments,
                                    },
                                }
                            )
                    self._session.chat_history.append(history_item)
            if isinstance(item, FunctionCallFrame):  # send input for function call
                func_res = await asyncio.to_thread(
                    FunctionManager.execute,
                    item.function_name,
                    self._session,
                    **item.arguments_dict,
                )
                history_item = {
                    "role": "input",
                    "content": [
                        {"type": "text", "text": func_res},
                        {
                            "type": "text",
                            "text": "\n\n\n请用口语化形式总结检索结果，简短地回答用户的问题。",
                        },
                    ],
                }
                if isinstance(self._audio_llm, VllmClientStepAudio2):
                    history_item["tool_call_id"] = item.tool_call_id
                self._session.chat_history.append(history_item)
                self._session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "<tts_start>",
                        "eot": False,
                    },  # Insert <tts_start> for speech response
                )
                self._session.ctx.state["messages"] = self._session.chat_history.to_list()
                self.send_input(self._session)
            if isinstance(item, ReasoningThinkTextFrame):  # send input for reasoning think result
                self._session.chat_history.pop(-1)
                self._session.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"\n<think>\n{item.text}\n<tts_start>",
                        "eot": False,
                    }
                )
                self._session.ctx.state["messages"] = self._session.chat_history.to_list()
                self.send_input(self._session)
            yield item
        self._session.increment_chat_round()


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


class StepMockAQAAProcessor(StepAudio2TextAudioChatProcessor):
    def __init__(self, token2wav=None, **kwargs):
        super().__init__(token2wav=token2wav, **kwargs)

    def put_out_audio_text(self, token_iter, is_out_text: bool = True, is_think: bool = False):
        wav_files = glob.glob(os.path.join(RECORDS_DIR, "bot_speak*.wav"))

        wav_files.sort(key=os.path.getmtime)

        for wav_file in wav_files:
            if self._generate_over is True:
                break
            try:
                with open(wav_file, "rb") as f:
                    self._token2wav.stream(
                        self._token2wav.WARMUP_TOKENS[
                            : self._token2wav.CHUNK_SIZE + self._token2wav.flow.pre_lookahead_len
                        ],
                        prompt_wav=self._prompt_wav,
                    )

                    frame = AudioRawFrame(
                        audio=f.read(),
                        sample_rate=self._audio_llm.RATE,
                        num_channels=1,
                    )
                    self._queue.put(frame)
            except Exception as e:
                print(f"Error reading {wav_file}: {str(e)}")

        self._queue.put(LLMGenedTokensFrame())

        return [], "", False


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
