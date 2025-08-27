import asyncio
import logging

from transformers import AutoTokenizer
from apipeline.frames.sys_frames import Frame
from apipeline.frames.data_frames import TextFrame
from apipeline.pipeline.pipeline import Pipeline, FrameProcessor, FrameDirection

from src.common import interface
from src.common.factory import EngineClass, EngineFactory
from src.core.llm import LLMEnvInit
from src.types.speech.language import TRANSLATE_LANGUAGE
from src.types.frames import TranslationStreamingFrame, TranslationFrame
from src.processors.session_processor import SessionProcessor
from src.common.session import Session, SessionCtx


class LLMTranslateProcessor(SessionProcessor):
    """
    LLM translate processor with transformers tokenizer
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        generator: interface.ILlmGenerator | EngineClass,
        session: Session | None = None,
        src: str = "en",
        target: str = "zh",
        streaming: bool = False,
        **kwargs,
    ):
        super().__init__(session, **kwargs)
        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        assert generator is not None, "generator must be provided"
        self.generator = generator

        if src not in TRANSLATE_LANGUAGE:
            raise Exception(f"src language {src} not supported")
        if target not in TRANSLATE_LANGUAGE:
            raise Exception(f"target language {target} not supported")

        self._src = src
        self._target = target
        self._streaming = streaming

        self._translate_frame = TextFrame
        self._translate_prompt = f"Translate the following {TRANSLATE_LANGUAGE[src]} sentence into {TRANSLATE_LANGUAGE[target]}:\n"
        logging.info(f"translate prompt: {self._translate_prompt} | streaming: {self._streaming}")

    @property
    def translate_prompt(self, prompt: str):
        return self._translate_prompt

    def set_translate_prompt(self, prompt: str):
        self._translate_prompt = prompt

    def set_translate_frame(self, frame_type=TextFrame):
        if not issubclass(frame_type, TextFrame):
            raise Exception(f"frame_type must be a subclass of {TextFrame.__name__}")
        self._translate_frame = frame_type

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._src == self._target:
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, self._translate_frame):
            await self.push_frame(frame, direction)
            return

        await self.handle_translate_frame(frame)

    async def handle_translate_frame(self, frame: TextFrame):
        prompt = (
            self._translate_prompt + frame.text + f" <{self._target}>"
        )  # need blank space at the end
        token_ids = self.tokenizer.encode(prompt)
        self.session.ctx.state["token_ids"] = token_ids
        if "ctranslate2" in self.generator.SELECTED_TAG:
            start_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            self.session.ctx.state["tokens"] = start_tokens

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        first = True
        gen_token_ids = []
        async for gen_token_id in self.generator.generate(self.session, max_new_tokens=512):
            if first:
                await self.stop_ttfb_metrics()
                first = False

            if gen_token_id == self.tokenizer.bos_token_id:
                continue
            if gen_token_id == self.tokenizer.eos_token_id:
                break

            if self._streaming:
                gen_text = self.tokenizer.decode(gen_token_id)
                await self.queue_frame(TranslationStreamingFrame(text=gen_text))
            gen_token_ids.append(gen_token_id)

        if self._streaming:
            await self.queue_frame(TranslationStreamingFrame(text="", is_final=True))

        await self.stop_processing_metrics()

        if self._streaming:
            return

        text = self.tokenizer.decode(gen_token_ids)
        await self.push_frame(
            TranslationFrame(
                src_lang=self._src,
                target_lang=self._target,
                src_text=frame.text,
                target_text=text,
            )
        )
