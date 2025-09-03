import asyncio
import logging
from string import Template

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

# use string template to replace placeholders, maybe use jinja2 like tokenizer chat_template with tpl logic
PROMPT_TPL_MAP = {
    "seed-x": "Translate the following $SRC_LANGUAGE sentence into $TARGET_LANGUAGE:\n$TEXT <$TARGET>",
    "hunyuan-mt": "<|startoftext|>Translate the following segment into $TARGET_LANGUAGE, without additional explanation.\n\n$TEXT<|extra_0|>",
    "hunyuan-mt-zh": "<|startoftext|>把下面的文本翻译成$TARGET_LANGUAGE，不要额外解释。\n\n$TEXT<|extra_0|>",
}


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
        prompt_tpl: str = "seed-x",
        **kwargs,
    ):
        super().__init__(session, **kwargs)
        assert tokenizer is not None, "tokenizer must be provided"
        self.tokenizer = tokenizer
        assert generator is not None, "generator must be provided"
        self.generator = generator
        prompt_tpl = prompt_tpl or "seed-x"
        assert prompt_tpl in PROMPT_TPL_MAP, f"prompt_tpl {prompt_tpl} not supported"

        if src not in TRANSLATE_LANGUAGE:
            raise Exception(f"src language {src} not supported")
        if target not in TRANSLATE_LANGUAGE:
            raise Exception(f"target language {target} not supported")

        self._src = src
        self._target = target
        self._streaming = streaming

        self._translate_frame = TextFrame
        self._translate_tpl = Template(PROMPT_TPL_MAP[prompt_tpl])
        logging.info(f"use {prompt_tpl} translate prompt with streaming: {self._streaming}")

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
        prompt = self._translate_tpl.substitute(
            SRC=self._src,
            TARGET=self._target,
            SRC_LANGUAGE=TRANSLATE_LANGUAGE[self._src],
            TARGET_LANGUAGE=TRANSLATE_LANGUAGE[self._target],
            TEXT=frame.text,
        )
        logging.info(f"{prompt=}")
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
                text=text,
            )
        )
