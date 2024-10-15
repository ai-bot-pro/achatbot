import logging
import asyncio
import os
from typing import List


from apipeline.frames.data_frames import TextFrame, Frame
from apipeline.pipeline.pipeline import FrameDirection


try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use ❤️ Google Generativ AI ❤️, you need to `pip install achatbot[google_llm_processor]`. Also, set the environment variable GOOGLE_API_KEY`."
    )
    raise Exception(f"Missing module: {e}")

from src.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from src.processors.llm.base import LLMProcessor
from src.types.frames.control_frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMModelUpdateFrame
from src.types.frames.data_frames import LLMMessagesFrame, VisionImageRawFrame


class GoogleAILLMProcessor(LLMProcessor):
    """This class implements inference with Google's AI models

    This processor translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM processor, so that it is easy to switch between different LLMs.
    see: https://ai.google.dev/gemini-api/docs/ @google-gemini

    TODO: tools
    """

    def __init__(self, *, api_key: str, model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        gai.configure(api_key=api_key)
        self.set_model(model)
        self._client = gai.GenerativeModel(model)

    def can_generate_metrics(self) -> bool:
        return True

    def _get_messages_from_openai_context(self, context: OpenAILLMContext) -> List[glm.Content]:
        openai_messages = context.get_messages()
        google_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            parts = [glm.Part(text=content)]
            if "mime_type" in message:
                parts.append(
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type=message["mime_type"], data=message["data"].getvalue()
                        )
                    )
                )
            google_messages.append({"role": role, "parts": parts})

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        try:
            logging.debug(f"Generating chat: {context.get_messages_json()}")
            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()
            response = self._client.generate_content(messages, stream=True)
            async for chunk in self._async_generator_wrapper(response):
                # logging.info(f"chunk:{chunk.model_dump_json()}")
                try:
                    text = chunk.text
                    if len(text) == 0:
                        continue
                    await self.stop_ttfb_metrics()
                    await self.push_frame(TextFrame(text))
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logging.debug(
                            f"LLM refused to generate content for safety reasons - {messages}."
                        )
                    else:
                        logging.exception(f"{self} error: {e}")

        except Exception as e:
            logging.exception(f"{self} exception: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logging.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
