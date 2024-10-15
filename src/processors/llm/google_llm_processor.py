import logging
import asyncio
import os
from typing import Iterable, List, AsyncGenerator, Literal


from apipeline.frames.data_frames import TextFrame, Frame
from apipeline.pipeline.pipeline import FrameDirection


try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google.generativeai.types import content_types, generation_types
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use ❤️ Google Generative AI ❤️, you need to `pip install achatbot[google_llm_processor]`. Also, set the environment variable GOOGLE_API_KEY`."
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

    def __init__(self, *,
                 api_key: str = "",
                 model: str = "gemini-1.5-flash-latest",
                 tools: content_types.FunctionLibraryType | None = None,
                 tools_mode: Literal["none", "auto", "any"] = "auto",
                 mode: Literal["auto", "manual"] = "manual",
                 **kwargs):
        r"""
        > !NOTE:
        mode:
        - auto: autiomatic function calling, use chat,
                have chat.history, don't need context,
        - manual: manual function calling, use generate,
                no history, need context,
        """
        super().__init__(**kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        gai.configure(api_key=api_key, transport="grpc_asyncio")
        self._client = gai.GenerativeModel(model, tools=tools)
        self._model = model
        self._tools_mode = tools_mode
        self._tools = tools
        self._chat = None
        self._mode = mode
        if mode == "auto":
            self._chat = self._client.start_chat(
                enable_automatic_function_calling=tools_mode != "none")

    def can_generate_metrics(self) -> bool:
        return True

    def set_model(self, model: str):
        self._model = model
        self._client._model_name = model

    def set_tools(self, tools: content_types.FunctionLibraryType | None):
        self._tools = tools

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

    def tool_config_from_mode(self, mode: str, fns: Iterable[str] = ()):
        """
        Create a tool config with the specified function calling mode.
        mode: none, auto, any
        fns: (function tools)
        """
        if mode not in ["none", "auto", "any"]:
            logging.error(f"Invalid mode: {mode}. Defaulting to 'none'.")
            mode = "none"

        return content_types.to_tool_config(
            {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
        )

    async def record_llm_usage_tokens(self, chunk_dict: dict):
        tokens = {
            "processor": self.name,
            "model": self._model,
        }
        if chunk_dict["usage_metadata"]:
            tokens["prompt_tokens"] = chunk_dict["usage_metadata"]["prompt_token_count"]
            tokens["completion_tokens"] = chunk_dict["usage_metadata"]["candidates_token_count"]
            tokens["total_tokens"] = chunk_dict["usage_metadata"]["total_token_count"]
            await self.start_llm_usage_metrics(tokens)

    async def infer(
            self,
            messages: content_types.ContentType,
            stream: bool = False,
    ) -> generation_types.AsyncGenerateContentResponse:
        if self._chat:
            # send_message with stream now just support text, no function tools
            stream = stream if self._tools_mode == "none" else False
            response = await self._chat.send_message_async(
                messages, stream=stream,
                tools=self._tools,
                tool_config=self.tool_config_from_mode(self._tools_mode),
            )
        else:
            response = await self._client.generate_content_async(
                messages, stream=stream,
                tools=self._tools,
                tool_config=self.tool_config_from_mode(self._tools_mode),
            )
        return response

    async def _process_context(self, context: OpenAILLMContext):
        try:
            logging.debug(f"Generating chat: {context.get_messages_json()}")
            messages = self._get_messages_from_openai_context(context)
            await self.start_ttfb_metrics()
            responese = await self.infer(messages, stream=True)
            async for chunk in responese:
                logging.info(f"chunk:{chunk}")
                await self.stop_ttfb_metrics()
                await self.record_llm_usage_tokens(chunk_dict=chunk.to_dict())
                try:
                    parts = chunk.parts
                    text = chunk.text
                    if len(text) == 0:
                        continue
                    await self.push_frame(TextFrame(text))
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logging.warning(
                            f"LLM refused to generate content"
                            f" for safety reasons - {messages}.")
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
