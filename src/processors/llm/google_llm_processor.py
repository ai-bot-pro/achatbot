import json
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
    from google.protobuf.struct_pb2 import Struct
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use ❤️ Google Generative AI ❤️, you need to `pip install achatbot[google_llm_processor]`. Also, set the environment variable GOOGLE_API_KEY`."
    )
    raise Exception(f"Missing module: {e}")

from src.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from src.processors.llm.base import LLMProcessor, UnhandledFunctionException
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMModelUpdateFrame,
)
from src.types.frames.data_frames import LLMMessagesFrame, VisionImageRawFrame


class GoogleAILLMProcessor(LLMProcessor):
    """This class implements inference with Google's AI models

    This processor translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM processor, so that it is easy to switch between different LLMs.
    see: https://ai.google.dev/gemini-api/docs/ @google-gemini

    tools: https://ai.google.dev/gemini-api/docs/function-calling
    @TODO: parallel_function_calls
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "gemini-1.5-flash-latest",
        tools: content_types.FunctionLibraryType | None = None,
        tools_mode: Literal["none", "auto", "any"] = "auto",
        mode: Literal["auto", "manual"] = "manual",
        generation_config: generation_types.GenerationConfigType | None = None,
        **kwargs,
    ):
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
        self._model = model
        self._tools_mode = tools_mode
        self._tools = tools
        self._chat = None
        self._mode = mode
        self._generation_config = generation_config
        self.set_client()

    def can_generate_metrics(self) -> bool:
        return True

    def set_model(self, model: str):
        self._model = model
        self._client._model_name = model

    def set_client(self):
        self._client = gai.GenerativeModel(
            model_name=self._model,
            tools=self._tools,
            generation_config=self._generation_config,
        )
        if self._mode == "auto":
            self._chat = self._client.start_chat(
                enable_automatic_function_calling=self._tools_mode != "none"
            )

    def set_tools(self, tools: content_types.FunctionLibraryType | None):
        self._tools = tools

    def _get_tools_from_openai_context(
        self,
        context: OpenAILLMContext,
    ) -> content_types.FunctionLibraryType | None:
        if not context.tools:
            return None

        google_tools = {"function_declarations": []}
        for tool in context.tools:
            if "function" in tool:
                google_tools["function_declarations"].append(tool["function"])
        return google_tools

    def _get_messages_from_openai_context(self, context: OpenAILLMContext) -> List[glm.Content]:
        r"""openai context history message
        [
          {
            "role": "system",
            "content": "You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.  You have access to two tools: get_weather   You can respond to questions about the weather using the get_weather tool.\n  If you need to use a tool, simply use the tool. Do not tell the user the tool you are using. Be brief and concise.\n Please communicate in Chinese"
          },
          { "role": "user", "content": " What's the weather like in New York today? " },
          {
            "role": "assistant",
            "tool_calls": [
              {
                "id": "",
                "function": {
                  "name": "get_weather",
                  "arguments": "{\"location\": \"New York, NY\"}"
                },
                "type": "function"
              }
            ]
          },
          {
            "role": "tool",
            "content": "\"The weather in New York, NY is currently 72 degrees and sunny.\"",
            "tool_call_id": "get_weather"
          }
        ]
        """
        openai_messages = context.get_messages()
        logging.debug(f"openai_messages-->{openai_messages}")
        google_messages = []

        for message in openai_messages:
            if message["role"] == "system" or message["role"] == "user":
                role = "user"
                if "content" in message:
                    content = message["content"]
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

            if message["role"] == "assistant":
                role = "model"
                if "content" in message:
                    content = message["content"]
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

                if "tool_calls" in message:
                    parts = []
                    for tool_call in message["tool_calls"]:
                        args = json.loads(tool_call["function"]["arguments"])
                        parts.append(
                            glm.Part(
                                function_call=glm.FunctionCall(
                                    name=tool_call["function"]["name"],
                                    args=args,
                                )
                            )
                        )

                    google_messages.append({"role": role, "parts": parts})

            if message["role"] == "tool":
                role = "user"
                if "content" in message:
                    s = Struct()
                    s.update({"result": message["content"]})
                    function_response = glm.Part(
                        function_response=glm.FunctionResponse(
                            name=message["tool_call_id"], response=s
                        )
                    )
                    parts = [function_response]
                    google_messages.append({"role": role, "parts": parts})

        logging.debug(f"google_messages-->{google_messages}")

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
        tools: content_types.FunctionLibraryType | None = None,
        stream: bool = False,
    ) -> generation_types.AsyncGenerateContentResponse:
        """
        https://ai.google.dev/api/generate-content
        """
        tools = tools or self._tools
        if self._chat:
            # send_message with stream now just support text, no function tools
            stream = stream if self._tools_mode == "none" else False
            response = await self._chat.send_message_async(
                messages,
                stream=stream,
                tools=tools,
                tool_config=self.tool_config_from_mode(self._tools_mode),
            )
        else:
            response = await self._client.generate_content_async(
                messages,
                stream=stream,
                tools=tools,
                tool_config=self.tool_config_from_mode(self._tools_mode),
            )
        return response

    async def _process_context(self, context: OpenAILLMContext):
        try:
            logging.debug(f"Generating chat: {context.get_messages_json()}")
            messages = self._get_messages_from_openai_context(context)
            tools = self._get_tools_from_openai_context(context)
            await self.start_ttfb_metrics()
            responese = await self.infer(messages, tools=tools, stream=True)
            async for chunk in responese:
                logging.debug(f"chunk:{chunk}")
                await self.record_llm_usage_tokens(chunk_dict=chunk.to_dict())

                if len(chunk.candidates) == 0:
                    continue
                await self.stop_ttfb_metrics()
                # Google LLMs seem to flag safety issues a lot!
                if chunk.candidates[0].finish_reason == 3:
                    logging.warning(
                        f"LLM refused to generate content" f" for safety reasons - {messages}."
                    )
                    continue

                if text := chunk.parts[0].text:
                    if len(text) > 0:
                        await self.push_frame(TextFrame(text))
                elif func_call := chunk.parts[0].function_call:
                    logging.debug(f"function_call:{func_call}")
                    args = {}
                    for key, item in func_call.args.items():
                        args[key] = item
                    if self.has_function(func_call.name):
                        # no tool_call_id, use func_call.name
                        await self.call_function(
                            context=context,
                            tool_call_id=func_call.name,
                            function_name=func_call.name,
                            arguments=args,
                        )
                    else:
                        raise UnhandledFunctionException(
                            f"The LLM tried to call a function named '{func_call.name}', "
                            f"but there isn't a callback registered for that function."
                        )

        except Exception as e:
            logging.exception(f"{self} exception: {e}")

    async def call_function(
        self, *, context: OpenAILLMContext, tool_call_id: str, function_name: str, arguments: dict
    ) -> None:
        f = None
        if function_name in self._callbacks.keys():
            f = self._callbacks[function_name]
        elif None in self._callbacks.keys():
            f = self._callbacks[None]
        else:
            return None
        await context.call_function(
            f, function_name=function_name, tool_call_id=tool_call_id, arguments=arguments, llm=self
        )

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
