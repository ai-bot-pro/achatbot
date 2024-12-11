"""
Copyright (c) 2024, Daily
SPDX-License-Identifier: BSD 2-Clause License

backend for https://github.com/rtvi-ai to adapter frontend client-side RTVI  stack
- implements RTVI events and data structures.
- processor orchestration
- config and event update

In addition, there will be future network protocols and standards that make sense to use for real-time AI. (For example, [Media over Quic](https://quic.video/)).
"""

import logging
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor
from apipeline.frames.sys_frames import SystemFrame, CancelFrame, ErrorFrame
from apipeline.frames.control_frames import StartFrame, EndFrame

from src.types.frames.sys_frames import BotInterruptionFrame
from src.types.frames.control_frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from src.types.frames.data_frames import (
    Frame,
    FunctionCallResultFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TransportMessageFrame,
)
from src.processors.aggregators.openai_llm_context import OpenAILLMContext

from dotenv import load_dotenv

load_dotenv(override=True)


RTVI_PROTOCOL_VERSION = "0.1"

ActionResult = Union[bool, int, float, str, list, dict]


class RTVIServiceOption(BaseModel):
    name: str
    type: Literal["bool", "number", "string", "array", "object", "dict", "list"]
    handler: Callable[["RTVIProcessor", str, "RTVIServiceOptionConfig"], Awaitable[None]] = Field(
        exclude=True
    )


class RTVIService(BaseModel):
    name: str
    options: List[RTVIServiceOption]
    _options_dict: Dict[str, RTVIServiceOption] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._options_dict = {}
        for option in self.options:
            self._options_dict[option.name] = option
        return super().model_post_init(__context)


class RTVIActionArgumentData(BaseModel):
    name: str
    value: Any


class RTVIActionArgument(BaseModel):
    name: str
    type: Literal["bool", "number", "string", "array", "object", "dict", "list"]


class RTVIAction(BaseModel):
    service: str
    action: str
    handler: Callable[["RTVIProcessor", str, Dict[str, Any]], Awaitable[ActionResult]] = Field(
        exclude=True
    )


#
# Client -> achatbot messages.
#


class RTVIServiceOptionConfig(BaseModel):
    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    config_list: List[RTVIServiceConfig]
    _arguments_dict: Dict[str, Any] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._arguments_dict = {}
        for conf in self.config_list:
            opt_dict = {}
            for opt in conf.options:
                opt_dict[opt.name] = opt.value
            self._arguments_dict[conf.service] = opt_dict
        return super().model_post_init(__context)


class RTVIActionRunArgument(BaseModel):
    name: str
    value: Any


class RTVIActionRun(BaseModel):
    service: str
    action: str
    arguments: Optional[List[RTVIActionRunArgument]] = None


class RTVIMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: str
    id: str
    data: Optional[Dict[str, Any]] = None


#
# achatbot -> Client responses and messages.
#


class RTVIErrorResponseData(BaseModel):
    error: str


class RTVIErrorResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error-response"] = "error-response"
    id: str
    data: RTVIErrorResponseData


class RTVIErrorData(BaseModel):
    error: str
    fatal: bool


class RTVIError(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["error"] = "error"
    data: RTVIErrorData


class RTVIDescribeConfigData(BaseModel):
    config: List[RTVIService]


class RTVIDescribeConfig(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["config-available"] = "config-available"
    id: str
    data: RTVIDescribeConfigData


class RTVIDescribeActionsData(BaseModel):
    actions: List[RTVIAction]


class RTVIDescribeActions(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["actions-available"] = "actions-available"
    id: str
    data: RTVIDescribeActionsData


class RTVIConfigResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["config"] = "config"
    id: str
    data: RTVIConfig


class RTVIActionResponseData(BaseModel):
    result: ActionResult


class RTVIActionResponse(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["action-response"] = "action-response"
    id: str
    data: RTVIActionResponseData


class RTVIBotReadyData(BaseModel):
    version: str
    config: List[RTVIServiceConfig]


class RTVIBotReady(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-ready"] = "bot-ready"
    id: str
    data: RTVIBotReadyData


class RTVILLMFunctionCallMessageData(BaseModel):
    function_name: str
    tool_call_id: str
    args: dict


class RTVILLMFunctionCallMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-function-call"] = "llm-function-call"
    data: RTVILLMFunctionCallMessageData


class RTVILLMFunctionCallStartMessageData(BaseModel):
    function_name: str


class RTVILLMFunctionCallStartMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["llm-function-call-start"] = "llm-function-call-start"
    data: RTVILLMFunctionCallStartMessageData


class RTVILLMFunctionCallResultData(BaseModel):
    function_name: str
    tool_call_id: str
    arguments: dict
    result: dict


class RTVITranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str
    final: bool


class RTVITranscriptionMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVITranscriptionMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIBotStartedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class RTVIBotStoppedSpeakingMessage(BaseModel):
    label: Literal["rtvi-ai"] = "rtvi-ai"
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class RTVIProcessorParams(BaseModel):
    send_bot_ready: bool = True


class RTVIProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        config: RTVIConfig = RTVIConfig(config_list=[]),
        params: RTVIProcessorParams = RTVIProcessorParams(),
    ):
        super().__init__()
        self._config = config
        self._params = params

        self._pipeline: FrameProcessor | None = None
        self._pipeline_started = False

        self._client_ready = False
        self._client_ready_id = ""

        self._registered_actions: Dict[str, RTVIAction] = {}
        self._registered_services: Dict[str, RTVIService] = {}

        self._push_frame_task = self.get_event_loop().create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

        self._message_task = self.get_event_loop().create_task(self._message_task_handler())
        self._message_queue = asyncio.Queue()

        self._curr_rtvi_message: RTVIMessage = None

    @property
    def curr_rtvi_meesage(self):
        return self._curr_rtvi_message

    def register_action(self, action: RTVIAction):
        id = self._action_id(action.service, action.action)
        self._registered_actions[id] = action

    def register_actions(self, actions: List[RTVIAction]):
        for action in actions:
            self.register_action(action)

    def register_service(self, service: RTVIService):
        self._registered_services[service.name] = service

    def register_services(self, services: List[RTVIService]):
        for service in services:
            self.register_service(service)

    async def interrupt_bot(self):
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def send_error(self, error: str):
        message = RTVIError(data=RTVIErrorData(error=error, fatal=False))
        await self._push_transport_message(message)

    async def set_client_ready(self):
        if not self._client_ready:
            self._client_ready = True
            await self._maybe_send_bot_ready()

    async def handle_function_call(
        self,
        function_name: str,
        tool_call_id: str,
        arguments: dict,
        llm: FrameProcessor,
        context: OpenAILLMContext,
        result_callback,
    ):
        fn = RTVILLMFunctionCallMessageData(
            function_name=function_name, tool_call_id=tool_call_id, args=arguments
        )
        message = RTVILLMFunctionCallMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def handle_function_call_start(
        self, function_name: str, llm: FrameProcessor, context: OpenAILLMContext
    ):
        fn = RTVILLMFunctionCallStartMessageData(function_name=function_name)
        message = RTVILLMFunctionCallStartMessage(data=fn)
        await self._push_transport_message(message, exclude_none=False)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, SystemFrame):
            await super().push_frame(frame, direction)
        else:
            await self._internal_push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # print("rtvi process frame", frame)
        # Specific system frames
        if isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, ErrorFrame):
            await self._send_error_frame(frame)
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        elif isinstance(frame, UserStartedSpeakingFrame) or isinstance(
            frame, UserStoppedSpeakingFrame
        ):
            await self._handle_interruptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame) or isinstance(
            frame, BotStoppedSpeakingFrame
        ):
            await self._handle_bot_speaking(frame)
            await self.push_frame(frame, direction)
        # Data frames
        elif isinstance(frame, TranscriptionFrame) or isinstance(frame, InterimTranscriptionFrame):
            await self._handle_transcriptions(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, TransportMessageFrame):
            await self._message_queue.put(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._pipeline:
            await self._pipeline.cleanup()

    async def _start(self, frame: StartFrame):
        self._pipeline_started = True
        await self._maybe_send_bot_ready()

    async def _stop(self, frame: EndFrame):
        # We need to cancel the message task handler because that one is not
        # processing EndFrames.
        self._message_task.cancel()
        await self._message_task
        await self._push_frame_task

    async def _cancel(self, frame: CancelFrame):
        self._message_task.cancel()
        await self._message_task
        self._push_frame_task.cancel()
        await self._push_frame_task

    async def _internal_push_frame(
        self, frame: Frame | None, direction: FrameDirection | None = FrameDirection.DOWNSTREAM
    ):
        await self._push_queue.put((frame, direction))

    async def _push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self._push_queue.get()
                await super().push_frame(frame, direction)
                self._push_queue.task_done()
                running = not isinstance(frame, EndFrame)
            except asyncio.CancelledError:
                break

    async def _push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        frame = TransportMessageFrame(
            message=model.model_dump(exclude_none=exclude_none), urgent=True
        )
        await self.push_frame(frame)

    async def _handle_transcriptions(self, frame: Frame):
        message = None
        if isinstance(frame, TranscriptionFrame):
            message = RTVITranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=True
                )
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            message = RTVITranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=False
                )
            )

        if message:
            await self._push_transport_message(message)

    async def _handle_interruptions(self, frame: Frame):
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVIUserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVIUserStoppedSpeakingMessage()

        if message:
            await self._push_transport_message(message)

    async def _handle_bot_speaking(self, frame: Frame):
        message = None
        if isinstance(frame, BotStartedSpeakingFrame):
            message = RTVIBotStartedSpeakingMessage()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = RTVIBotStoppedSpeakingMessage()

        if message:
            await self._push_transport_message(message)

    async def _message_task_handler(self):
        while True:
            try:
                frame = await self._message_queue.get()
                await self._handle_message(frame)
                self._message_queue.task_done()
            except asyncio.CancelledError:
                break

    async def _handle_message(self, frame: TransportMessageFrame):
        logging.debug(f"handle TransportMessageFrame: {frame}")
        try:
            message = RTVIMessage.model_validate(frame.message)
        except ValidationError as e:
            await self.send_error(f"Invalid incoming message: {e}")
            logging.warning(f"Invalid incoming  message: {e}")
            return

        logging.info(f"rtvi server handle message: {message}")
        self._curr_rtvi_message = message
        try:
            match message.type:
                case "client-ready":
                    await self._handle_client_ready(message.id)
                case "describe-actions":
                    await self._handle_describe_actions(message.id)
                case "describe-config":
                    await self._handle_describe_config(message.id)
                case "get-config":
                    await self._handle_get_config(message.id)
                case "update-config":
                    config = RTVIConfig.model_validate(message.data)
                    await self._handle_update_config(message.id, config)
                case "action":
                    action = RTVIActionRun.model_validate(message.data)
                    await self._handle_action(message.id, action)
                case "llm-function-call-result":
                    data = RTVILLMFunctionCallResultData.model_validate(message.data)
                    await self._handle_function_call_result(data)

                case _:
                    await self._send_error_response(message.id, f"Unsupported type {message.type}")

        except ValidationError as e:
            await self._send_error_response(message.id, f"Invalid incoming message: {e}")
            logging.warning(f"Invalid incoming  message: {e}")
        except Exception as e:
            await self._send_error_response(message.id, f"Exception processing message: {e}")
            logging.warning(f"Exception processing message: {e}", exc_info=True)

    async def _handle_client_ready(self, request_id: str):
        self._client_ready = True
        self._client_ready_id = request_id
        await self._maybe_send_bot_ready()

    async def _handle_describe_config(self, request_id: str):
        services = list(self._registered_services.values())
        message = RTVIDescribeConfig(id=request_id, data=RTVIDescribeConfigData(config=services))
        await self._push_transport_message(message)

    async def _handle_describe_actions(self, request_id: str):
        actions = list(self._registered_actions.values())
        message = RTVIDescribeActions(id=request_id, data=RTVIDescribeActionsData(actions=actions))
        await self._push_transport_message(message)

    async def _handle_get_config(self, request_id: str):
        message = RTVIConfigResponse(id=request_id, data=self._config)
        await self._push_transport_message(message)

    def _update_config_option(self, service: str, config: RTVIServiceOptionConfig):
        for service_config in self._config.config_list:
            if service_config.service == service:
                for option_config in service_config.options:
                    if option_config.name == config.name:
                        option_config.value = config.value
                        return
                # If we couldn't find a value for this config, we simply need to
                # add it.
                service_config.options.append(config)

    async def _update_service_config(self, config: RTVIServiceConfig):
        if config.service not in self._registered_services:
            return
        service = self._registered_services[config.service]
        for option in config.options:
            if option.name in service._options_dict:
                handler = service._options_dict[option.name].handler
                await handler(self, service.name, option)
            else:
                logging.info(f"{option.name} not handler")
            self._update_config_option(service.name, option)

    async def _update_config(self, data: RTVIConfig):
        for service_config in data.config_list:
            await self._update_service_config(service_config)

    async def _handle_update_config(self, request_id: str, data: RTVIConfig):
        # NOTE: The bot might be talking while we receive a new
        # config. Let's interrupt it for now and update the config. Another
        # solution is to wait until the bot stops speaking and then apply the
        # config, but this definitely is more complicated to achieve.
        await self.interrupt_bot()
        await self._update_config(data)
        await self._handle_get_config(request_id)

    async def _handle_function_call_result(self, data):
        frame = FunctionCallResultFrame(
            function_name=data.function_name,
            tool_call_id=data.tool_call_id,
            arguments=data.arguments,
            result=data.result,
        )
        await self.push_frame(frame)

    async def _handle_action(self, request_id: str, data: RTVIActionRun):
        action_id = self._action_id(data.service, data.action)
        if action_id not in self._registered_actions:
            await self._send_error_response(request_id, f"Action {action_id} not registered")
            return
        action = self._registered_actions[action_id]
        arguments = {}
        if data.arguments:
            for arg in data.arguments:
                arguments[arg.name] = arg.value
        result = await action.handler(self, action.service, arguments)
        message = RTVIActionResponse(id=request_id, data=RTVIActionResponseData(result=result))
        await self._push_transport_message(message)

    async def _maybe_send_bot_ready(self):
        if self._pipeline_started and self._client_ready:
            await self._send_bot_ready()
            await self._update_config(self._config)

    async def _send_bot_ready(self):
        if not self._params.send_bot_ready:
            return

        message = RTVIBotReady(
            id=self._client_ready_id,
            data=RTVIBotReadyData(version=RTVI_PROTOCOL_VERSION, config=self._config.config_list),
        )
        await self._push_transport_message(message)

    async def _send_error_frame(self, frame: ErrorFrame):
        message = RTVIError(data=RTVIErrorData(error=frame.error, fatal=frame.fatal))
        await self._push_transport_message(message)

    async def _send_error_response(self, id: str, error: str):
        message = RTVIErrorResponse(id=id, data=RTVIErrorResponseData(error=error))
        await self._push_transport_message(message)

    def _action_id(self, service: str, action: str) -> str:
        return f"{service}:{action}"
