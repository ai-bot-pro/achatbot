import logging
import asyncio
from typing import AsyncGenerator
import uuid

from google.adk import Runner
from google.adk.events.event import Event as ADKEvent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event_actions import EventActions as ADKEventActions
from google.adk.agents import LiveRequestQueue
from google.genai import types
from a2a.types import Message, Role, Part, TextPart

from src.services.a2a_live.types import LiveMultiModalInputMessage, LiveMultiModalOutputMessage
from src.services.a2a_multiagents.adk_host_agent_manager import ADKHostAgentManager
from src.services.a2a_multiagents.types import Conversation


class ADKHostLiveAgentManager(ADKHostAgentManager):
    """
    - https://adk.wiki/streaming/ (BIDI)
    - https://ai.google.dev/gemini-api/docs/live
    An implementation of use adk host agent as the multi-agent manager with live BIDI stream.
    - supervisor_agent
    """

    def __init__(
        self,
        http_client,
        app_name,
        api_key="",
        mode="supervisor",
        system_prompt="",
        remote_agent_addresses=...,
        session_service=None,
        memory_service=None,
        artifact_service=None,
    ):
        super().__init__(
            http_client,
            app_name,
            api_key,
            mode,
            system_prompt,
            remote_agent_addresses,
            session_service,
            memory_service,
            artifact_service,
        )
        self._live_request_queue = LiveRequestQueue()
        self._runner_config: RunConfig | None = None

    def init_host_agent(self):
        if self.mode == "supervisor":
            from .adk_supervisor_agent import ADKSupervisorLiveAgent

            self._host_agent = ADKSupervisorLiveAgent(
                self.remote_agent_addresses, self.httpx_client, system_prompt=self.system_prompt
            )
        else:
            raise ValueError(f"Unsupported agent mode: {self.mode}")

    def initialize_host(self):
        self.init_host_agent()
        agent = self._host_agent.create_agent()
        self._host_runner = Runner(
            app_name=self.app_name,
            agent=agent,
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )

    async def create_conversation(self) -> Conversation:
        # Automatically determine response modality based on model architecture
        # Native audio models (containing "native-audio" in name) ONLY support AUDIO response modality
        # Half-cascade models support both TEXT and AUDIO, we default to TEXT for better performance
        model_name = self._host_agent.model
        is_native_audio = "native-audio" in model_name.lower()

        if is_native_audio:
            # Native audio models require AUDIO response modality with audio transcription
            response_modalities = ["AUDIO"]
            self._run_config = RunConfig(
                streaming_mode=StreamingMode.BIDI,
                response_modalities=response_modalities,
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                session_resumption=types.SessionResumptionConfig(),
            )
            logging.debug(
                f"Native audio model detected: {model_name}, using AUDIO response modality"
            )
        else:
            # Half-cascade models support TEXT response modality for faster performance
            response_modalities = ["TEXT"]
            self._run_config = RunConfig(
                streaming_mode=StreamingMode.BIDI,
                response_modalities=response_modalities,
                input_audio_transcription=None,
                output_audio_transcription=None,
                session_resumption=types.SessionResumptionConfig(),
            )
            logging.debug(
                f"Half-cascade model detected: {model_name}, using TEXT response modality"
            )
        logging.info(f"RunConfig created: {self._run_config}")

        return await super().create_conversation()

    async def send_live_message(self, message: LiveMultiModalInputMessage):
        """
        send LiveMultiModalInputMessage to live queue for async run_live
        """
        assert self._session.id == message.context_id
        # Update state must happen in an event
        state_update = {
            "task_id": message.task_id,
            "context_id": message.context_id,
            "message_id": message.message_id,
        }
        # Need to upsert session state now, only way is to append an event.
        event = ADKEvent(
            id=ADKEvent.new_id(),
            author="host_live_agent",
            invocation_id=ADKEvent.new_id(),
            actions=ADKEventActions(state_delta=state_update),
        )
        await self._session_service.append_event(self._session, event)

        if message.text_content is not None:
            self._live_request_queue.send_content(message.text_content)
        if message.audio_blob is not None:
            self._live_request_queue.send_realtime(message.audio_blob)
        if message.image_blob_list is not None:
            for image_blob in message.image_blob_list:
                self._live_request_queue.send_realtime(image_blob)

    async def recieve_message(self) -> AsyncGenerator[LiveMultiModalOutputMessage, None]:
        input_text = ""
        output_text = ""
        output_audio_chunk = bytes()
        context_id = self._session.id
        conversation = self.get_conversation(context_id)
        async for event in self._host_runner.run_live(
            user_id=self.user_id,
            session_id=self._session.id,
            live_request_queue=self._live_request_queue,
            run_config=self._run_config,
        ):
            if event.turn_complete:
                # input conversation text message
                response = Message(
                    message_id=str(uuid.uuid4()),
                    role="user",
                    parts=[
                        Part(root=TextPart(text=input_text)),
                    ],
                )
                conversation and conversation.messages.append(response)
                input_text = ""

                # output conversation text message
                response = Message(
                    message_id=str(uuid.uuid4()),
                    role="model",
                    parts=[
                        Part(root=TextPart(text=output_text)),
                    ],
                )
                conversation and conversation.messages.append(response)
                output_text = ""

                # output conversation audio message don't to save
                output_audio_chunk = bytes()

            if event.input_transcription and event.input_transcription.text:  # input text
                # print(f"{event.input_transcription.text=}")
                is_first_text = False
                if not input_text:
                    is_first_text = True
                event = LiveMultiModalOutputMessage(
                    context_id=context_id,
                    kind="transcription",
                    is_first_text=is_first_text,
                    text_content=types.Content(
                        role="user", parts=[types.Part(text=event.input_transcription.text)]
                    ),
                )
                input_text += event.input_transcription.text
                yield event
            elif event.output_transcription and event.output_transcription.text:  # output text
                # print(f"{event.output_transcription.text=}")
                is_first_text = False
                if not output_text:
                    is_first_text = True
                event = LiveMultiModalOutputMessage(
                    context_id=context_id,
                    kind="text",
                    is_first_text=is_first_text,
                    text_content=types.Content(
                        role="model", parts=[types.Part(text=event.output_transcription.text)]
                    ),
                )
                output_text += event.output_transcription.text
                yield event
            elif (
                event.content
                and event.content.parts
                and len(event.content.parts) > 0
                and event.content.parts[0].inline_data
                and event.content.parts[0].inline_data.mime_type
                and "audio/pcm" in event.content.parts[0].inline_data.mime_type
            ):  # output audio
                # print(
                #    f"audio: {event.content.parts[0].inline_data.mime_type} {len(event.content.parts[0].inline_data.data)=}"
                # )
                is_first_audio_chunk = False
                if not input_text:
                    is_first_text = True
                event = LiveMultiModalOutputMessage(
                    context_id=context_id,
                    kind="audio",
                    is_first_audio_chunk=is_first_audio_chunk,
                    audio_blob=types.Blob(
                        data=event.content.parts[0].inline_data.data,
                        mime_type=event.content.parts[0].inline_data.mime_type,
                    ),
                )
                output_audio_chunk += event.content.parts[0].inline_data.data
                yield event
