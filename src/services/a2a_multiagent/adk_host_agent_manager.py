import logging
import asyncio
import base64
import datetime
import json
import os
import uuid

import httpx
from a2a.types import (
    AgentCard,
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
)
from google.adk import Runner
from google.adk.events.event import Event as ADKEvent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event_actions import EventActions as ADKEventActions
from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import BaseMemoryService, InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import BaseSessionService, InMemorySessionService
from google.genai import types

from .types import Conversation, Event
from src.services.help.a2a.agent_card import get_agent_card, async_get_agent_card


class ADKHostAgentManager:
    """
    An implementation of use adk host agent as the multi-agent manager.
    - planer_agent
    - supervisor_agent
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        app_name: str,
        api_key: str = "",
        mode: str = "supervisor",  # supervisor or planer
        system_prompt: str = "",
        remote_agent_addresses: list[str] = [],
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
        artifact_service: BaseArtifactService | None = None,
    ):
        self._conversations: list[Conversation] = []
        self._tasks: list[Task] = []  # TODO for long-time run task
        self._events: dict[str, Event] = {}
        self._agents: list[AgentCard] = []
        self._session_service = session_service or InMemorySessionService()
        self._memory_service = memory_service or InMemoryMemoryService()
        self._artifact_service = artifact_service or InMemoryArtifactService()
        self._host_agent = None
        if mode == "planer":
            from .planer_agent import PlanerAgent

            self._host_agent = PlanerAgent(
                remote_agent_addresses, http_client, system_prompt=system_prompt
            )
        elif mode == "supervisor":
            from .supervisor_agent import SupervisorAgent

            self._host_agent = SupervisorAgent(
                remote_agent_addresses, http_client, system_prompt=system_prompt
            )
        else:
            raise ValueError(f"Unsupported agent mode: {mode}")
        self.httpx_client = http_client

        self.user_id = None
        self.app_name = app_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        self._initialize_host()

    def _initialize_host(self):
        agent = self._host_agent.create_agent()
        self._host_runner = Runner(
            app_name=self.app_name,
            agent=agent,
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )

    def set_client(self, httpx_client: httpx.AsyncClient):
        self._host_agent.set_client(httpx_client)

    def set_user_id(self, user_id: str):
        self.user_id = user_id

    async def create_conversation(self) -> Conversation:
        if self.user_id is None:
            logging.warning("no user_id")
            return
        session = await self._session_service.create_session(
            app_name=self.app_name, user_id=self.user_id
        )
        conversation_id = session.id
        c = Conversation(conversation_id=conversation_id, is_active=True)
        self._conversations.append(c)
        return c

    def update_api_key(self, api_key: str):
        """Update the API key and reinitialize the host if needed"""
        if api_key and api_key != self.api_key:
            self.api_key = api_key

            # Reinitialize host with new API key
            self._initialize_host()

    def sanitize_message(self, message: Message) -> Message:
        if message.context_id:
            conversation = self.get_conversation(message.context_id)
            if not conversation:
                return message
            # Check if the last event in the conversation was tied to a task.
            if conversation.messages:
                task_id = conversation.messages[-1].task_id
                if task_id and task_still_open(
                    next(
                        filter(lambda x: x and x.id == task_id, self._tasks),
                        None,
                    )
                ):
                    message.task_id = task_id
        return message

    async def process_message(self, message: Message, queue: asyncio.Queue = None):
        user_id = self.user_id
        context_id = message.context_id
        conversation = self.get_conversation(context_id)
        conversation and conversation.messages.append(message)
        self.add_event(
            Event(
                id=str(uuid.uuid4()),
                actor="user",
                content=message,
                timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            )
        )
        # Determine if a task is to be resumed.
        session = await self._session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=context_id
        )
        # print(session)
        task_id = message.task_id
        # Update state must happen in an event
        state_update = {
            "task_id": task_id,
            "context_id": context_id,
            "message_id": message.message_id,
        }
        # Need to upsert session state now, only way is to append an event.
        event = ADKEvent(
            id=ADKEvent.new_id(),
            author="host_agent",
            invocation_id=ADKEvent.new_id(),
            actions=ADKEventActions(state_delta=state_update),
        )
        queue and await queue.put(event)
        await self._session_service.append_event(session, event)

        msg = self.adk_content_from_message(message)
        # print("msg->", msg)
        final_event = None
        async for event in self._host_runner.run_async(
            user_id=user_id,
            session_id=context_id,
            new_message=msg,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            # print("event-->", event)
            if event.actions.state_delta and "task_id" in event.actions.state_delta:
                task_id = event.actions.state_delta["task_id"]
            self.add_event(
                Event(
                    id=event.id,
                    actor=event.author,
                    content=await self.adk_content_to_message(event.content, context_id, task_id),
                    timestamp=event.timestamp,
                )
            )
            final_event = event
            queue and await queue.put(event)

        response: Message | None = None
        if final_event:
            if final_event.actions.state_delta and "task_id" in final_event.actions.state_delta:
                task_id = final_event.actions.state_delta["task_id"]
            final_event.content.role = "model"
            response = await self.adk_content_to_message(final_event.content, context_id, task_id)
            # self._memory_service.add_session_to_memory(session)

        if conversation and response:
            conversation.messages.append(response)

    def add_event(self, event: Event):
        self._events[event.id] = event

    def get_conversation(self, conversation_id: str | None) -> Conversation | None:
        if not conversation_id:
            return None
        return next(
            filter(
                lambda c: c and c.conversation_id == conversation_id,
                self._conversations,
            ),
            None,
        )

    async def register_agent(self, url):
        agent_data = await async_get_agent_card(url, self.httpx_client)
        if not agent_data.url:
            agent_data.url = url
        self._agents.append(agent_data)
        self._host_agent.register_agent_card(agent_data)
        # Now update the host agent definition
        self._initialize_host()

    @property
    def agents(self) -> list[AgentCard]:
        return self._agents

    @property
    def conversations(self) -> list[Conversation]:
        return self._conversations

    @property
    def events(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda x: x.timestamp)

    def adk_content_from_message(self, message: Message) -> types.Content:
        parts: list[types.Part] = []
        for p in message.parts:
            part = p.root
            if part.kind == "text":
                parts.append(types.Part.from_text(text=part.text))
            elif part.kind == "data":
                json_string = json.dumps(part.data)
                parts.append(types.Part.from_text(text=json_string))
            elif part.kind == "file":
                if isinstance(part.file, FileWithUri):
                    parts.append(
                        types.Part.from_uri(
                            file_uri=part.file.uri,
                            mime_type=part.file.mime_type,
                        )
                    )
                else:
                    parts.append(
                        types.Part.from_bytes(
                            data=part.file.bytes.encode("utf-8"),
                            mime_type=part.file.mime_type,
                        )
                    )
        return types.Content(parts=parts, role=message.role)

    async def adk_content_to_message(
        self,
        content: types.Content,
        context_id: str | None,
        task_id: str | None,
    ) -> Message:
        parts: list[Part] = []
        if not content.parts:
            return Message(
                parts=[],
                role=content.role if content.role == Role.user else Role.agent,
                context_id=context_id,
                task_id=task_id,
                message_id=str(uuid.uuid4()),
            )
        for part in content.parts:
            if part.text:
                # try parse as data
                try:
                    data = json.loads(part.text)
                    parts.append(Part(root=DataPart(data=data)))
                except json.JSONDecodeError:
                    parts.append(Part(root=TextPart(text=part.text)))
            elif part.inline_data:
                parts.append(
                    Part(
                        root=FilePart(
                            file=FileWithBytes(
                                bytes=part.inline_data.decode("utf-8"),
                                mime_type=part.file_data.mime_type,
                            ),
                        )
                    )
                )
            elif part.file_data:
                parts.append(
                    Part(
                        root=FilePart(
                            file=FileWithUri(
                                uri=part.file_data.file_uri,
                                mime_type=part.file_data.mime_type,
                            )
                        )
                    )
                )
            # These aren't managed by the A2A message structure, these are internal
            # details of ADK, we will simply flatten these to json representations.
            elif part.video_metadata:
                parts.append(Part(root=DataPart(data=part.video_metadata.model_dump())))
            elif part.thought:
                parts.append(Part(root=TextPart(text="thought")))
            elif part.executable_code:
                parts.append(Part(root=DataPart(data=part.executable_code.model_dump())))
            elif part.function_call:
                parts.append(Part(root=DataPart(data=part.function_call.model_dump())))
            elif part.function_response:
                parts.extend(await self._handle_function_response(part, context_id, task_id))
            else:
                raise ValueError("Unexpected content, unknown type")
        return Message(
            role=content.role if content.role == Role.user else Role.agent,
            parts=parts,
            context_id=context_id,
            task_id=task_id,
            message_id=str(uuid.uuid4()),
        )

    async def _handle_function_response(
        self, part: types.Part, context_id: str | None, task_id: str | None
    ) -> list[Part]:
        parts = []
        try:
            for p in part.function_response.response["result"]:
                if isinstance(p, str):
                    parts.append(Part(root=TextPart(text=p)))
                elif isinstance(p, dict):
                    if "kind" in p and p["kind"] == "file":
                        parts.append(Part(root=FilePart(**p)))
                    else:
                        parts.append(Part(root=DataPart(data=p)))
                elif isinstance(p, DataPart):
                    if "artifact-file-id" in p.data:
                        file_part = await self._artifact_service.load_artifact(
                            user_id=self.user_id,
                            session_id=context_id,
                            app_name=self.app_name,
                            filename=p.data["artifact-file-id"],
                        )
                        file_data = file_part.inline_data
                        base64_data = base64.b64encode(file_data.data).decode("utf-8")
                        parts.append(
                            Part(
                                root=FilePart(
                                    file=FileWithBytes(
                                        bytes=base64_data,
                                        mime_type=file_data.mime_type,
                                        name="artifact_file",
                                    )
                                )
                            )
                        )
                    else:
                        parts.append(Part(root=DataPart(data=p.data)))
                else:
                    parts.append(Part(root=TextPart(text="Unknown content")))
        except Exception as e:
            logging.error(f"Couldn't convert to messages:{e}")
            parts.append(Part(root=DataPart(data=part.function_response.model_dump())))
        return parts

    def process_message_threadsafe(
        self, message: Message, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue = None
    ):
        """Safely run process_message from a thread using the given event loop."""
        future = asyncio.run_coroutine_threadsafe(self.process_message(message, queue), loop)
        return future  # You can call future.result() to get the result if needed


def get_message_id(m: Message | None) -> str | None:
    if not m or not m.metadata or "message_id" not in m.metadata:
        return None
    return m.metadata["message_id"]


def task_still_open(task: Task | None) -> bool:
    if not task:
        return False
    return task.status.state in [
        TaskState.submitted,
        TaskState.working,
        TaskState.input_required,
    ]
