import os
import uuid
import asyncio
import logging
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from apipeline.processors.async_frame_processor import FrameDirection
from apipeline.frames import CancelFrame, StartFrame, EndFrame, Frame, TextFrame, ErrorFrame
from a2a.utils.message import new_agent_text_message
from google.adk.events.event import Event as ADKEvent
from google.adk.artifacts import BaseArtifactService
from google.adk.memory.in_memory_memory_service import BaseMemoryService
from google.adk.sessions.in_memory_session_service import BaseSessionService

from src.processors.session_processor import SessionProcessor
from src.services.a2a_multiagents.adk_host_agent_manager import ADKHostAgentManager
from src.services.help.httpx import HTTPXClientWrapper
from src.common.session import Session
from src.types.frames import LLMMessagesFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)


class A2AConversationProcessor(SessionProcessor):
    def __init__(
        self,
        app_name: str,
        api_key: str = "",
        mode: str = "supervisor",  # supervisor or planer
        system_prompt: str = "",
        http_timeout: float = 30.0,
        agent_urls: Optional[list[str]] = [],  # agent http url
        max_workers: int = 0,
        session: Session | None = None,
        session_service: BaseSessionService | None = None,
        memory_service: BaseMemoryService | None = None,
        artifact_service: BaseArtifactService | None = None,
        **kwargs,
    ):
        super().__init__(session=session, **kwargs)
        self.http_client_wrapper = HTTPXClientWrapper()
        self.http_client_wrapper.start(timeout=http_timeout)
        http_client = self.http_client_wrapper()
        self.manager = ADKHostAgentManager(
            http_client,
            app_name,
            api_key=api_key,
            mode=mode,
            remote_agent_addresses=agent_urls,
            system_prompt=system_prompt,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        self.queue = asyncio.Queue()
        self.push_task: Optional[asyncio.Task] = None
        self.running = False
        self.conversation = None

        max_workers = max_workers or os.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def host_agent_manager(self):
        return self.manager

    def set_user_id(self, user_id: str):
        self.manager.set_user_id(user_id)

    async def create_conversation(self):
        self.conversation = await self.manager.create_conversation()

    async def register_agents(self, urls: List[str]):
        for url in urls:
            await self.manager.register_agent(url)

    async def start(self, frame: StartFrame):
        await self.create_conversation()
        self.running = True
        self.push_task = self.get_event_loop().create_task(self._push_frames())

        if self.http_client_wrapper is None:
            self.http_client_wrapper = HTTPXClientWrapper()
            self.http_client_wrapper.start()
            self.manager.set_client(self.http_client_wrapper())

        logging.info(f"{self.name} Conversation started")

    async def stop(self, frame: EndFrame):
        self.running = False
        if self.http_client_wrapper:
            self.http_client_wrapper.stop()
            self.http_client_wrapper = None

        logging.info(f"{self.name} Conversation end")

    async def cancel(self, frame: CancelFrame):
        self.push_task.cancel()
        try:
            await self.push_task
        except asyncio.CancelledError:
            logging.info(f"{self.name} push_task cancelled.")
        if self.http_client_wrapper:
            self.http_client_wrapper.stop()
            self.http_client_wrapper = None
        self.executor.shutdown(wait=True)
        logging.info(f"{self.name} Conversation cancelled")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, intercept context frames for memory integration.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        else:
            await self.queue_frame(frame, direction)

        context = None
        messages = None
        if isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = OpenAILLMContext(messages=messages)
        elif isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        if not context:
            return
        # Get the latest user message to use as a query for a2a agents
        context_messages = context.get_messages()
        for message in reversed(context_messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                query: str = message.get("content")
                await self.send_message(query)
                break

    async def send_message(self, text: str):
        try:
            message = new_agent_text_message(
                text=text,
                context_id=self.conversation.conversation_id,
                # task_id=str(uuid.uuid4()),# new message no task_id
            )
            message = self.manager.sanitize_message(message)
            # print("message-->", message)
            # Send the message in a separate thread to avoid blocking the event loop.
            self.executor.submit(
                self.manager.process_message_threadsafe, message, self.get_event_loop(), self.queue
            )
            # t = threading.Thread(
            #    target=lambda: self.manager.process_message_threadsafe(
            #        message, self.get_event_loop(), self.queue
            #    )
            # )
            # t.start()
        except Exception as e:
            logging.error(f"Error processing: {str(e)}")
            await self.push_frame(ErrorFrame(f"Error processing: {str(e)}"))

    async def _push_frames(self):
        while self.running:
            try:
                event: ADKEvent = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                # print("event-->", event)
                # Send the message in a separate thread to avoid blocking the event loop.
                if event.id and event.partial is None:
                    await self.queue_frame(LLMFullResponseStartFrame())
                if event.content and event.content.parts and event.partial:
                    for part in event.content.parts:
                        if part.text:
                            await self.queue_frame(
                                TextFrame(text=part.text), FrameDirection.DOWNSTREAM
                            )
                if event.content and event.content.parts and event.partial is False:
                    await self.queue_frame(LLMFullResponseEndFrame())

                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logging.info(f"{self.name} _push_frames cancelled")
                break
            except Exception as ex:
                logging.exception(f"{self.name} Unexpected error in _push_frames: {ex}")
                if self.get_event_loop().is_closed():
                    logging.warning(f"{self.name} event loop is closed")
                    break
