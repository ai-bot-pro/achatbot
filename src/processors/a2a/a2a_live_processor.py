import os
import uuid
import asyncio
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from apipeline.processors.async_frame_processor import FrameDirection
from apipeline.frames import CancelFrame, StartFrame, EndFrame, Frame, TextFrame, ErrorFrame
from google.adk.artifacts import BaseArtifactService
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai import types

from src.processors.session_processor import SessionProcessor
from src.services.a2a_live.adk_host_agent_manager import ADKHostLiveAgentManager
from src.services.help.httpx import HTTPXClientWrapper
from src.common.session import Session
from src.common.utils.time import time_now_iso8601
from src.types.frames import (
    VisionImageVoiceRawFrame,
    AudioRawFrame,
    VisionImageRawFrame,
    TranscriptionFrame,
)
from src.services.a2a_live.types import LiveMultiModalInputMessage, LiveMultiModalOutputMessage


class A2ALiveProcessor(SessionProcessor):
    def __init__(
        self,
        app_name: str,
        host_agent_name: str = "",
        api_key: str = "",
        mode: str = "supervisor",
        model: str = "",
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
        self.manager = ADKHostLiveAgentManager(
            http_client,
            app_name,
            host_agent_name=host_agent_name,
            api_key=api_key,
            mode=mode,
            model=model,
            system_prompt=system_prompt,
            remote_agent_addresses=agent_urls,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        self.manager.initialize_host()
        self.push_task: Optional[asyncio.Task] = None
        self.conversation = None

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

    async def create_push_task(self):
        if self.push_task is not None:
            self.push_task.cancel()
            try:
                await self.push_task
            except asyncio.CancelledError:
                logging.info(f"{self.name} push_task cancelled.")
            self.push_task = None

        if self.push_task is None:
            self.push_task = self.get_event_loop().create_task(self._push_frames())
            logging.info(f"{self.name} push_task created.")

    async def start(self, frame: StartFrame):
        await self.create_conversation()
        await self.create_push_task()

        if self.http_client_wrapper is None:
            self.http_client_wrapper = HTTPXClientWrapper()
            self.http_client_wrapper.start()
            self.manager.set_client(self.http_client_wrapper())

        logging.info(f"{self.name} Conversation started")

    async def stop(self, frame: EndFrame):
        if self.http_client_wrapper:
            await self.http_client_wrapper.stop()
            self.http_client_wrapper = None

        logging.info(f"{self.name} Conversation end")

    async def cancel(self, frame: CancelFrame):
        self.push_task.cancel()
        try:
            await self.push_task
        except asyncio.CancelledError:
            logging.info(f"{self.name} push_task cancelled.")
        if self.http_client_wrapper:
            await self.http_client_wrapper.stop()
            self.http_client_wrapper = None
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
        elif isinstance(frame, TextFrame):
            await self.manager.send_live_message(
                LiveMultiModalInputMessage(
                    context_id=self.conversation.conversation_id,
                    message_id=str(uuid.uuid4()),
                    kind="text",
                    text_content=types.Content(role="user", parts=[types.Part(text=frame.text)]),
                )
            )
        elif isinstance(frame, AudioRawFrame):
            await self.manager.send_live_message(
                LiveMultiModalInputMessage(
                    context_id=self.conversation.conversation_id,
                    message_id=str(uuid.uuid4()),
                    kind="audio",
                    audio_blob=types.Blob(
                        data=frame.audio,
                        mime_type=f"audio/pcm;rate={frame.sample_rate}",
                    ),
                )
            )
        elif isinstance(frame, VisionImageRawFrame):
            await self.manager.send_live_message(
                LiveMultiModalInputMessage(
                    context_id=self.conversation.conversation_id,
                    message_id=str(uuid.uuid4()),
                    kind="text_images",
                    text_content=types.Content(role="user", parts=[types.Part(text=frame.text)]),
                    image_blob_list=[
                        types.Blob(
                            data=frame.image,
                            mime_type=f"image/{frame.format.lower()}",
                        )
                    ],
                )
            )
        elif isinstance(frame, VisionImageVoiceRawFrame):
            await self.manager.send_live_message(
                LiveMultiModalInputMessage(
                    context_id=self.conversation.conversation_id,
                    message_id=str(uuid.uuid4()),
                    kind="audio_images",
                    audio_blob=types.Blob(
                        data=frame.audio.audio,
                        mime_type=f"audio/pcm;rate={frame.audio.sample_rate}",
                    ),
                    image_blob_list=[
                        types.Blob(
                            data=img.image,
                            mime_type=f"image/{img.format.lower()}",
                        )
                        for img in frame.images
                    ],
                )
            )
        else:
            await self.queue_frame(frame, direction)


    async def _push_frames(self):
        try:
            async for msg in self.manager.recieve_message():
                if msg.kind == "transcription":
                    await self.queue_frame(
                        TranscriptionFrame(
                            text=msg.text_content.parts[0].text,
                            user_id=self.session.ctx.client_id,
                            timestamp=time_now_iso8601(),
                        )
                    )
                if msg.kind == "text":
                    await self.queue_frame(TextFrame(text=msg.text_content.parts[0].text))
                if msg.kind == "audio":
                    rate = int(msg.audio_blob.mime_type.split("=")[1])
                    await self.queue_frame(
                        AudioRawFrame(audio=msg.audio_blob.data, sample_rate=rate)
                    )
                if (
                    msg.kind == "images"
                ):  # gemini live 2.5 no image , wait gemini live 3.0 (maybe gen image)
                    # TODO: add image frame support
                    pass
        except Exception as e:
            logging.exception(f"{self.name} Error in _push_frames: {e}")
