"""
use mem0(RAG) for LTM
"""

import os
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from apipeline.processors.frame_processor import FrameDirection
from apipeline.frames import Frame, ErrorFrame

from src.processors.context.memory import MemoryProcessor
from src.types.frames import LLMMessagesFrame
from src.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)


try:
    from mem0 import AsyncMemory, AsyncMemoryClient
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Mem0, you need to `pip install achatbot[mem0]`. Also, set the environment variable MEM0_API_KEY."
    )
    raise Exception(f"Missing module: {e}")


class InputParams(BaseModel):
    """Configuration parameters for Mem0 memory service.

    Parameters:
        search_limit: Maximum number of memories to retrieve per query.
        search_threshold: Minimum similarity threshold for memory retrieval.
        api_version: API version to use for Mem0 client operations.
        system_prompt: Prefix text for memory context messages.
        add_as_system_message: Whether to add memories as system messages.
        position: Position to insert memory messages in context.
    """

    search_limit: int = Field(default=10, ge=1)
    search_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    api_version: str = Field(default="v2")
    system_prompt: str = Field(default="Based on previous conversations, I recall: \n\n")
    add_as_system_message: bool = Field(default=True)
    position: int = Field(default=1)


class Mem0MemoryProcessor(MemoryProcessor):
    """A standalone memory service that integrates with Mem0.

    This service intercepts message frames in the pipeline, stores them in Mem0,
    and enhances context with relevant memories before passing them downstream.
    Supports both local and cloud-based Mem0 configurations.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        host: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Mem0 memory service.

        Args:
            api_key: The API key for accessing Mem0's cloud API.
            local_config: Local configuration for Mem0 client (alternative to cloud API).
            user_id: The user ID to associate with memories in Mem0.
            agent_id: The agent ID to associate with memories in Mem0.
            run_id: The run ID to associate with memories in Mem0.
            params: Configuration parameters for memory retrieval and storage.
            host: The host of the Mem0 server.

        Raises:
            ValueError: If none of user_id, agent_id, or run_id are provided.
        """
        super().__init__(user_id=user_id, agent_id=agent_id, run_id=run_id, **kwargs)

        local_config = local_config or {}
        params = InputParams(**kwargs)

        if local_config:
            # DIY with advance RAG(embd,ranker,vector db, graph db) locally, maybe use agentic RAG
            self.memory_client = self._loop.run_until_complete(
                AsyncMemory.from_config(local_config)
            )
        else:
            # Use Mem0 cloud service with dashboard
            api_key = api_key or os.getenv("MEM0_API_KEY")
            self.memory_client = AsyncMemoryClient(api_key=api_key, host=host)

        self.search_limit = params.search_limit
        self.search_threshold = params.search_threshold
        self.api_version = params.api_version
        self.system_prompt = params.system_prompt
        self.add_as_system_message = params.add_as_system_message
        self.position = params.position
        self.last_query = None
        logging.info(f"Initialized Mem0MemoryProcessor with {user_id=}, {agent_id=}, {run_id=}")

    async def get_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(
            self.memory_client, AsyncMemory
        ):  # DIY with advance RAG(embd,ranker,vector db, graph db) locally, maybe use agentic RAG
            filters = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
            filters = {k: v for k, v in filters.items() if v is not None}
            memories = await self.memory_client.get_all(**filters)
            return memories

        # Use Mem0 cloud service with dashboard
        id_pairs = [
            # Create filters based on available IDs
            ("user_id", user_id),
            ("agent_id", agent_id),
            ("run_id", run_id),
        ]
        clauses = [{name: value} for name, value in id_pairs if value is not None]
        filters = {"AND": clauses} if clauses else {}
        # Get all memories for this user
        memories = await self.memory_client.get_all(
            filters=filters, version="v2", output_format="v1.1"
        )
        return memories

    async def _store_messages(self, messages: List[Dict[str, Any]]):
        """Store messages in Mem0.

        Args:
            messages: List of message dictionaries to store in memory.
        """
        try:
            logging.debug(f"Storing {len(messages)} messages in Mem0")
            params = {
                "async_mode": True,
                "metadata": {"platform": "achatbot"},
                # "output_format": "v1.1",
            }
            for key in ["user_id", "agent_id", "run_id"]:
                if getattr(self, key):
                    params[key] = getattr(self, key)

            if isinstance(self.memory_client, AsyncMemory):
                del params["async_mode"]
            # run this in background thread pool to avoid blocking the conversation
            await self.memory_client.add(messages=messages, **params)
        except Exception as e:
            logging.error(f"Error storing messages in Mem0: {e}")

    async def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0.

        Args:
            query: The query to search for relevant memories.

        Returns:
            List of relevant memory dictionaries matching the query.
        """
        try:
            memories = []
            logging.debug(f"Retrieving memories for query: {query}")
            if isinstance(self.memory_client, AsyncMemory):
                params = {
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "run_id": self.run_id,
                    "limit": self.search_limit,
                }
                params = {k: v for k, v in params.items() if v is not None}
                res = await self.memory_client.search(query=query, filters=None, **params)
                memories = res.get("results", [])

            else:
                id_pairs = [
                    ("user_id", self.user_id),
                    ("agent_id", self.agent_id),
                    ("run_id", self.run_id),
                ]
                clauses = [{name: value} for name, value in id_pairs if value is not None]
                filters = {"OR": clauses} if clauses else {}
                res = await self.memory_client.search(
                    query=query,
                    filters=filters,
                    version=self.api_version,
                    top_k=self.search_limit,
                    threshold=self.search_threshold,
                    output_format="v1.1",
                )
                memories = res.get("results", [])

            logging.debug(f"Retrieved {len(memories)} memories from Mem0 {query} {filters}")
            return memories
        except Exception as e:
            logging.error(f"Error retrieving memories from Mem0: {e}")
            return []

    async def _enhance_context_with_memories(self, context: OpenAILLMContext, query: str):
        """Enhance the LLM context with relevant memories.

        Args:
            context: The LLM context to enhance with memory information.
            query: The query to search for relevant memories.
        """
        # Skip if this is the same query we just processed
        if self.last_query == query:
            return

        self.last_query = query

        memories = await self._retrieve_memories(query)
        logging.debug(f"Enhanced context with memories {memories}")
        if not memories:
            return

        # Format memories as a message
        memory_text = self.system_prompt
        for i, memory in enumerate(memories):
            memory_text += f"{i}. {memory.get('memory', '')}\n\n"

        # Add memories as a system message or user message based on configuration
        if self.add_as_system_message:
            context.add_message({"role": "system", "content": memory_text})
        else:
            # Add as a user message that provides context
            context.add_message({"role": "user", "content": memory_text})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, intercept context frames for memory integration.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        context = None
        messages = None

        if isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = OpenAILLMContext(messages=messages)
        elif isinstance(frame, OpenAILLMContextFrame):
            context = frame.context

        if context:
            try:
                # Get the latest user message to use as a query for memory retrieval
                context_messages = context.get_messages()
                latest_user_message = None

                for message in reversed(context_messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        latest_user_message = message.get("content")
                        break

                if latest_user_message:
                    # Enhance context with memories before passing it downstream
                    await self._enhance_context_with_memories(context, latest_user_message)
                    # Store the conversation in Mem0. Only call this when user message is detected
                    await self._store_messages(context_messages)

                # If we received an LLMMessagesFrame, create a new one with the enhanced messages
                if messages is not None:
                    await self.queue_frame(LLMMessagesFrame(context.get_messages()))
                else:
                    # Otherwise, pass the enhanced context frame downstream
                    await self.push_frame(frame)
            except Exception as e:
                logging.error(f"Error processing with Mem0: {str(e)}")
                await self.push_frame(ErrorFrame(f"Error processing with Mem0: {str(e)}"))
                await self.queue_frame(frame)  # Still pass the original frame through
        else:
            # For non-context frames, just pass them through
            await self.queue_frame(frame, direction)
