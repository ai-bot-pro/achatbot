import logging
from typing import Optional
import uuid

from apipeline.processors.async_frame_processor import AsyncFrameProcessor


class MemoryProcessor(AsyncFrameProcessor):
    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_id = user_id or f"user_{str(uuid.uuid4())}"
        self.agent_id = agent_id or f"agent_{str(uuid.uuid4())}"
        self.run_id = run_id or f"run_{str(uuid.uuid4())}"

    def set_user_id(self, user_id):
        self.user_id = user_id

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id

    def set_run_id(self, run_id):
        self.run_id = run_id

    async def get_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError("Subclasses must implement get_all_memories method")

    async def get_initial_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Fetch all memories for the user to create a personalized greeting with memories.

        Returns:
            A personalized greeting based on user memories
        """
        try:
            memories = await self.get_all_memories(
                user_id=user_id, agent_id=agent_id, run_id=run_id, **kwargs
            )

            if not memories or not memories.get("results"):
                logging.debug(
                    f"!!! No memories found for this user {user_id}, agent {agent_id}, run {run_id}"
                )
                return ""

            # Create a personalized greeting based on memories
            memory_str = ""
            # Add some personalization based on memories (limit to 3 memories for brevity)
            for i, memory in enumerate(memories["results"][:3]):
                memory_content = memory.get("memory", "")
                # Keep memory references brief
                if len(memory_content) > 100:
                    memory_content = memory_content[:97] + "..."
                memory_str += f"{memory_content} "

            logging.debug(
                f"get {len(memories)} memories for this user {user_id}, agent {agent_id}, run {run_id}"
            )
            return memory_str

        except Exception as e:
            logging.error(f"Error retrieving initial memories from Mem0: {e}")
            return ""
