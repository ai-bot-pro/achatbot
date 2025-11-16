import asyncio
import base64
import json
import os
import logging

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    DataPart,
    Part,
    TransportProtocol,
)
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback
from .timestamp_ext import TimestampExtension

DEFAULT_LITELLM_MODEL = os.getenv("ADK_LITELLM_MODEL", "gemini/gemini-2.5-flash")


class BaseHostAgent:
    """The base host agent.

    This is the agent responsible for choosing which remote agents

    use http sse for streaming response from remote agents
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        model: str = DEFAULT_LITELLM_MODEL,
        system_prompt: str = "",
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.task_callback = task_callback  # todo
        self.httpx_client = http_client
        self.timestamp_extension = TimestampExtension()
        config = ClientConfig(
            httpx_client=self.httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
                # TransportProtocol.grpc,
            ],
        )
        client_factory = ClientFactory(config)
        client_factory = self.timestamp_extension.wrap_client_factory(client_factory)
        self.client_factory = client_factory
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self.loop = loop or asyncio.get_running_loop()
        self.init_task_done = asyncio.Event()
        init_task = self.loop.create_task(self.init_remote_agent_addresses(remote_agent_addresses))
        init_task.add_done_callback(self._handle_init_exception)

    def _handle_init_exception(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as e:
            logging.error(f"Initialization task failed with exception: {e}", exc_info=True)

    async def init_remote_agent_addresses(self, remote_agent_addresses: list[str]):
        async with asyncio.TaskGroup() as task_group:
            for address in remote_agent_addresses:
                task_group.create_task(self.retrieve_card(address))
        # The task groups run in the background and complete.
        # Once completed the self.agents string is set and the remote
        # connections are established.
        self.init_task_done.set()

    async def wait_init(self):
        await self.init_task_done.wait()
        self.init_task_done.clear()

    async def retrieve_card(self, address: str):
        card_resolver = A2ACardResolver(self.httpx_client, address)
        card = await card_resolver.get_agent_card()
        self.register_agent_card(card)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(self.client_factory, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = "\n".join(agent_info)
        logging.info(f"registered agents {self.agents}")

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append({"name": card.name, "description": card.description})
        return remote_agent_info

    def create_agent(self) -> Agent:
        pass

    def root_instruction(self, context: ReadonlyContext) -> str:
        pass

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            "context_id" in state
            and "session_active" in state
            and state["session_active"]
            and "agent" in state
        ):
            return {"active_agent": f"{state['agent']}"}
        return {"active_agent": "None"}

    def set_client(self, httpx_client: httpx.AsyncClient):
        self.httpx_client = httpx_client


async def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext):
    if part.root.kind == "text":
        return part.root.text
    if part.root.kind == "data":
        return part.root.data
    if part.root.kind == "file":
        # Repackage A2A FilePart to google.genai Blob
        # Currently not considering plain text as files
        file_id = part.root.file.name
        file_bytes = base64.b64decode(part.root.file.bytes)
        file_part = types.Part(
            inline_data=types.Blob(mime_type=part.root.file.mime_type, data=file_bytes)
        )
        await tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        return DataPart(data={"artifact-file-id": file_id})
    return f"Unknown type: {part.kind}"
