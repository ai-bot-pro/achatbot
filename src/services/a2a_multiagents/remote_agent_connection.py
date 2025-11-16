import logging
import json
from collections.abc import Callable

from a2a.client import (
    Client,
    ClientFactory,
)
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)


TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, client_factory: ClientFactory, agent_card: AgentCard):
        self.agent_client: Client = client_factory.create(agent_card)
        self.card: AgentCard = agent_card
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message: Message) -> Task | Message | None:
        lastTask: Task | None = None
        try:
            async for event in self.agent_client.send_message(message):
                if isinstance(event, Message):
                    return event
                if isinstance(event, tuple) and self.is_terminal_or_interrupted(event[0]):
                    return event[0]
                lastTask = event[0]
        except Exception as e:
            logging.exception("Exception found in send_message")
            raise e
        return lastTask

    def is_terminal_or_interrupted(self, task: Task) -> bool:
        return task.status.state in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.input_required,
            TaskState.unknown,
        ]


"""
python -m src.services.a2a_multiagents.remote_agent_connection
"""
if __name__ == "__main__":
    import asyncio
    import uuid

    import httpx
    from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
    from a2a.types import (
        AgentCard,
        TransportProtocol,
    )
    from a2a.utils.message import new_agent_text_message

    async def a_run():
        async with httpx.AsyncClient(timeout=60) as httpx_client:
            config = ClientConfig(
                httpx_client=httpx_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                    # TransportProtocol.grpc,
                ],
            )
            client_factory = ClientFactory(config)
            card_resolver = A2ACardResolver(httpx_client, "http://0.0.0.0:6666")
            card = await card_resolver.get_agent_card()
            conn = RemoteAgentConnections(client_factory, card)

            message = new_agent_text_message(
                text="what's a chatbot",
                context_id=str(uuid.uuid4()),
                # task_id=str(uuid.uuid4()),# new message no task_id
            )
            res = await conn.send_message(message)
            print(json.dumps(res.model_dump_json()))
            # print("res-->", res)
            await asyncio.sleep(1)

    asyncio.run(a_run())
