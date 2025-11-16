import click
import uvicorn

from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    GetTaskRequest,
    GetTaskResponse,
    SendMessageRequest,
    SendMessageResponse,
)

from .agent_executor import GitMcpAgentExecutor


class A2ARequestHandler(DefaultRequestHandler):
    """A2A Request Handler for the achatbot Repo Agent."""

    def __init__(self, agent_executor: AgentExecutor, task_store: InMemoryTaskStore):
        super().__init__(agent_executor, task_store)

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        return await super().on_get_task(request)

    async def on_message_send(self, request: SendMessageRequest) -> SendMessageResponse:
        return await super().on_message_send(request)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=6666)
def main(host: str, port: int):
    """Start the achatbot Repo Agent server.

    This function initializes the achatbot Repo Agent server with the specified host and port.
    It creates an agent card with the agent's name, description, version, and capabilities.

    Args:
        host (str): The host address to run the server on.
        port (int): The port number to run the server on.
    """
    skill = AgentSkill(
        id="answer_detail_about_achatbot_repo",
        name="Answer any information about a chatbot repo",
        description="The agent will look up the information about a chatbot repo and answer the question.",
        tags=["achatbot repo"],
        examples=["What is a chatbot repo?"],
    )

    agent_card = AgentCard(
        name="achatbot Agent",
        description="a chatbot knowledge agent who has information about a chatbot project and can answer questions about it",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            input_modes=["text"],
            output_modes=["text"],
            streaming=True,
        ),
        skills=[skill],
        # authentication=AgentAuthentication(schemes=['public']),
        examples=["What is achatbot?"],
    )

    task_store = InMemoryTaskStore()
    request_handler = A2ARequestHandler(
        agent_executor=GitMcpAgentExecutor(),
        task_store=task_store,
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    uvicorn.run(server.build(), host=host, port=port)


"""
python -m a2a_server.cmd.mcp_github
"""
if __name__ == "__main__":
    main()
