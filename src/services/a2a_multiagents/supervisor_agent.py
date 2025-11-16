import base64
import uuid
import logging

from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

from src.services.a2a_multiagents import BaseHostAgent
from . import convert_parts


class SupervisorAgent(BaseHostAgent):
    """The supervisor agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate/summary or directly return their work.

    like langchain/langgraph multi-agent supervisor agent:
    - https://github.com/weedge/doraemon-nb/blob/main/langchain/agent_supervisor.ipynb
    - https://github.com/weedge/doraemon-nb/blob/main/langchain/hierarchical_agent_teams.ipynb

    use http sse for streaming response from remote agents
    """

    def create_agent(self) -> Agent:
        return Agent(
            model=LiteLlm(model=self.model),
            name="supervisor_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This agent orchestrates the decomposition of the user request into"
                " tasks that can be performed by the child agents."
            ),
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_state(context)
        prompt = (
            self.system_prompt
            or f"""You are an expert delegator that can delegate the user request to the
appropriate remote agents.

Execution:
- For actionable requests, you can use `send_message` to interact with remote agents to take action.

Be sure to include the remote agent name when you respond to the user.

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily. It is not necessary to disclose which agent the information was obtained from.

Your output will be converted to audio, so please do not include special characters in your response.
Respond to what the user says in a creative and helpful way. 
Don't over-explain what you're doing. When making tool calls, respond with only short sentences.
"""
        )
        instruction = f"""{prompt}

Agents:
{self.agents}

Current agent: {current_agent["active_agent"]}
"""
        logging.debug(f"{instruction=}")
        return instruction

    async def send_message(self, agent_name: str, message: str, tool_context: ToolContext):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.
          tool_context: The tool context this method runs in.

        Yields:
          A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
        state["agent"] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        task_id = state.get("task_id", None)
        context_id = state.get("context_id", None)
        message_id = state.get("message_id", None)
        task: Task
        if not message_id:
            message_id = str(uuid.uuid4())

        request_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )
        response = await client.send_message(request_message)
        if isinstance(response, Message):
            response = await convert_parts(response.parts, tool_context)
            logging.info(f"send_message response:{response}")
            return response
        task: Task = response
        # Assume completion unless a state returns that isn't complete
        state["session_active"] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if task.context_id:
            state["context_id"] = task.context_id
        state["task_id"] = task.id
        if task.status.state == TaskState.input_required:
            # Force user input back
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f"Agent {agent_name} task {task.id} is cancelled")
        elif task.status.state == TaskState.failed:
            # Raise error for failure
            raise ValueError(f"Agent {agent_name} task {task.id} failed")
        response = []
        if task.status.message:
            # Assume the information is in the task message.
            if ts := self.timestamp_extension.get_timestamp(task.status.message):
                response.append(f"[at {ts.astimezone().isoformat()}]")
            response.extend(await convert_parts(task.status.message.parts, tool_context))
        if task.artifacts:
            for artifact in task.artifacts:
                if ts := self.timestamp_extension.get_timestamp(artifact):
                    response.append(f"[at {ts.astimezone().isoformat()}]")
                response.extend(await convert_parts(artifact.parts, tool_context))
        logging.info(f"send_message response:{response}")
        return response

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            state["session_active"] = True


"""
LOG_LEVEL=debug python -m src.services.a2a_multiagents.supervisor_agent
python -m src.services.a2a_multiagents.supervisor_agent
"""
if __name__ == "__main__":
    import os
    import asyncio
    import uuid

    from google.adk import Runner
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.genai import types

    from src.services.help.httpx import HTTPXClientWrapper
    from src.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    async def arun():
        http_client_wrapper = HTTPXClientWrapper()
        http_client_wrapper.start(timeout=60)
        host_agent = SupervisorAgent(
            ["http://0.0.0.0:6666"], http_client_wrapper()
        )  # github repo describe is long-time task
        await host_agent.wait_init()  # sync init

        agent = host_agent.create_agent()

        session_service = InMemorySessionService()
        app_name = "A2A_app"
        user_id = "user1"
        host_runner = Runner(
            app_name=app_name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=session_service,
            memory_service=InMemoryMemoryService(),
        )
        session = await session_service.create_session(app_name=app_name, user_id=user_id)
        async for event in host_runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                parts=[types.Part.from_text(text="what's a chatbot?")], role="user"
            ),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            print("event-->", event)

        # await http_client_wrapper.stop()

    asyncio.run(arun())
