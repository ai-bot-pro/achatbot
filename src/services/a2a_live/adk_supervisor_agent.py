import os
import logging
import uuid

from pathlib import Path
from jinja2 import Template
from a2a.types import (
    Message,
    Role,
    Task,
    Part,
    TaskState,
    TextPart,
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import LiveRequestQueue

from src.services.a2a_multiagents import ADKBaseHostAgent, convert_parts


class ADKSupervisorLiveAgent(ADKBaseHostAgent):
    def create_agent(self) -> Agent:
        tools = [] if len(self.remote_agent_connections) == 0 else [self.send_message]
        return Agent(
            model=self.model,
            name=self.name or "supervisor_live_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This agent orchestrates the decomposition of the user request into"
                " tasks that can be performed by the child agents."
            ),
            tools=tools,
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

        instruction = prompt
        dir_path = Path(__file__).parent
        with Path(dir_path / "tpl" / "supervisor.jinja").open("r") as f:
            instruction_template = Template(f.read())
            print(f"{current_agent=}")
            instruction = instruction_template.render(
                prompt=prompt, agents=self.agents, current_agent=current_agent["active_agent"]
            )
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
        response = Message(
            role=Role.user,
            message_id=message_id,
            parts=[Part(root=TextPart(text=os.getenv("MOCK_RESP", "")))],
        )
        if not os.getenv("MOCK_RESP"):
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
LOG_LEVEL=debug python -m src.services.a2a_live.adk_supervisor_agent
python -m src.services.a2a_live.adk_supervisor_agent
MOCK_RESP="achatbot factory, create chat bots with vad,turn, asr, llm(tools)/mllm/audio-llm/omni-llm, tts, avatar, ocr, detect object" python -m src.services.a2a_live.adk_supervisor_agent
"""
if __name__ == "__main__":
    import asyncio

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
        host_agent = ADKSupervisorLiveAgent(
            ["http://0.0.0.0:6666"],
            http_client_wrapper(),
            model="gemini-2.5-flash-native-audio-preview-09-2025",
        )  # github repo describe is long-time task
        await host_agent.wait_init()  # sync init

        app_name = "A2A_app"
        user_id = "user1"
        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name=app_name, user_id=user_id)

        # Configure response format based on client preference
        # IMPORTANT: You must choose exactly ONE modality per session
        # Either ["TEXT"] for text responses OR ["AUDIO"] for voice responses
        # You cannot use both modalities simultaneously in the same session

        agent = host_agent.create_agent()
        # Force AUDIO modality for native audio models regardless of client preference
        model_name = agent.model if isinstance(agent.model, str) else agent.model.model
        is_native_audio = "native-audio" in model_name.lower()

        modality = "AUDIO" if is_native_audio else "TEXT"

        # Enable session resumption for improved reliability
        # For audio mode, enable output transcription to get text for UI display
        voice_config = types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfigDict(voice_name="Aoede")
        )
        speech_config = types.SpeechConfig(voice_config=voice_config)
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,  # use BIDI
            response_modalities=[modality],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=speech_config,
            output_audio_transcription=types.AudioTranscriptionConfig()
            if is_native_audio
            else None,
        )

        # Create LiveRequestQueue in async context (recommended best practice)
        # This ensures the queue uses the correct event loop
        live_request_queue = LiveRequestQueue()
        host_runner = Runner(
            app_name=app_name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=session_service,
            memory_service=InMemoryMemoryService(),
        )

        content = types.Content(role="user", parts=[types.Part.from_text(text="what' a chatbot?")])
        live_request_queue.send_content(content=content)

        live_events = host_runner.run_live(
            user_id=user_id,
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        )
        async for event in live_events:
            if event.turn_complete:
                break

            if (
                event.content
                and event.content.parts
                and len(event.content.parts) > 0
                and event.content.parts[0].inline_data
                and event.content.parts[0].inline_data.mime_type
                and "audio/pcm" in event.content.parts[0].inline_data.mime_type
            ):  # output audio
                print(
                    f"audio: {event.content.parts[0].inline_data.mime_type} {len(event.content.parts[0].inline_data.data)=}"
                )
            elif event.input_transcription and event.input_transcription.text:  # input text
                print(f"{event.input_transcription.text=}")
            elif event.output_transcription and event.output_transcription.text:  # output text
                print(f"{event.output_transcription.text=}")
            elif event.usage_metadata:
                print(f"{event.usage_metadata=}")
            else:
                print("event--->", event)

        # Clean up resources (always runs, even if asyncio.wait fails)
        live_request_queue.close()
        await http_client_wrapper.stop()
        print(f"Client #{user_id} disconnected")

    asyncio.run(arun())
