import logging
from typing import Callable, Literal

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from .agent import Agent


class GitMcpAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(
        self,
        mode: Literal["complete", "stream"] = "stream",
        token_stream_callback: Callable[[str], None] | None = print,
        mcp_url: str | None = "https://gitmcp.io/ai-bot-pro/achatbot",
    ):
        self.agent = Agent(
            mode=mode,
            token_stream_callback=token_stream_callback,
            mcp_url=mcp_url,
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        if not context.message:
            raise Exception("No message provided")

        logging.info(f"{query=} {context.current_task=} {context.message=}")

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        async for event in self.agent.stream(query):
            if event["is_task_complete"]:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name="current_result",
                            description="Result of request to agent.",
                            text=event["content"],
                        ),
                    )
                )
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            elif event["require_user_input"]:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                event["content"],
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            else:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        append=True,
                        status=TaskStatus(
                            state=TaskState.working,
                            message=new_agent_text_message(
                                event["content"],
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=False,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
