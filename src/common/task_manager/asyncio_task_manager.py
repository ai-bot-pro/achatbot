import asyncio
import logging
import uuid

from src.common.task_manager.base import Task, TaskManager


class AsyncioTaskManager(TaskManager):
    """
    use asyncio
    !TODO: @weedge
    - use [uvloop](https://github.com/MagicStack/uvloop) to speed up
    """

    def __init__(
        self, task_done_timeout: int = 5, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        super().__init__(task_done_timeout=task_done_timeout)
        self._loop = loop or asyncio.get_event_loop()

    async def run_task(self, target, name: str, tag: str, **kwargs):
        task_id = str(uuid.uuid4())

        async def wrapped_target():
            if asyncio.iscoroutinefunction(target):
                await target(**kwargs)
            else:
                # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
                # Run in the default loop's executor(default thread pool)
                await self._loop.run_in_executor(None, target, **kwargs)

        task = self._loop.create_task(wrapped_target(), name=name)
        self._tasks[task_id] = Task(tid=task_id, name=name, tag=tag, task=task)

        return task_id

    async def cleanup(self):
        for task_id, val in self._tasks.items():
            task: asyncio.Task = val.task
            tag = val.tag
            try:
                if not task.done():
                    task.cancel()
                    try:
                        # wait task timeout
                        await asyncio.wait_for(task, self._task_done_timeout)
                    except asyncio.CancelledError:
                        logging.info(f"task_id:{task_id} tag:{tag} task: {task} cancelled")
                else:
                    logging.warning(f"task_id:{task_id} tag:{tag} task: {task} already completed")
            except Exception as e:
                logging.error(f"Error while cleaning up task {task_id}: {e}")
