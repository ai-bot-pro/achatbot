import asyncio
import logging
from typing import Coroutine, Dict, Optional, Sequence
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
        self._tasks: Dict[str, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

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



    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not self._loop:
            raise Exception("TaskManager missing event loop, use TaskManager.set_event_loop().")
        return self._loop

    def create_task(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        """
        Creates and schedules a new asyncio Task that runs the given coroutine.

        The task is added to a global set of created tasks.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop to use for creating the task.
            coroutine (Coroutine): The coroutine to be executed within the task.
            name (str): The name to assign to the task for identification.

        Returns:
            asyncio.Task: The created task object.
        """

        async def run_coroutine():
            try:
                await coroutine
            except asyncio.CancelledError:
                logging.trace(f"{name}: task cancelled")
                # Re-raise the exception to ensure the task is cancelled.
                raise
            except Exception as e:
                logging.exception(f"{name}: unexpected exception: {e}")

        if not self._loop:
            raise Exception("TaskManager missing event loop, use TaskManager.set_event_loop().")

        task = self._loop.create_task(run_coroutine())
        task.set_name(name)
        self._add_task(task)
        logging.trace(f"{name}: task created")
        return task

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for an asyncio.Task to complete with optional timeout handling.

        This function awaits the specified asyncio.Task and handles scenarios for
        timeouts, cancellations, and other exceptions. It also ensures that the task
        is removed from the set of registered tasks upon completion or failure.

        Args:
            task (asyncio.Task): The asyncio Task to wait for.
            timeout (Optional[float], optional): The maximum number of seconds
                to wait for the task to complete. If None, waits indefinitely.
                Defaults to None.
        """
        name = task.get_name()
        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logging.warning(f"{name}: timed out waiting for task to finish")
        except asyncio.CancelledError:
            logging.trace(f"{name}: unexpected task cancellation (maybe Ctrl-C?)")
            raise
        except Exception as e:
            logging.exception(f"{name}: unexpected exception while stopping task: {e}")
        finally:
            self._remove_task(task)

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancels the given asyncio Task and awaits its completion with an
        optional timeout.

        This function removes the task from the set of registered tasks upon
        completion or failure.

        Args:
            task (asyncio.Task): The task to be cancelled.
            timeout (Optional[float]): The optional timeout in seconds to wait for the task to cancel.

        """
        name = task.get_name()
        task.cancel()
        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logging.warning(f"{name}: timed out waiting for task to cancel")
        except asyncio.CancelledError:
            # Here are sure the task is cancelled properly.
            pass
        except Exception as e:
            logging.exception(f"{name}: unexpected exception while cancelling task: {e}")
        finally:
            self._remove_task(task)

    def current_tasks(self) -> Sequence[asyncio.Task]:
        """Returns the list of currently created/registered tasks."""
        return list(self._tasks.values())

    def _add_task(self, task: asyncio.Task):
        name = task.get_name()
        self._tasks[name] = task

    def _remove_task(self, task: asyncio.Task):
        name = task.get_name()
        try:
            del self._tasks[name]
        except KeyError as e:
            logging.trace(f"{name}: unable to remove task (already removed?): {e}")