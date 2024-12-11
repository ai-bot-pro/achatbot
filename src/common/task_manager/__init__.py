import asyncio
import logging
from typing import Literal

from src.common.task_manager.base import TaskManager
from src.common.task_manager.multiprocessing_task_manager import MultiprocessingTaskManager
from src.common.task_manager.asyncio_task_manager import AsyncioTaskManager
from src.common.task_manager.threading_task_manager import ThreadingTaskManager


class TaskManagerFactory:
    loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def task_manager(
        type: Literal["multiprocessing", "threading", "asyncio"] = "multiprocessing",
        task_done_timeout=5,
    ) -> TaskManager:
        match type:
            case "asyncio":
                return AsyncioTaskManager(
                    task_done_timeout=task_done_timeout, loop=TaskManagerFactory.loop
                )
            case "threading":
                return ThreadingTaskManager(task_done_timeout=task_done_timeout)
            case "multiprocessing":
                return MultiprocessingTaskManager(task_done_timeout=task_done_timeout)
            case _:
                return MultiprocessingTaskManager(task_done_timeout=task_done_timeout)

    @staticmethod
    def cleanup(func):
        try:
            if asyncio.iscoroutinefunction(func):
                if TaskManagerFactory.loop is None or TaskManagerFactory.loop.is_closed():
                    TaskManagerFactory.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(TaskManagerFactory.loop)

                TaskManagerFactory.loop.run_until_complete(func())
            else:
                func()

            if TaskManagerFactory.loop:
                TaskManagerFactory.loop.close()
        except Exception as e:
            logging.error(f"Error during TaskManagerFactory cleanup: {e}")
