from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import multiprocessing
import threading
from typing import Dict


@dataclass
class Task:
    tid: str = ""
    name: str = ""
    tag: str = ""
    task: asyncio.Task | threading.Thread | multiprocessing.Process = None

    def __str__(self) -> str:
        return f"tid: {self.tid} name:{self.name} tag:{self.tag} is_alive:{self.is_alive()}"

    def is_alive(self) -> bool:
        if self.task is None:
            return False

        if isinstance(self.task, multiprocessing.Process):
            return self.task.is_alive()
        if isinstance(self.task, threading.Thread):
            return self.task.is_alive()
        if isinstance(self.task, asyncio.Task):
            return self.task.done()

        return False


class TaskManager(ABC):
    def __init__(self, task_done_timeout: int = 5) -> None:
        """
        just use dict to store process for local task
        !TODO: @weedge
        - if distributed task, need database to storage process info
        - shecdule task
        """
        self._tasks: Dict[str, Task] = {}
        self._task_done_timeout = task_done_timeout

    @property
    def tasks(self):
        return self._tasks

    @abstractmethod
    async def run_task(self, target, name: str, tag: str, **kwargs):
        """
        - use multiprocessing to run task
        - use threading to run task
        - use asyncio create task to run
        """

    def get_task_num(self, tag: str):
        num = 0
        for val in self._tasks.values():
            if val.tag == tag and val.is_alive():
                num += 1
        return num

    def get_task(self, tid: str):
        if tid in self._tasks:
            return self._tasks[tid]
        return None

    @abstractmethod
    def cleanup(self):
        """
        clean up process
        """
