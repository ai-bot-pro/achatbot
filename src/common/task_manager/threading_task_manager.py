import logging
import threading

from src.common.task_manager.base import Task, TaskManager


class ThreadingTaskManager(TaskManager):
    async def run_task(self, target, name: str, tag: str, **kwargs):
        thread = threading.Thread(target=target, name=name, kwargs=kwargs)
        thread.daemon = True  # 使得主线程结束时，所有子线程也会结束
        thread.start()
        tid = str(thread.ident)
        self._tasks[tid] = Task(tid=tid, name=name, tag=tag, task=thread)
        return tid

    def cleanup(self):
        for tid, val in list(self._tasks.items()):
            thread: threading.Thread = val.task
            tag = val.tag
            try:
                if thread.is_alive():
                    thread.join(timeout=self._task_done_timeout)
                    logging.info(f"tid:{tid} tag:{tag} thread: {thread} joined")
                else:
                    logging.warning(f"tid:{tid} tag:{tag} thread: {thread} already finished")
            except Exception as e:
                logging.error(f"Error while cleaning up thread {tid}: {e}")
