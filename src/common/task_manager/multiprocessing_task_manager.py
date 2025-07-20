import logging
import multiprocessing
import traceback

from src.common.task_manager.base import Task, TaskManager


class MultiprocessingTaskManager(TaskManager):
    async def run_task(self, target, name: str, tag: str, **kwargs):
        """
        use multiprocessing(spawn/fork/forkserver,default fork) to run task
        """
        process: multiprocessing.Process = multiprocessing.Process(
            target=target, name=name, kwargs=kwargs
        )
        process.start()
        pid = str(process.pid)
        self._tasks[pid] = Task(tid=pid, name=name, tag=tag, task=process)
        return pid

    def cleanup(self):
        # Clean up function, just to be extra safe
        for pid, val in self._tasks.items():
            proc: multiprocessing.Process = val.task
            tag = val.tag
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=self._task_done_timeout)
                    proc.close()
                    logging.info(f"pid:{pid} tag:{tag} proc: {proc} close", exc_info=True)
                else:
                    logging.warning(f"pid:{pid} tag:{tag} proc: {proc} already closed")
            except Exception as e:
                logging.error(f"Error while cleaning up process {pid}: {e}", exc_info=True)
                if proc.is_alive():
                    proc.kill()
                    logging.warning(f"pid:{pid} tag:{tag} proc: {proc} killed")
