#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
from typing import Callable, Any
import asyncio
import queue


async def async_task(sync_func: Callable, *args, **kwargs) -> Any:
    """
    https://docs.python.org/3.11/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        # Futures
        return await loop.run_in_executor(pool, sync_func, *args, **kwargs)


def fetch_async_items(queue: queue.Queue, asyncFunc, *args) -> None:
    async def get_items() -> None:
        try:
            async for item in asyncFunc(*args):
                queue.put(item)
            queue.put(None)
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error(f"error:{e} trace: {error_message}")

            queue.put(None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(get_items())
    loop.close()
