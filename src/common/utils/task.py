#!/usr/bin/env python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


async def async_task(sync_func: Callable, *args, **kwargs) -> Any:
    """
    https://docs.python.org/3.11/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        # Futures
        return await loop.run_in_executor(pool, sync_func, *args, **kwargs)
