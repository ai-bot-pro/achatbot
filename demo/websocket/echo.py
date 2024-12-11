# !/usr/bin/env python
r"""
https://websockets.readthedocs.io/en/stable/
"""

from websockets.server import serve
import asyncio


async def echo(websocket):
    async for message in websocket:
        print(message)
        await websocket.send(message)


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


asyncio.run(main())
