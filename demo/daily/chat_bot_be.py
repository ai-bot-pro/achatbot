import argparse
import asyncio
import aiohttp
import os
import sys

from daily import (
    CallClient,
    Daily,
    EventHandler,
    VirtualCameraDevice,
    VirtualMicrophoneDevice,
    VirtualSpeakerDevice)


def main(room_url, token):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chat-bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    main(config.u, config.t)
