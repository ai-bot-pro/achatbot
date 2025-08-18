#!/usr/bin/env python3

"""
Streaming TTS example with subtitles.

This example is similar to the example basic_audio_streaming.py, but it shows
WordBoundary events to create subtitles using SubMaker.
"""

import asyncio

import edge_tts

TEXT = "您好！有什么可以帮助您的吗？请告诉我您需要的信息或问题，我会尽力为您解答。你叫什么名字?"
VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_FILE = "test.mp3"
WEBVTT_FILE = "test.vtt"


async def amain() -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE, boundary="SentenceBoundary")
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                print(chunk["type"], len(chunk.get("data", "")))
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(chunk)
                submaker.feed(chunk)
            elif chunk["type"] == "SentenceBoundary":
                print(chunk)
                submaker.feed(chunk)

    with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
        file.write(submaker.get_srt())


if __name__ == "__main__":
    asyncio.run(amain())
