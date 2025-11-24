import asyncio
from typing import AsyncGenerator

from google.adk.agents import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.genai import Client
from google.genai import types as genai_types


async def monitor_stock_price(stock_symbol: str) -> AsyncGenerator[str, None]:
    """该函数将以持续、流式和异步的方式监控给定股票代码的价格。"""
    print(f"Start monitor stock price for {stock_symbol}!")

    # 模拟股票价格变化。
    await asyncio.sleep(4)
    price_alert1 = f"the price for {stock_symbol} is 300"
    yield price_alert1
    print(price_alert1)

    await asyncio.sleep(4)
    price_alert1 = f"the price for {stock_symbol} is 400"
    yield price_alert1
    print(price_alert1)

    await asyncio.sleep(20)
    price_alert1 = f"the price for {stock_symbol} is 900"
    yield price_alert1
    print(price_alert1)

    await asyncio.sleep(20)
    price_alert1 = f"the price for {stock_symbol} is 500"
    yield price_alert1
    print(price_alert1)


# 对于视频流，`input_stream: LiveRequestQueue` 是 ADK 保留的关键参数，用于传递视频流。
async def monitor_video_stream(
    input_stream: LiveRequestQueue,
) -> AsyncGenerator[str, None]:
    """监控视频流中有多少人。"""
    print("start monitor_video_stream!")
    client = Client(vertexai=False)
    prompt_text = "Count the number of people in this image. Just respond with a numeric number."
    last_count = None
    while True:
        last_valid_req = None
        print("Start monitoring loop")

        # 用此循环拉取最新图片并丢弃旧图片
        while not input_stream._queue.empty():
            live_req = await input_stream.get()

            if live_req.blob is not None and live_req.blob.mime_type == "image/jpeg":
                last_valid_req = live_req

        # 如果找到有效图片，则处理
        if last_valid_req is not None:
            print("Processing the most recent frame from the queue")

            # 用 blob 的数据和 mime 类型创建图片 part
            image_part = genai_types.Part.from_bytes(
                data=last_valid_req.blob.data, mime_type=last_valid_req.blob.mime_type
            )

            contents = genai_types.Content(
                role="user",
                parts=[image_part, genai_types.Part.from_text(prompt_text)],
            )

            # 调用模型根据图片和提示生成内容
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=(
                        "You are a helpful video analysis assistant. You can count"
                        " the number of people in this image or video. Just respond"
                        " with a numeric number."
                    )
                ),
            )
            if not last_count:
                last_count = response.candidates[0].content.parts[0].text
            elif last_count != response.candidates[0].content.parts[0].text:
                last_count = response.candidates[0].content.parts[0].text
                yield response
                print("response:", response)

        # 等待后再检查新图片
        await asyncio.sleep(0.5)


# 使用此函数帮助 ADK 在需要时停止你的流式工具。
# 例如，如果我们想停止 `monitor_stock_price`，智能体会
# 调用此函数 stop_streaming(function_name=monitor_stock_price)。
def stop_streaming(function_name: str):
    """停止流式

    参数：
      function_name: 要停止的流式函数名。
    """
    pass


root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="video_streaming_agent",
    instruction="""
      你是一个监控智能体。你可以使用提供的工具/函数进行视频监控和股票价格监控。
      当用户想要监控视频流时，
      你可以使用 monitor_video_stream 函数。当 monitor_video_stream
      返回警报时，你应该告知用户。
      当用户想要监控股票价格时，你可以使用 monitor_stock_price。
      不要问太多问题。不要太啰嗦。
    """,
    tools=[
        monitor_video_stream,
        monitor_stock_price,
        FunctionTool(stop_streaming),
    ],
)
