import asyncio
from collections import deque

from llm_llamacpp import generate

g_text_deque = deque(maxlen=100)


async def send_text():
    for i in range(1):
        print(f"case {i}")
        g_text_deque.appendleft(f"你好")


async def llm_generate(model_path):
    while True:
        if g_text_deque:
            text = g_text_deque.pop()
            generate(model_path, text)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model_path', "-lm", type=str,
                        default="./models/Phi-3-mini-4k-instruct-q4.gguf", help='llm model path')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(send_text())
    loop.create_task(llm_generate(args.llm_model_path))
    loop.run_forever()


if __name__ == "__main__":
    main()
