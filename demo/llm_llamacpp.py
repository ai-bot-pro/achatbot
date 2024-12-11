#!/usr/bin/env python
# -*- coding: utf-8 -*-
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import json
import subprocess
import platform

from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer


def text_to_speech(text: str):
    if platform.system() != "Darwin":
        return
    print("bot speech:", text)
    try:
        subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather = {
        "location": location,
        "temperature": "50",
        "unit": unit,
    }

    return json.dumps(weather)


def web_search(query: str):
    info = {"query": query, "result": "search dumpy result"}

    return json.dumps(info)


map_functions = {
    "get_current_weather": get_current_weather,
    "web_search": web_search,
}


def generate(model_path: str, prompt: str, stream=True):
    # the same as forward
    llm = Llama(
        # path to GGUF file
        model_path=model_path,
        n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=4,  # The number of CPU threads to use, tailor to your system and the resulting performance
        # The number of layers to offload to GPU, if you have GPU acceleration
        # available. Set to 0 if no GPU acceleration is available on your system.
        n_gpu_layers=0,
    )
    # Simple inference example
    # output = llm(
    #    f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
    #    max_tokens=256,  # Generate up to 256 tokens
    #    stop=["<|end|>"],
    #    echo=True,  # Whether to echo the prompt
    # )
    # print(output['choices'][0]['text'])

    output = llm(
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
        max_tokens=256,  # Generate up to 256 tokens
        stop=["<|end|>", "</s>", "<s>", "<|user|>"],
        # echo=True,  # Whether to echo the prompt
        stream=stream,
    )
    print(output)

    item = ""
    if stream:
        for output in output:
            item += output["choices"][0]["text"]
            # if len(item) == 12:
            if "\n" in item:
                print(item)
                text_to_speech(item)
                item = ""

        if len(item) > 0:
            print(item)
            text_to_speech(item)
    else:
        text_to_speech(output["choices"][0]["text"])


def chat_completion(model_path: str, system: str, query: str, stream=False):
    llm = Llama(model_path=model_path, chat_format="chatml")
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {"role": "user", "content": query},
        ],
        max_tokens=256,
        stop=["<|end|>", "</s>", "<s>", "<|user|>"],
        stream=stream,
        # response_format={"type": "json_object"},
        temperature=0.7,
    )
    out_str = ""
    if stream is False:
        print(output)
        if "content" in output["choices"][0]["message"]:
            out_str = output["choices"][0]["message"]["content"]
    else:
        for item in output:
            print(item)
            if "content" in item["choices"][0]["delta"]:
                out_str += item["choices"][0]["delta"]["content"]
    text_to_speech(out_str)


def chat_completion_func_call(
    model_name: str, model_path: str, system: str, query: str, stream=False
):
    if "functionary" in model_name:
        # llm = Llama.from_pretrained(
        #    repo_id="meetkai/functionary-small-v2.4-GGUF",
        #    filename="functionary-small-v2.4.Q4_0.gguf",
        #    chat_format="functionary-v2",
        #    tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
        # )
        llm = Llama(
            model_path=model_path,
            chat_format="functionary-v2",
            tokenizer=LlamaHFTokenizer.from_pretrained(
                "./models/meetkai/functionary-small-v2.4-GGUF"
            ),
        )
    else:
        llm = Llama(
            model_path=model_path,
            chat_format="chatml-function-calling",
        )

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {"role": "user", "content": query},
    ]
    print(messages)
    tools = [  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "web search by query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "web search query"}},
                    "required": ["query"],
                },
            },
        },
    ]

    output = llm.create_chat_completion(
        messages=messages,
        # tools=None,
        tools=tools,
        # tool_choice="none",
        tool_choice="auto",
        # tool_choice={
        #    "type": "function",
        #    "function": {
        #        "name": "get_current_weather"
        #    }
        # },
        max_tokens=256,
        stop=["<|end|>", "</s>", "<s>", "<|user|>"],
        stream=stream,
        # response_format={"type": "json_object"},
        temperature=0.7,
    )

    out_str = ""
    args_str = ""
    function_name = ""
    is_tool_call = False
    if stream is False:
        print(output["choices"])
        if "tool_calls" in output["choices"][0]["message"]:
            is_tool_call = True
            args_str = output["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            function_name = output["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        else:
            if "content" in output["choices"][0]["message"]:
                out_str = output["choices"][0]["message"]["content"]
    else:
        for item in output:
            print(item["choices"][0])
            if "tool_calls" in item["choices"][0]["delta"]:
                is_tool_call = True
                tool_calls = item["choices"][0]["delta"]["tool_calls"]
                if tool_calls is None or len(tool_calls) == 0:
                    continue
                args_str += tool_calls[0]["function"]["arguments"]
                if tool_calls[0]["function"]["name"] is not None:
                    function_name = tool_calls[0]["function"]["name"]
            else:
                if "content" in item["choices"][0]["delta"]:
                    out_str += item["choices"][0]["delta"]["content"]
    if is_tool_call is True:
        print(function_name)
        args = json.loads(args_str)
        out_str = map_functions[function_name](**args)

    text_to_speech(out_str)


def chat_vision_img(model_path, clip_model_path):
    chat_handler = MiniCPMv26ChatHandler(clip_model_path=clip_model_path)
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
        chat_format="minicpm-v-2.6",
    )
    res = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述下图片"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/OpenBMB/MiniCPM-V/main/assets/airplane.jpeg"
                        },
                    },
                ],
            },
        ]
    )
    print(res)


r"""
python demo/llm_llamacpp.py -t chat-func
python demo/llm_llamacpp.py -t chat-func  -q "今天天气怎么样"

TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func  -mn functionary -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf
TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func  -mn functionary -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf -q "今天天气怎么样"
TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func  -mn functionary -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf -q "查下今天的美股股票价格"
TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func  -mn functionary -st 1 -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf
TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func -q "今天天气怎么样" -mn functionary -st 1 -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf
TOKENIZERS_PARALLELISM=true python demo/llm_llamacpp.py -t chat-func -q "查下今天的美股股票价格" -mn functionary -st 1 -m ./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf

python demo/llm_llamacpp.py -t chat-vision-img -st 1 -m ./models/openbmb/MiniCPM-V-2_6-gguf/ggml-model-Q4_0.gguf -cm ./models/openbmb/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-mn", type=str, default="qwen", help="e.g. qwen,functionary"
    )
    parser.add_argument(
        "--type", "-t", type=str, default="generate", help="choice generate | chat | chat-func"
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="./models/qwen2-1_5b-instruct-q8_0.gguf",
        help="model path",
    )
    parser.add_argument(
        "--clip_model_path",
        "-cm",
        type=str,
        default="./models/openbmb/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf",
        help="clip model path",
    )

    parser.add_argument(
        "--system",
        "-s",
        type=str,
        default="你是一个中国人,智能助手,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。默认使用公制单位。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码，例如不要返回用户的经度。",
        help="system prompt",
    )

    parser.add_argument("--query", "-q", type=str, default="你好", help="query prompt")
    parser.add_argument("--stream", "-st", type=bool, default=False, help="stream output")
    args = parser.parse_args()
    system_prompt = args.system
    model_path = args.model_path
    model_name = args.model_name
    clip_model_path = args.clip_model_path
    prompt = args.query
    if args.type == "chat":
        chat_completion(model_path, system_prompt, prompt, args.stream)
    elif args.type == "chat-func":
        chat_completion_func_call(model_name, model_path, system_prompt, prompt, args.stream)
    elif args.type == "chat-vision-img":
        chat_vision_img(model_path, clip_model_path)
    else:
        generate(model_path, prompt, args.stream)
