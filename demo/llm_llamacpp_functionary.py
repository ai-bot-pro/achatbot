#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer


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

# We should use HF AutoTokenizer instead of llama.cpp's tokenizer because
# we found that Llama.cpp's tokenizer doesn't give the same result as that
# from Huggingface. The reason might be in the training, we added new
# tokens to the tokenizer and Llama.cpp doesn't handle this successfully
llm = Llama.from_pretrained(
    repo_id="meetkai/functionary-small-v2.4-GGUF",
    filename="functionary-small-v2.4.Q4_0.gguf",
    chat_format="functionary-v2",
    local_dir="./models",
    tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
    n_gpu_layers=0,
)

messages = [
    {
        "role": "system",
        "content": "你是一个中国人,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。默认使用公制单位。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码，例如不要返回用户的经度。",
    },
    # {"role": "user", "content": "what's the weather like in Hanoi?"}
    {"role": "user", "content": "你好"},
]
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
    }
]

result = llm.create_chat_completion(
    messages=messages,
    tools=tools,
    tool_choice="auto",
    # tool_choice="auto",
    # tool_choice={
    #    "type": "function",
    #    "function": {
    #        "name": "get_current_weather"
    #    }
    # },
    max_tokens=256,
    stop=["<|end|>", "</s>", "<s>", "<|user|>"],
    stream=False,
    # response_format={"type": "json_object"},
    temperature=0.7,
)

print(result["choices"][0]["message"])
