#!/usr/bin/env python
# -*- coding: utf-8 -*-

from llama_cpp import Llama
import subprocess
import platform


def text_to_speech(text: str):
    if platform.system() != 'Darwin':
        return
    print("bot speech:", text)
    try:
        subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


def generate(model_path: str, prompt: str, stream=True):
    # the same as forward
    llm = Llama(
        # path to GGUF file
        model_path=model_path,
        n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=4,  # The number of CPU threads to use, tailor to your system and the resulting performance
        # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
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
        stop=["<|end|>"],
        # echo=True,  # Whether to echo the prompt
        stream=stream,
    )
    print(output)

    item = ""
    if stream:
        for output in output:
            item += output['choices'][0]['text']
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
    llm = Llama(
        model_path=model_path,
        chat_format="chatml")
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
        if 'content' in output['choices'][0]['message']:
            out_str = output['choices'][0]['message']['content']
    else:
        for item in output:
            print(item)
            if 'content' in item['choices'][0]['delta']:
                out_str += item['choices'][0]['delta']['content']
    text_to_speech(out_str)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', "-t", type=str,
                        default="generate", help='choice generate or chat')
    parser.add_argument('--model_path', "-m", type=str,
                        default="./models/qwen2-1_5b-instruct-q8_0.gguf", help='model path')
    parser.add_argument('--system', "-s", type=str,
                        default="你是一个中国人,请用中文回答。回答限制在1-5句话内。要友好、乐于助人且简明扼要。默认使用公制单位。保持对话简短而甜蜜。只用纯文本回答，不要包含链接或其他附加内容。不要回复计算机代码，例如不要返回用户的经度。", help='system prompt')
    parser.add_argument('--query', "-q", type=str,
                        default="你好", help='query prompt')
    parser.add_argument('--stream', "-st", type=bool,
                        default=False, help='stream output')
    args = parser.parse_args()
    system_prompt = args.system,
    model_path = args.model_path
    prompt = args.query
    if args.type == "chat":
        chat_completion(model_path, system_prompt, prompt, args.stream)
    else:
        generate(model_path, prompt, args.stream)
