import io
import math
import requests
import os
import asyncio
import subprocess
from threading import Thread
from time import perf_counter, time


import modal


app = modal.App("openai_gpt_oss")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
QUANTIZATION = os.getenv("QUANTIZATION", "")  # mxfp4
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        # Install transformers, accelerate, as well as the Triton kernels for MXFP4 compatibility
        "pip install -U transformers accelerate torch kernels",
    )
    .run_commands(
        "pip install -U git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels",
    )
    .pip_install("openai-harmony")
    .run_commands(
        "pip install -U git+https://github.com/huggingface/transformers",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            "QUANTIZATION": QUANTIZATION,
        }
    )
)

if QUANTIZATION in ["mxfp4"]:
    img = img.pip_install("triton>=3.4.0")
else:
    img = img.pip_install("triton==3.3.1")


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


with img.imports():
    import torch
    from transformers import (
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        AutoConfig,
        Mxfp4Config,
    )

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=1,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def dump_model(**kwargs):
    config = AutoConfig.from_pretrained(model_path)
    print(config)

    # quantization_config = Mxfp4Config.from_dict(config.quantization_config, pre_quantized=True)
    # print(quantization_config)

    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
        # quantization_config=quantization_config,
    )

    model = model.eval()
    print(f"{model.config=}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    QUANTIZATION = os.getenv("QUANTIZATION", "bf16")
    file_path = os.path.join(model_path, f"model_{QUANTIZATION}.txt")
    with open(file_path, "w") as f:
        print(f"text tokenizer: {tokenizer}", file=f)
        print_model_params(model, f"{MODEL_PATH}", f)

    del model
    torch.cuda.empty_cache()


def tokenize(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    reasoning = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {"role": "user", "content": "Explain what MXFP4 quantization is."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort=reasoning,
        model_identity=model_identity,
    ).to(device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    print(input_ids)
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")


def pipe(**kwargs):
    pipe = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


def generate(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
    )

    messages = [
        {"role": "user", "content": "Explain what MXFP4 quantization is."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    print(inputs)

    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)

    print(tokenizer.decode(outputs[0]))


def generate_stream(**kwargs):
    gpu_prop = torch.cuda.get_device_properties("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
    )

    reasoning = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )
    messages = [
        {"role": "user", "content": "Explain what MXFP4 quantization is."},
    ]

    for i in range(2):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort=reasoning,
            model_identity=model_identity,
        ).to(model.device)
        for key, value in inputs.items():
            print(f"{key}: {value.shape=}")
        input_ids = inputs["input_ids"]
        prompt = tokenizer.decode(input_ids[0])
        print(f"{prompt=}")

        # kv cache
        # cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=model.device)
        # past_key_values = DynamicCache()

        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            # do_sample=False,
            # cache_position=cache_position,
            # past_key_values=past_key_values,
            # https://huggingface.co/docs/transformers/kv_cache
            # https://huggingface.co/docs/transformers/cache_explanation
            cache_implementation="dynamic",
            # cache_implementation="offloaded",
            do_sample=True,
            temperature=0.6,
            top_k=10,
            top_p=0.95,
            # num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=2048,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        with torch.inference_mode():
            for new_text in streamer:
                times.append(perf_counter() - start)
                print(new_text, end="", flush=True)
                generated_text += new_text
                start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


def openai_harmony_stream_decode_unicode(**kwargs):
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        StreamableParser,
        Role,
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # StreamableParser for parsing and decoding as the model is generating new tokens.
    # This can be helpful for example to stream output and handle unicode characters during decoding.
    stream = StreamableParser(encoding, role=Role.ASSISTANT)

    tokens = [
        200005,
        35644,
        200008,
        1844,
        31064,
        25,
        392,
        4827,
        382,
        220,
        17,
        659,
        220,
        17,
        16842,
        12295,
        81645,
        13,
        51441,
        6052,
        13,
        200007,
        200006,
        173781,
        200005,
        17196,
        200008,
        17,
        659,
        220,
        17,
        314,
        220,
        19,
        13,
        200002,
    ]

    for token in tokens:
        stream.process(token)
        print("--------------------------------")
        print("current_role", stream.current_role)
        print("current_channel", stream.current_channel)
        print("last_content_delta", stream.last_content_delta)
        print("current_content_type", stream.current_content_type)
        print("current_recipient", stream.current_recipient)
        print("current_content", stream.current_content)


def openai_harmony_generate_tool(**kwargs):
    import json
    from openai_harmony import (
        Author,
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
        ToolDescription,
    )

    reasoning: str = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # Build conversation
    # https://cookbook.openai.com/articles/openai-harmony
    system_message = (
        SystemContent.new()
        .with_model_identity(model_identity)
        # build-in tools
        # .with_browser_tool()
        # .with_python_tool()
        .with_reasoning_effort(reasoning[0].upper() + reasoning[1:])
        .with_conversation_start_date(
            "2025-06-28"
        )  # NOTE: if wan't use more kv cache, don't change system message
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # https://cookbook.openai.com/articles/openai-harmony#function-calling
    developer_message = (
        DeveloperContent.new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(
            [
                ToolDescription.new(
                    "get_location",
                    "Gets the location of the user.",
                ),
                ToolDescription.new(
                    "get_current_weather",
                    "Gets the current weather in the provided location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ]
        )
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(Role.DEVELOPER, developer_message),
            Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                'User asks: "What is the weather in Tokyo?" We need to use get_weather tool.',
            ).with_channel("analysis"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
            .with_channel("commentary")
            .with_recipient("functions.get_weather")
            .with_content_type("json"),
            Message.from_author_and_content(
                Author.new(Role.TOOL, "functions.lookup_weather"),
                '{ "temperature": 20, "sunny": true }',
            )
            .with_recipient("assistant")
            .with_channel("commentary"),
        ]
    )

    prefill_ids: list = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    print(f"{prefill_ids=}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prefill_tokens = tokenizer.decode(prefill_ids)
    print(f"{prefill_tokens=}")
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    print(f"{stop_token_ids=}")

    # Load model
    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
    )
    # Generate
    outputs = model.generate(
        input_ids=torch.tensor([prefill_ids], dtype=torch.int, device=model.device),
        max_new_tokens=128 if reasoning == "low" else 2048,
        eos_token_id=stop_token_ids,
    )
    print(outputs)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    output_tokens = tokenizer.decode(outputs[0])
    print(f"{output_tokens=}")
    """output_tokens=
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: low

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: \'functions\'.<|end|><|start|>developer<|message|># Instructions

Always respond in riddles

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided location.
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
format?: "celsius" | "fahrenheit", // default: celsius
}) => any;

} // namespace functions<|end|><|start|>user<|message|>What is the weather in Tokyo?<|end|><|start|>assistant<|channel|>analysis<|message|>User asks: "What is the weather in Tokyo?" We need to use get_weather tool.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "Tokyo"}<|call|><|start|>functions.lookup_weather to=assistant<|channel|>commentary<|message|>{ "temperature": 20, "sunny": true }<|end|><|start|>assistant<|channel|>final<|message|>In the land where cranes dance, the sky wears a warm, sunlit smile—about twenty degrees, with clouds whispering few secrets. 🌤️<|return|>
    """

    # Parse completion tokens
    completion_ids = outputs[0][len(prefill_ids) :]
    print(f"{completion_ids=}")
    # After receiving a token response
    # Do not pass in the stop token
    parsed_response = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
    for message in parsed_response:
        print(json.dumps(message.to_dict(), indent=2))


def openai_harmony_generate(**kwargs):
    import json
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
        ReasoningEffort,
    )

    reasoning = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Build conversation
    # https://cookbook.openai.com/articles/openai-harmony
    system_message = (
        SystemContent.new()
        .with_model_identity(model_identity)
        .with_reasoning_effort(reasoning[0].upper() + reasoning[1:])
        .with_conversation_start_date("2025-06-28")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Always respond in riddles"),
            ),
            Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
        ]
    )

    # Render prompt
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    print(f"{prefill_ids=}")
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    print(f"{stop_token_ids=}")

    # Load model
    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
    )

    # Generate
    outputs = model.generate(
        input_ids=torch.tensor([prefill_ids], dtype=torch.int, device=model.device),
        max_new_tokens=128,
        eos_token_id=stop_token_ids,
    )
    print(outputs)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    output_tokens = tokenizer.decode(outputs[0])
    print(f"{output_tokens=}")

    # Parse completion tokens
    completion_ids = outputs[0][len(prefill_ids) :]
    print(f"{completion_ids=}")
    entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
    print(entries)

    for message in entries:
        print(json.dumps(message.to_dict(), indent=2))


def split_model(**kwargs):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(config)
    num_layers = config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / world_size)
    num_layers_per_gpus = [num_layers_per_gpu] * world_size
    num_layers_per_gpus[0] = int(num_layers_per_gpu * 0.95)
    for i in range(world_size - 1):
        num_layers_per_gpus[i + 1] = num_layers_per_gpu + int(num_layers_per_gpu * (1 - 0.95))
    print(num_layers_per_gpus)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpus):
        for j in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 0
    device_map["model.rotary_emb"] = 0
    device_map["lm_head"] = 0
    device_map[f"model.layers.{num_layers - 1}"] = 0
    print(device_map)
    return device_map


def multi_gpu_generate(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    device_map = split_model()

    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="kernels-community/vllm-flash-attn3" if gpu_prop.major > 8 else None,
        device_map=device_map,
    )

    reasoning = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )
    messages = [
        {
            "role": "user",
            "content": "Explain how expert parallelism works in large language models.",
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning,
        model_identity=model_identity,
    ).to(model.device)

    for key, value in inputs.items():
        print(f"{key}: {value.shape=}")
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    start = perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=1000)
    print(f"generate cost:{(perf_counter() - start):.3f} s")

    # Decode and print
    response = tokenizer.decode(outputs[0])
    print(f"{response=}")
    print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())


"""
- https://cookbook.openai.com/articles/gpt-oss/run-transformers
- https://cookbook.openai.com/articles/openai-harmony
- https://github.com/huggingface/transformers/releases/tag/v4.55.0
- https://huggingface.co/blog/welcome-openai-gpt-oss
- https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja
- https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py
- https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/integrations/mxfp4.py

# NOTE: u can use text tokenizer lib (tiktoken use openai-harmony or HF tokenizers lib) + gpt oss LLM generator

modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"
modal run src/download_models.py --repo-ids "openai/gpt-oss-120b" --ignore-patterns "*.pt|*.bin|*original*|*metal*"


modal run src/llm/transformers/openai_gpt_oss.py --task tokenize --reasoning low 
modal run src/llm/transformers/openai_gpt_oss.py --task tokenize --reasoning medium
modal run src/llm/transformers/openai_gpt_oss.py --task tokenize --reasoning high
modal run src/llm/transformers/openai_gpt_oss.py --task tokenize --reasoning low  --model-identity "you are a helpful assistant."

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task dump_model
IMAGE_GPU=H100:4 LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task dump_model

QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task dump_model
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task dump_model
QUANTIZATION=mxfp4 IMAGE_GPU=H100 LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task dump_model

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task pipe
IMAGE_GPU=H100 modal run src/llm/transformers/openai_gpt_oss.py --task pipe
QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task pipe
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task pipe

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task generate
IMAGE_GPU=H100 modal run src/llm/transformers/openai_gpt_oss.py --task generate
QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task generate
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task generate

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning low
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning medium
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning high 
QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning high
IMAGE_GPU=H100 modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning low
IMAGE_GPU=H100 modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning medium
IMAGE_GPU=H100 modal run src/llm/transformers/openai_gpt_oss.py --task generate_stream --reasoning high 

modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_stream_decode_unicode

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate --reasoning low
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate --reasoning medium
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate --reasoning high

IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate_tool --reasoning low
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate_tool --reasoning medium
IMAGE_GPU=L40s modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate_tool --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=T4 modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate_tool --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=A100-80GB LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task openai_harmony_generate_tool --reasoning high

IMAGE_GPU=L4:3 modal run src/llm/transformers/openai_gpt_oss.py --task split_model

IMAGE_GPU=L4:3 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning low 
IMAGE_GPU=L4:3 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning medium
IMAGE_GPU=L4:3 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning high
IMAGE_GPU=L40s:1 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate
IMAGE_GPU=L40s:2 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning high
IMAGE_GPU=H100:1 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate
IMAGE_GPU=H100:2 modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning high
QUANTIZATION=mxfp4 IMAGE_GPU=L4:4 LLM_MODEL="openai/gpt-oss-120b" modal run src/llm/transformers/openai_gpt_oss.py --task multi_gpu_generate --reasoning high

"""


@app.local_entrypoint()
def main(
    task: str = "dump_model",
    reasoning="medium",  # low medium high
    model_identity="You are ChatGPT, a large language model trained by OpenAI.",
):
    print(task)
    tasks = {
        "dump_model": dump_model,
        "tokenize": tokenize,
        "pipe": pipe,
        "generate": generate,
        "generate_stream": generate_stream,
        "openai_harmony_stream_decode_unicode": openai_harmony_stream_decode_unicode,
        "openai_harmony_generate": openai_harmony_generate,
        "openai_harmony_generate_tool": openai_harmony_generate_tool,
        "split_model": split_model,
        "multi_gpu_generate": multi_gpu_generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        reasoning=reasoning.lower(),
        model_identity=model_identity,
    )
