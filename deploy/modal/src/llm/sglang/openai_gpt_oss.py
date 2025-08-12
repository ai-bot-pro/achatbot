import asyncio
import datetime
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time


import modal
import urllib


app = modal.App("openai_gpt_oss_sglang")
RUN_IMAGE_GPU = os.getenv("RUN_IMAGE_GPU", None)
SERVE_IMAGE_GPU = os.getenv("SERVE_IMAGE_GPU", None)
RUN_MAX_CONTAINERS = int(os.getenv("RUN_MAX_CONTAINERS", 1))
SERVE_MAX_CONTAINERS = int(os.getenv("SERVE_MAX_CONTAINERS", 1))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TP = os.getenv("TP", "1")
SERVE_ARGS = os.getenv("SERVE_ARGS", "")
SGLANG_VER = os.getenv("SGLANG_VER", "v0.5.0rc0")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/sgl-project/sglang",
        f"cd /sglang && git checkout {SGLANG_VER}",
        "cd /sglang && pip install -e python[all]",  # make sure you have the correct transformers version installed!
    )
    .run_commands(
        "pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129"
    )
    .run_commands(
        "pip3 install https://github.com/sgl-project/whl/releases/download/v0.3.3/sgl_kernel-0.3.3-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall"
    )
    .apt_install("libnuma-dev")  # Add NUMA library for sgl_kernel
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers.git",
    )
    .run_commands(
        "git clone https://github.com/weedge/gpt-oss.git", "cd /gpt-oss && pip install -e ."
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX 10.0+PTX",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": LLM_MODEL,
            "TP": TP,
            "SERVE_ARGS": SERVE_ARGS,
        }
    )
)


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
EVAL_OUTPUT_DIR = "/eval_output"
eval_out_vol = modal.Volume.from_name("eval_output", create_if_missing=True)


with img.imports():
    import torch
    import termcolor
    import sglang as sgl
    from sglang.srt.hf_transformers_utils import get_tokenizer
    from sglang.utils import async_stream_and_merge, stream_and_merge

    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
        ReasoningEffort,
        StreamableParser,
        StreamState,
        TextContent,
        ToolDescription,
        Author,
    )

    # sys.path.insert(0, "/gpt-oss")
    from gpt_oss.tools import apply_patch
    from gpt_oss.tools.simple_browser import SimpleBrowserTool
    from gpt_oss.tools.simple_browser.backend import ExaBackend
    from gpt_oss.tools.python_docker.docker_tool import PythonTool

    class TokenGenerator:
        def __init__(
            self,
            model_path: str,
            tensor_parallel_size: int = 1,
            mem_fraction_static: float = 0.7,
        ):
            # Create an LLM.
            self.engine = sgl.Engine(
                model_path=model_path,
                skip_tokenizer_init=True,
                tp_size=tensor_parallel_size,
                mem_fraction_static=mem_fraction_static,
                # quantization="mxfp4",
            )

        def generate(
            self,
            prompt_tokens: list[int],
            stop_tokens: list[int] | None = None,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 0,
            return_logprobs: bool = False,
        ):
            # https://docs.sglang.ai/backend/sampling_params.html
            sampling_params = {
                "n": 1,  # number of samples to generate
                "temperature": temperature,
                "top_p": top_p,
                "stop_token_ids": stop_tokens,
            }
            if max_tokens > 0:
                sampling_params["max_new_tokens"] = max_tokens
            pre_len = 0
            gen_iter = self.engine.generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params,
                stream=True,
                return_logprob=return_logprobs,
            )
            for output in gen_iter:
                token_ids = output["output_ids"]
                logprobs_list = (
                    output.logprobs
                    if hasattr(output["meta_info"], "output_token_logprobs")
                    else None
                )
                if return_logprobs is True:
                    new_logprobs = logprobs_list[pre_len:]
                else:
                    new_logprobs = [(None, token_id, None) for token_id in token_ids[pre_len:]]
                pre_len = len(token_ids)
                for logprob_val, token_id, _ in new_logprobs:
                    if logprob_val is None:
                        yield token_id
                    else:
                        yield (token_id, logprob_val)
                    if stop_tokens is not None and token_id in stop_tokens:
                        break

        async def async_generate(
            self,
            prompt_tokens: list[int],
            stop_tokens: list[int] | None = None,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 0,
            return_logprobs: bool = False,
        ):
            # https://docs.sglang.ai/backend/sampling_params.html
            sampling_params = {
                "n": 1,  # number of samples to generate
                "temperature": temperature,
                "top_p": top_p,
                "stop_token_ids": stop_tokens,
            }
            if max_tokens > 0:
                sampling_params["max_new_tokens"] = max_tokens
            pre_len = 0
            gen_iter = await self.engine.async_generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params,
                stream=True,
                return_logprob=return_logprobs,
            )
            async for output in gen_iter:
                token_ids = output["output_ids"]
                logprobs_list = (
                    output.logprobs
                    if hasattr(output["meta_info"], "output_token_logprobs")
                    else None
                )
                if return_logprobs is True:
                    new_logprobs = logprobs_list[pre_len:]
                else:
                    new_logprobs = [(None, token_id, None) for token_id in token_ids[pre_len:]]
                pre_len = len(token_ids)
                for logprob_val, token_id, _ in new_logprobs:
                    if logprob_val is None:
                        yield token_id
                    else:
                        yield (token_id, logprob_val)
                    if stop_tokens is not None and token_id in stop_tokens:
                        break

    REASONING_EFFORT = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]
    eval_out_dir = os.path.join(EVAL_OUTPUT_DIR, "sglang", MODEL_PATH.split("/")[-1])
    os.makedirs(eval_out_dir, exist_ok=True)


@app.function(
    gpu=RUN_IMAGE_GPU,
    cpu=8.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        EVAL_OUTPUT_DIR: eval_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=RUN_MAX_CONTAINERS,
)
async def remote_run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        run(func, **kwargs)


def run(func, **kwargs):
    func(**kwargs)


def run_cmd(cmd, capture_output=False):
    print(cmd)

    # os.environ["PYTHONPATH"] = "/sglang:" + os.environ["PYTHONPATH"]
    try:
        res = subprocess.run(
            cmd.strip(),
            shell=True,
            check=True,
            env=os.environ,
            capture_output=capture_output,
        )

    except subprocess.CalledProcessError as e:
        print("erro_code:", e.returncode)
        print("error:", e.stderr)
        raise

    return res


def benchmark(**kwargs):
    """
    https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
    """
    cmd = """
    python3 -m sglang.bench_serving --help     
    """
    run_cmd(cmd)

    url = serve.get_web_url()
    print(url)

    test_timeout = kwargs.get("test_timeout", 30 * 60)
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(url + "/health") as response:
                up = response.getcode() == 200
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {url}"

    print(f"Successful health check for server at {url}")

    num_prompts = kwargs.get("num_prompts", 5)
    max_concurrency = kwargs.get("max_concurrency", 1)
    local_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
    cmd = f"""
    python3 -m sglang.bench_serving --backend sglang-oai --base-url {url} \\
        --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 \\
        --num-prompts {num_prompts} --max-concurrency {max_concurrency} \\
        --output-file {eval_out_dir}/res_{formatted_time}.jsonl
    """
    run_cmd(cmd)


def generate(**kwargs):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Tokenize inputs
    tokenizer = get_tokenizer(model_path)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]
    print(token_ids_list)

    # Create an LLM.
    llm = sgl.Engine(
        model_path=model_path,
        skip_tokenizer_init=True,
        tp_size=int(os.getenv("TP", "1")),
        mem_fraction_static=0.7,
        # quantization="mxfp4",
    )

    # Create a sampling params object.
    # sampling_params = {"temperature": 0.8, "top_p": 0.95}
    sampling_params = {
        "max_new_tokens": kwargs.get("max_new_tokens", 128),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": kwargs.get("top_k", 20),
        "min_p": kwargs.get("min_p", 0.0),
        # Penalizers
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        # "min_new_tokens": kwargs.get("min_new_tokens", 1),
        # "stop_token_ids": tokenizer.all_special_ids,
    }
    outputs = llm.generate(input_ids=token_ids_list, sampling_params=sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        decode_output = tokenizer.decode(output["output_ids"])
        print("===============================")
        print(output)
        print(
            f"Prompt: {prompt}\nGenerated token ids: {output['output_ids']}\nGenerated text: {decode_output}"
        )
        print()


def batch_generate_stream(**kwargs):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Tokenize inputs
    tokenizer = get_tokenizer(model_path)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]
    print(token_ids_list)

    # Create an LLM.
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=int(os.getenv("TP", "1")),
        mem_fraction_static=0.7,
        # quantization="mxfp4",
    )

    # Create a sampling params object.
    # sampling_params = {"temperature": 0.8, "top_p": 0.95}
    sampling_params = {
        "max_new_tokens": kwargs.get("max_new_tokens", 128),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": kwargs.get("top_k", 20),
        "min_p": kwargs.get("min_p", 0.0),
        # Penalizers
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        # "min_new_tokens": kwargs.get("min_new_tokens", 1),
        "stop_token_ids": tokenizer.all_special_ids,
    }

    print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        merged_output = stream_and_merge(llm, prompt, sampling_params)
        print("Generated text:", merged_output)
        print()

    llm.shutdown()


def generate_stream(**kwargs):
    # Sample prompts.
    prompt = "The future of AI is"
    # Tokenize inputs
    tokenizer = get_tokenizer(model_path)
    token_ids = tokenizer.encode(prompt)
    print(token_ids)

    # Create an LLM.
    llm = sgl.Engine(
        model_path=model_path,
        skip_tokenizer_init=True,
        tp_size=int(os.getenv("TP", "1")),
        mem_fraction_static=0.7,
        quantization="mxfp4",
    )

    # Create a sampling params object.
    # sampling_params = {"temperature": 0.8, "top_p": 0.95}
    sampling_params = {
        "n": 1,  # number of samples to generate
        "max_new_tokens": kwargs.get("max_new_tokens", 128),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": kwargs.get("top_k", 20),
        "min_p": kwargs.get("min_p", 0.0),
        # Penalizers
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        # "min_new_tokens": kwargs.get("min_new_tokens", 1),
        "stop_token_ids": tokenizer.all_special_ids,
    }

    for i in range(2):
        gen_iter = llm.generate(
            input_ids=token_ids, sampling_params=sampling_params, stream=True, return_logprob=True
        )
        # Print the outputs.
        times = []
        start = time.perf_counter()
        pre_len = 0
        for output in gen_iter:
            print(output)
            text = tokenizer.decode(output["output_ids"])
            times.append(time.perf_counter() - start)
            print(text[pre_len:], end="", flush=True)
            pre_len = len(text)
            start = time.perf_counter()
        print(f"{i} generated Text: {text}")
        print(f"{i} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")

    llm.shutdown()


def harmony_generate(**kwargs):
    # --- 1) Render the prefill with Harmony ---
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Always respond in riddles"),
            ),
            Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
        ]
    )

    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    print(prefill_ids)

    # Harmony stop tokens (pass to sampler so they won't be included in output)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    print(stop_token_ids)

    # --- 2) Run SGLang with prefill ---
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=int(os.getenv("TP", "1")),
        skip_tokenizer_init=True,
        mem_fraction_static=0.7,
        # quantization="mxfp4",
    )

    # https://docs.sglang.ai/backend/sampling_params.html
    sampling_params = {
        "n": 1,  # number of samples to generate
        "max_new_tokens": kwargs.get("max_new_tokens", 128),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": kwargs.get("top_k", 20),
        "min_p": kwargs.get("min_p", 0.0),
        # Penalizers
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        # "min_new_tokens": kwargs.get("min_new_tokens", 1),
        "stop_token_ids": stop_token_ids,
    }
    for i in range(2):
        start = time.perf_counter()
        # https://github.com/sgl-project/sglang/blob/main/examples/runtime/token_in_token_out/token_in_token_out_llm_engine.py
        outputs = engine.generate(
            input_ids=[prefill_ids],  # batch of size 1
            sampling_params=sampling_params,
        )
        print(f"{i} generate cost:{(time.perf_counter() - start):.3f} s")
        print(outputs)
        assert len(outputs) > 0

        # SGLang gives you both text and token IDs
        output_token_ids = outputs[0]["output_ids"]

        # --- 3) Parse the completion token IDs back into structured Harmony messages ---
        entries = encoding.parse_messages_from_completion_tokens(output_token_ids, Role.ASSISTANT)

        # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
        for message in entries:
            print(json.dumps(message.to_dict(), indent=2))

    engine.shutdown()


def chat_stream(**kwargs):
    """
    use harmony chat format to generate token step by step
    """

    # --- 1) Render the prefill with Harmony ---
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    reasoning_effort = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(REASONING_EFFORT[reasoning_effort])
                .with_model_identity(model_identity)
                .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d")),
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Always respond in riddles"),
            ),
            Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
        ]
    )

    prompt_token_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    # Harmony stop tokens (pass to sampler so they won't be included in output)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    generator = TokenGenerator(
        model_path=model_path, tensor_parallel_size=int(os.getenv("TP", "1"))
    )
    tokenizer = get_tokenizer(model_path)
    prompt_token = tokenizer.decode(prompt_token_ids)
    print(f"{prompt_token=}")

    field_created = False
    current_output_text = ""
    output_text_delta_buffer = ""
    for predicted_token in generator.generate(
        prompt_token_ids,
        stop_tokens=stop_token_ids,
        temperature=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 1.0),
        max_tokens=kwargs.get("max_tokens", 128),
    ):
        parser.process(predicted_token)

        if parser.state == StreamState.EXPECT_START:
            print("")  # new line
            field_created = False

        if not parser.last_content_delta:
            continue

        if not field_created:
            field_created = True
            if parser.current_channel == "final":
                print(termcolor.colored("Assistant:", "green"), flush=True)
            elif parser.current_recipient is not None:
                print(
                    termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"),
                    flush=True,
                )
            else:
                print(termcolor.colored("CoT:", "yellow"), flush=True)

        should_send_output_text_delta = True
        output_text_delta_buffer += parser.last_content_delta
        if should_send_output_text_delta:
            print(output_text_delta_buffer, end="", flush=True)
            current_output_text += output_text_delta_buffer
            output_text_delta_buffer = ""
    print(f"{current_output_text=}")

    print(f"{parser.messages=}")


async def chat_tool_stream(**kwargs):
    generator = TokenGenerator(
        model_path=model_path, tensor_parallel_size=int(os.getenv("TP", "1"))
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    reasoning_effort = kwargs.get("reasoning", "medium")
    model_identity = kwargs.get(
        "model_identity", "You are ChatGPT, a large language model trained by OpenAI."
    )
    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[reasoning_effort])
        .with_model_identity(model_identity)
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    build_in_tool = kwargs.get("build_in_tool", "browser")

    if build_in_tool == "browser":
        backend = ExaBackend(
            source="web",
        )
        browser_tool = SimpleBrowserTool(backend=backend)
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    if build_in_tool == "python":
        python_tool = PythonTool()
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    is_apply_patch = kwargs.get("is_apply_patch", None)
    developer_message = kwargs.get("developer_message", "")
    if is_apply_patch:
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = developer_message + "\n" if developer_message else ""
        developer_message += apply_patch_instructions.read_text()
        developer_message_content = (
            DeveloperContent.new()
            .with_instructions(developer_message)
            .with_function_tools(
                [
                    ToolDescription.new(
                        "apply_patch",
                        "Patch a file",
                        parameters={
                            "type": "string",
                            "description": "Formatted patch code",
                            "default": "*** Begin Patch\n*** End Patch\n",
                        },
                    ),
                ]
            )
        )
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        developer_message_content = DeveloperContent.new().with_instructions(developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))

    raw = kwargs.get("raw", None)
    if raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # System message
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(
            termcolor.colored("Model Identity:", "cyan"),
            system_message_content.model_identity,
            flush=True,
        )
        print(
            termcolor.colored("Reasoning Effort:", "cyan"),
            system_message_content.reasoning_effort,
            flush=True,
        )
        print(
            termcolor.colored("Conversation Start Date:", "cyan"),
            system_message_content.conversation_start_date,
            flush=True,
        )
        print(
            termcolor.colored("Knowledge Cutoff:", "cyan"),
            system_message_content.knowledge_cutoff,
            flush=True,
        )
        print(
            termcolor.colored("Browser Tool:", "cyan"),
            "Enabled" if build_in_tool == "browser" else "Disabled",
            flush=True,
        )
        print(
            termcolor.colored("Python Tool:", "cyan"),
            "Enabled" if build_in_tool == "python" else "Disabled",
            flush=True,
        )
        print(
            termcolor.colored("Apply Patch Function:", "cyan"),
            "Enabled" if is_apply_patch else "Disabled",
            flush=True,
        )
        # Developer message
        print(termcolor.colored("Developer Message:", "yellow"), flush=True)
        print(developer_message_content.instructions, flush=True)

    # Print the system message and the user message start
    MESSAGE_PADDING = 12
    q = kwargs.get("q", None)
    while True:
        last_message = messages[-1]
        if last_message.recipient is None:
            if raw:
                print(user_message_start, end="", flush=True)
                user_message = await get_user_input(q)
                print(user_message_end, flush=True, end="")
            else:
                print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                user_message = await get_user_input(q)
            user_message = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_message)
        else:
            # Tool or function call
            if last_message.recipient.startswith("browser."):
                assert build_in_tool == "browser", "Browser tool is not enabled"
                tool_name = "Search"

                async def run_tool():
                    results = []
                    async for msg in browser_tool.process(last_message):
                        results.append(msg)
                    return results

                result = await run_tool()
                messages += result
            elif last_message.recipient.startswith("python"):
                assert build_in_tool == "python", "Python tool is not enabled"
                tool_name = "Python"

                async def run_tool():
                    results = []
                    async for msg in python_tool.process(last_message):
                        results.append(msg)
                    return results

                result = await run_tool()
                messages += result
            elif last_message.recipient == "functions.apply_patch":
                assert is_apply_patch, "Apply patch tool is not enabled"
                tool_name = "Apply Patch"
                text = last_message.content[0].text
                tool_output = None

                if text.startswith("{"):
                    # this is json, try to extract the patch from it
                    try:
                        some_dict = json.loads(text)
                        _, text = some_dict.popitem()
                    except Exception as e:
                        tool_output = f"Error parsing JSON: {e}"

                if tool_output is None:
                    try:
                        tool_output = apply_patch.apply_patch(text)
                    except Exception as e:
                        tool_output = f"Error applying patch: {e}"

                message = Message(
                    author=Author.new(Role.TOOL, last_message.recipient),
                    content=[TextContent(text=tool_output)],
                ).with_recipient("assistant")
                if last_message.channel:
                    message = message.with_channel(last_message.channel)

                result = [message]
                messages += result
            else:
                msg = f"Unknown tool or function call: {last_message=}"
                print(msg)
                # raise ValueError(f"Unknown tool or function call: {last_message.recipient}")

            # Print the tool or function call result
            if raw:
                rendered_result = encoding.render_conversation(Conversation.from_messages(result))
                print(encoding.decode(rendered_result), flush=True, end="")
            else:
                print(
                    termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"),
                    flush=True,
                )
                show_browser_results = kwargs.get("show_browser_results", None)
                if tool_name == "Search" and not show_browser_results:
                    print("[Search results fed to the model]")
                else:
                    print(result[0].content[0].text)

        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        if raw:
            # Print the last two tokens, which are the start of the assistant message
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        async for predicted_token in generator.async_generate(
            tokens,
            stop_tokens=encoding.stop_tokens_for_assistant_actions(),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=kwargs.get("max_tokens", 2048),
        ):
            parser.process(predicted_token)
            if raw:
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            if parser.state == StreamState.EXPECT_START:
                print("")  # new line
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    print(
                        termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"),
                        flush=True,
                    )
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta
            if build_in_tool == "browser":
                updated_output_text, _annotations, has_partial_citations = (
                    browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
                )
                output_text_delta_buffer = updated_output_text[len(current_output_text) :]
                if has_partial_citations:
                    should_send_output_text_delta = False
            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

        messages += parser.messages


async def get_user_input(q: modal.Queue):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        # user_input = input()
        user_input = await q.get.aio(partition="text")
    else:
        user_input = ""
    user_input_list = [user_input]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(user_input_list, 0)
    return user_input_list[0]


@app.function(
    gpu=SERVE_IMAGE_GPU,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        EVAL_OUTPUT_DIR: eval_out_vol,
    },
    # secrets=[modal.Secret.from_name("achatbot")],
    timeout=86400,  # default 300s
    max_containers=SERVE_MAX_CONTAINERS,
)
@modal.web_server(port=30000, startup_timeout=60 * 60)
@modal.concurrent(max_inputs=100, target_inputs=4)
def serve():
    """
    modal + sglang :
    - https://modal.com/docs/examples/sgl_vlm
    - https://modal.com/llm-almanac/advisor
    """
    run_cmd("python3 -m sglang.launch_server --help")

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    cmd = f"""
    python3 -m sglang.launch_server --model {model_path} --host 0.0.0.0 --port 30000 \\
        --tp {os.getenv("TP", "1")} {os.getenv("SERVE_ARGS", "")}
    """
    print(cmd)
    subprocess.Popen(cmd, shell=True, env=os.environ)


def local_api_completions(**kwargs):
    from openai import OpenAI

    url = serve.get_web_url()
    print(url)

    client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    result = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what MXFP4 quantization is."},
        ],
    )
    print(result)
    print(f"{result.choices[0].message.content=}")
    print(f"{result.choices[0].message.reasoning_content=}")
    print(result.usage)


def local_api_tool_completions(**kwargs):
    from openai import OpenAI

    url = serve.get_web_url()
    print(url)

    client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    result = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
        tools=tools,
    )
    print(result)
    print(f"{result.choices[0].message.content=}")
    print(f"{result.choices[0].message.reasoning_content=}")
    print(result.usage)


@app.local_entrypoint()
def url_request(test_timeout=30 * 60):
    url = serve.get_web_url()
    print(f"Running health check for server at {url}")
    print("Note: startup takes a while on the first two iterations, but is much faster after that.")
    print("On the first iteration with a new model, weights are downloaded at ~100 MB/s.")
    print("On the second iteration, a file read profile is recorded and used for future runs.")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(url + "/health") as response:
                up = response.getcode() == 200
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {url}"

    print(f"Successful health check for server at {url}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {url}", *messages, sep="\n")

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"messages": messages, "model": MODEL_PATH, "max_tokens": 128})
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))


def run_async_in_thread(coroutine, **kwargs):
    def run_coro_in_thread():
        asyncio.run(coroutine(**kwargs))

    thread = threading.Thread(target=run_coro_in_thread)
    thread.start()
    return thread


async def loop_input(q: modal.Queue):
    while True:
        text = input()
        print(f"local input: {text}")
        await q.put.aio(text, partition="text")


"""
# 0. download model weight(safetensors), tokenizer and config
modal run src/download_models.py --repo-ids "lmsys/gpt-oss-20b-bf16"
modal run src/download_models.py --repo-ids "lmsys/gpt-oss-120b-bf16"
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"
modal run src/download_models.py --repo-ids "openai/gpt-oss-120b" --ignore-patterns "*.pt|*.bin|*original*|*metal*"


# 1. run server and test with urllib request raw http api (dev)
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::url_request
# mxfp4 
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::url_request


# 2. run server and test with openai client sdk (dev/test)
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions 
LLM_MODEL=lmsys/gpt-oss-120b-bf16 SERVE_IMAGE_GPU=H200:4 TP=4 SERVE_ARGS="--cuda-graph-max-bs 2" modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions 
# mxfp4 
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_tool_completions
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_tool_completions
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task local_api_completions 


# 3. benchmark
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 40 --max-concurrency 8
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 80 --max-concurrency 16
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 160 --max-concurrency 32
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 320 --max-concurrency 64
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 80 --max-concurrency 16
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 160 --max-concurrency 32
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 20 --max-concurrency 4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 80 --max-concurrency 16
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=B200 TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 160 --max-concurrency 32
LLM_MODEL=lmsys/gpt-oss-120b-bf16 SERVE_IMAGE_GPU=H100:8 TP=8 modal run src/llm/sglang/openai_gpt_oss.py::main --task benchmark --num-prompts 320 --max-concurrency 64

# 4. run server (gray)
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=H100 TP=1 modal serve src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal serve src/llm/sglang/openai_gpt_oss.py
# mxfp4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 modal serve src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_ARGS="--cuda-graph-max-bs 4" modal serve src/llm/sglang/openai_gpt_oss.py

# 5. deploy (online)
# fp8/bf16
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=H100 TP=1 SERVE_MAX_CONTAINERS=10 modal deploy src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=lmsys/gpt-oss-20b-bf16 SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_MAX_CONTAINERS=10 SERVE_ARGS="--cuda-graph-max-bs 4" modal serve src/llm/sglang/openai_gpt_oss.py
# mxfp4
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=H100 TP=1 SERVE_MAX_CONTAINERS=10 modal deploy src/llm/sglang/openai_gpt_oss.py
LLM_MODEL=openai/gpt-oss-20b SERVE_IMAGE_GPU=L40s:2 TP=2 SERVE_MAX_CONTAINERS=10 SERVE_ARGS="--cuda-graph-max-bs 4" modal deploy src/llm/sglang/openai_gpt_oss.py


===================

# generate
LLM_MODEL=openai/gpt-oss-20b RUN_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task generate
LLM_MODEL=openai/gpt-oss-20b RUN_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task generate_stream
LLM_MODEL=openai/gpt-oss-20b RUN_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task batch_generate_stream
LLM_MODEL=openai/gpt-oss-20b RUN_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task harmony_generate 
LLM_MODEL=openai/gpt-oss-20b RUN_IMAGE_GPU=A100-80GB TP=1 modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_stream

# local input --- queue --> remote to loop chat

## use browser tool(find,open,search), need env EXA_API_KEY from https://exa.ai
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool browser
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream \
    --max-tokens 2048 --temperature=1.0 --top-p=1.0 \
    --build-in-tool browser --show-browser-results --model-identity "你是一名聊天助手，请用中文回复。"
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool browser --is-apply-patch --show-browser-results
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool browser --raw --is-apply-patch --show-browser-results

## need python tool to run script need change python docker, u can::main change python tools to do local env python or use serverless function 
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool python
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool python --is-apply-patch 
RUN_IMAGE_GPU=A100-80GB modal run src/llm/sglang/openai_gpt_oss.py::main --task chat_tool_stream --build-in-tool python --raw --is-apply-patch
"""


@app.local_entrypoint()
def main(
    task: str = "generate",
    reasoning: str = "medium",  # low medium high
    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI.",
    prompt: str = "什么是快乐星球?",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    # for chat_tool_stream
    build_in_tool: str = "browser",  # build-in tools: browser,python
    is_apply_patch: bool = False,  # Make apply_patch tool available to the model (default: False)
    show_browser_results: bool = False,  # Show browser results (default: False)
    developer_message: str = "",  # Developer message (default: )
    raw: bool = False,  # Raw mode (does not render Harmony encoding) (default: False)
    # benchmark
    num_prompts: int = 5,
    max_concurrency: int = 1,
):
    print(task)
    tasks = {
        # local test
        "local_api_completions": local_api_completions,
        "local_api_tool_completions": local_api_tool_completions,
        # benchmark
        "benchmark": benchmark,
        # use SGLang Engine to generate
        "generate": generate,
        "batch_generate_stream": batch_generate_stream,
        "generate_stream": generate_stream,
        "harmony_generate": harmony_generate,
        "chat_stream": chat_stream,
        "chat_tool_stream": chat_tool_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    if "local" in task:
        func = run
    else:
        func = remote_run.remote

    with modal.Queue.ephemeral() as q:
        if task == "chat_tool_stream":
            run_async_in_thread(loop_input, q=q)

        func(
            tasks[task],
            reasoning=reasoning.lower(),
            model_identity=model_identity,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            build_in_tool=build_in_tool,
            is_apply_patch=is_apply_patch,
            show_browser_results=show_browser_results,
            developer_message=developer_message,
            raw=raw,
            num_prompts=num_prompts,
            max_concurrency=max_concurrency,
            q=q,
        )
