import datetime
import os
from pathlib import Path
import sys
import asyncio
import subprocess
import threading
import json
from time import perf_counter


import modal


app = modal.App("openai_gpt_oss")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
BACKEND = os.getenv("BACKEND", "")
APP_NAME = os.getenv("APP_NAME", "")
TP = os.getenv("TP", "1")

img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "uv pip install --system --pre --no-cache vllm==0.10.1+gptoss "
        "--extra-index-url https://wheels.vllm.ai/gpt-oss/ "
        "--extra-index-url https://download.pytorch.org/whl/nightly/cu128 "
        "--index-strategy unsafe-best-match",
    )
    .pip_install("openai-harmony")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            "VLLM_USE_V1": "1",
            "TP": TP,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TORCH_CUDA_ARCH_LIST": "8.0 8.9 9.0+PTX",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
        }
    )
    .run_commands("git clone https://github.com/weedge/gpt-oss.git")
)
if BACKEND == "flashinfer":
    img = img.pip_install(
        f"flashinfer-python",
        extra_index_url="https://wheels.vllm.ai/gpt-oss/",
    )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
VLLM_CACHE_DIR = "/root/.cache/vllm"
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


with img.imports():
    import termcolor
    import torch
    from vllm import LLM, LLMEngine, EngineArgs, SamplingParams, TokensPrompt
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

    sys.path.insert(0, "/gpt-oss")
    from gpt_oss.tools import apply_patch
    from gpt_oss.tools.simple_browser import SimpleBrowserTool
    from gpt_oss.tools.simple_browser.backend import ExaBackend
    from gpt_oss.tools.python_docker.docker_tool import PythonTool

    REASONING_EFFORT = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }

    MODEL_PATH = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    model_path = os.path.join(HF_MODEL_DIR, MODEL_PATH)
    model_name = MODEL_PATH.split("/")[-1]

    class TokenGenerator:
        def __init__(
            self,
            model_path: str,
            tensor_parallel_size: int = 1,
            gpu_memory_utilization: float = 0.7,
        ):
            args = EngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                # quantization="mxfp4",
            )
            self.engine = LLMEngine.from_engine_args(args)
            self.request_id = 0

        def generate(
            self,
            prompt_tokens: list[int],
            stop_tokens: list[int] | None = None,
            temperature: float = 1.0,
            top_p: float = 1.0,
            max_tokens: int = 0,
            return_logprobs: bool = False,
        ):
            if max_tokens == 0:
                max_tokens = None
            request_id = str(self.request_id)
            self.request_id += 1
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_tokens,
                logprobs=0 if return_logprobs else None,
            )
            prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            self.engine.add_request(request_id, prompt, sampling_params)
            last_token_id = []
            while self.engine.has_unfinished_requests():
                step_outputs = self.engine.step()
                output = step_outputs[0].outputs[0]
                token_ids = output.token_ids
                logprobs_list = output.logprobs if hasattr(output, "logprobs") else None
                new_token_ids = token_ids[len(last_token_id) :]
                new_logprobs = (
                    logprobs_list[len(last_token_id) :]
                    if logprobs_list is not None
                    else [None] * len(new_token_ids)
                )
                for token_id, logprobs in zip(new_token_ids, new_logprobs):
                    last_token_id.append(token_id)
                    if return_logprobs:
                        logprob_val = None
                        if logprobs is not None and token_id in logprobs:
                            logprob_val = logprobs[token_id].logprob
                        yield (token_id, logprob_val)
                    else:
                        yield token_id
                    if stop_tokens is not None and token_id in stop_tokens:
                        break


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("achatbot")],
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)
        assert gpu_prop.major >= 8, f"now vllm gpt oss model just support cuda arch sm8.0+ :)"

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def get_tokenizer():
    import tiktoken

    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )
    return tokenizer


def tokenizer(**kwargs):
    tokenizer = get_tokenizer()
    stop_ids = tokenizer.encode("<|endoftext|>", allowed_special="all")
    print(stop_ids)
    assert tokenizer.eot_token == stop_ids[0]

    prompt = kwargs.get("prompt", "什么是快乐星球?")
    token_ids = tokenizer.encode(prompt)
    print(f"promt_token_ids: {token_ids}")
    tokens = tokenizer.decode(token_ids)
    print(f"promt_tokens: {tokens}")


def harmony_chat_tokenizer(**kwargs):
    tokenizer = get_tokenizer()

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

    prompt_token = tokenizer.decode(prompt_token_ids)
    print(prompt_token_ids, prompt_token)

    print("\n" + "----" * 20 + "\n")

    stop_tokens = tokenizer.decode(stop_token_ids)
    print(stop_tokens, stop_token_ids)


def generate(**kwargs):
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

    # Harmony stop tokens (pass to sampler so they won't be included in output)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    print(stop_token_ids)

    # --- 2) Run vLLM with prefill ---
    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(os.getenv("TP", "1")),
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        quantization="mxfp4",
    )

    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=128,
        repetition_penalty=1.1,
        stop_token_ids=stop_token_ids,
    )
    for i in range(2):
        start = perf_counter()
        outputs = llm.generate(
            prompt_token_ids=[prefill_ids],  # batch of size 1
            sampling_params=sampling,
        )
        print(f"{i} generate cost:{(perf_counter() - start):.3f} s")
        print(outputs)

        # vLLM gives you both text and token IDs
        gen = outputs[0].outputs[0]
        text = gen.text
        print(f"generate text: {text}")
        output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)

        # --- 3) Parse the completion token IDs back into structured Harmony messages ---
        entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

        # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
        for message in entries:
            print(json.dumps(message.to_dict(), indent=2))


def generate_stream(**kwargs):
    """
    no chat instruct, generate token step by step with tiktoken(o200k_harmony) encode/decode
    """
    generator = TokenGenerator(
        model_path=model_path, tensor_parallel_size=int(os.getenv("TP", "1"))
    )
    tokenizer = get_tokenizer()

    temperature = kwargs.get("temperature", 1.0)
    max_tokens = kwargs.get("max_tokens", 128)
    top_p = kwargs.get("top_p", 1.0)

    for i in range(3):  # compile, need warmup
        prompt = kwargs.get("prompt", "什么是快乐星球?")
        token_ids = tokenizer.encode(prompt)
        print(f"promt_token_ids: {token_ids}")

        times = []
        start = perf_counter()
        text = ""
        for token_id, logprob in generator.generate(
            token_ids,
            stop_tokens=[tokenizer.eot_token],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            return_logprobs=True,
        ):
            token_ids.append(token_id)
            decoded_token = tokenizer.decode(
                [token_id]
            )  # NOTE: unicode decode need stream parser, need to stop
            times.append(perf_counter() - start)
            text += decoded_token
            print(f"Generated token: {repr(decoded_token)}, logprob: {logprob}")
            start = perf_counter()

        print(f"{i} generated Text: {text}")
        print(f"{i} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")


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
    tokenizer = get_tokenizer()
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
        for predicted_token in generator.generate(
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


@app.function(
    gpu=IMAGE_GPU,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    # secrets=[modal.Secret.from_name("achatbot")],
    timeout=86400,  # default 300s
    # max_containers=1,
)
@modal.web_server(port=8000, startup_timeout=60 * 60)
@modal.concurrent(max_inputs=100, target_inputs=4)
def serve():
    """
    modal + vllm :
    - https://modal.com/docs/examples/vllm_inference
    - https://modal.com/llm-almanac/advisor
    """
    cmd = f"""
    VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve {model_path} --async-scheduling --served-model-name {model_name} --trust_remote_code --port 8000 --tensor-parallel-size {os.getenv("TP", "1")}
    """
    subprocess.Popen(cmd, shell=True, env=os.environ)


"""
# download hf tranformers weight(safetensors) for vllm to load
modal run src/download_models.py --repo-ids "openai/gpt-oss-20b"
modal run src/download_models.py --repo-ids "openai/gpt-oss-120b" --ignore-patterns "*.pt|*.bin|*original*|*metal*"

# see help
modal run src/llm/vllm/openai_gpt_oss.py --help 


# tokenizer
modal run src/llm/vllm/openai_gpt_oss.py --task tokenizer
modal run src/llm/vllm/openai_gpt_oss.py --task harmony_chat_tokenizer

# generate
IMAGE_GPU=A100 modal run src/llm/vllm/openai_gpt_oss.py --task generate
IMAGE_GPU=L40s modal run src/llm/vllm/openai_gpt_oss.py --task generate
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task generate

IMAGE_GPU=A100 modal run src/llm/vllm/openai_gpt_oss.py --task generate_stream
IMAGE_GPU=L40s modal run src/llm/vllm/openai_gpt_oss.py --task generate_stream
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task generate_stream

# chat
IMAGE_GPU=A100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_stream
IMAGE_GPU=L40s modal run src/llm/vllm/openai_gpt_oss.py --task chat_stream
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_stream

# local input --- queue --> remote to loop chat

## use browser tool(find,open,search), need env EXA_API_KEY from https://exa.ai
IMAGE_GPU=L40s modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool browser
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool browser
IMAGE_GPU=L40s modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream \
    --max-tokens 2048 --temperature=1.0 --top-p=1.0 \
    --build-in-tool browser --show-browser-results --model-identity "你是一名聊天助手，请用中文回复。"
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream \
    --max-tokens 2048 --temperature=1.0 --top-p=1.0 \
    --build-in-tool browser --show-browser-results --model-identity "你是一名聊天助手，请用中文回复。"
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool browser --is-apply-patch --show-browser-results
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool browser --raw --is-apply-patch --show-browser-results

## need python tool to run script need change python docker, u can change python tools to do local env python or use serverless function 
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool python
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool python --is-apply-patch 
IMAGE_GPU=H100 modal run src/llm/vllm/openai_gpt_oss.py --task chat_tool_stream --build-in-tool python --raw --is-apply-patch

# run vllm serve
IMAGE_GPU=H100 modal serve src/llm/vllm/openai_gpt_oss.py 
IMAGE_GPU=H200 modal serve src/llm/vllm/openai_gpt_oss.py 
# first ping to start serve
curl -XGET "https://weedge--openai-gpt-oss-serve-dev.modal.run/ping"
```
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /docs, Methods: HEAD, GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /redoc, Methods: HEAD, GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /health, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /load, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /ping, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /ping, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /tokenize, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /detokenize, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/models, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /version, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/responses, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/completions, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/embeddings, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /pooling, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /classify, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /score, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/score, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /rerank, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v1/rerank, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /v2/rerank, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /invocations, Methods: POST
(APIServer pid=28) INFO 08-09 07:26:21 [launcher.py:37] Route: /metrics, Methods: GET
```
"""


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


@app.local_entrypoint()
def main(
    task: str = "tokenizer",
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
):
    print(task)
    tasks = {
        "tokenizer": tokenizer,
        "harmony_chat_tokenizer": harmony_chat_tokenizer,
        "generate": generate,
        "generate_stream": generate_stream,
        "chat_stream": chat_stream,
        "chat_tool_stream": chat_tool_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    with modal.Queue.ephemeral() as q:
        if task == "chat_tool_stream":
            run_async_in_thread(loop_input, q=q)

        run.remote(
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
            q=q,
        )
