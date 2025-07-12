# author: weedge (weege007@gmail.com)

import os
import subprocess
from threading import Thread
from time import perf_counter
import time
import traceback
from typing import Generator, Optional
import uuid

import modal


APP_NAME = os.getenv("APP_NAME", None)
LLM_MODEL = os.getenv("LLM_MODEL", "baidu/ERNIE-4.5-0.3B-Paddle")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100")
FASTDEPLOY_VERSION = os.getenv("FASTDEPLOY_VERSION", "stable")  # stable, nightly
GPU_ARCHS = os.getenv("GPU_ARCHS", "80_90")  # 80_90, 86_89
QUANTIZATION = os.getenv("quantization", "wint4")  # wint8, wint4
TP = os.getenv("TP", "1")
app = modal.App("fastdeploy-offline-inference")
img = (
    # use openai triton
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/nvidia_gpu.md
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        # "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.0.0",
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "paddlepaddle-gpu==3.1.0",
        index_url=" https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .run_commands(
        f"python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/{FASTDEPLOY_VERSION}/fastdeploy-gpu-{GPU_ARCHS}/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    )
    .env(
        {
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": LLM_MODEL,
            "IMAGE_GPU": IMAGE_GPU,
            "QUANTIZATION": QUANTIZATION,
            "TP": TP,
        }
    )
)


if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.21",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "A100"),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=86400,  # default 300s
    max_containers=1,
)
def run(task, thinking=True):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)

    task(thinking)


def check(thinking):
    import paddle
    from paddle.jit.marker import unified

    print(paddle.device.get_device())  # 输出类似于'gpu:0'

    # Verify GPU availability
    paddle.utils.run_check()
    # Verify FastDeploy custom operators compilation
    from fastdeploy.model_executor.ops.gpu import beam_search_softmax


def generate(thinking):
    import paddle
    from fastdeploy import LLM, SamplingParams

    gpu_device_count = paddle.device.cuda.device_count()

    prompts = [
        "把李白的静夜思改写为现代诗",
        "Write me a poem about large language model.",
    ]

    # Sampling parameters
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
    sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

    # Load model
    LLM_MODEL = os.getenv("LLM_MODEL")
    model_path = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md
    llm = LLM(model=model_path, tensor_parallel_size=gpu_device_count, max_model_len=8192)

    # Batch inference (internal request queuing and dynamic batching)
    outputs = llm.generate(prompts, sampling_params)

    # Output results
    for output in outputs:
        print(output)


def chat(thinking):
    import paddle
    from fastdeploy import LLM, SamplingParams

    gpu_device_count = paddle.device.cuda.device_count()
    print(f"{gpu_device_count=}")

    msg1 = [
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ]
    msg2 = [
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "Write me a poem about large language model."},
    ]
    messages = [msg1, msg2]

    # Sampling parameters
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
    sampling_params = SamplingParams(top_p=0.95, max_tokens=6400)

    # Load model
    LLM_MODEL = os.getenv("LLM_MODEL")
    model_path = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    # https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md
    llm = LLM(model=model_path, tensor_parallel_size=gpu_device_count, max_model_len=8192)
    # Batch inference (internal request queuing and dynamic batching)
    outputs = llm.chat(messages, sampling_params)

    # Output results
    for output in outputs:
        print(output)


def vision_chat(thinking):
    import io
    import os
    import requests
    from PIL import Image

    from fastdeploy.entrypoints.llm import LLM
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
    from fastdeploy.utils import llm_logger, retrive_model_from_server

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)

    vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
    for i in range(len(vocab_file_names)):
        if os.path.exists(os.path.join(PATH, vocab_file_names[i])):
            ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
            break
    tokenizer = ErnieBotTokenizer.from_pretrained(PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                {"type": "text", "text": "这张图片的内容是什么"},
            ],
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    images, videos = [], []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                image_bytes = requests.get(url).content
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            elif part["type"] == "video_url":
                url = part["video_url"]["url"]
                video_bytes = requests.get(url).content
                videos.append({"video": video_bytes, "max_frames": 30})

    sampling_params = SamplingParams(temperature=0.1, max_tokens=6400)
    print(f"{sampling_params=}")

    class LLMv2(LLM):
        def _receive_output(self):
            """
            Recieve output from token processor and store them in cache
            """
            while True:
                try:
                    results = self.llm_engine._get_generated_result()
                    for request_id, contents in results.items():
                        with self.mutex:
                            for result in contents:
                                print(request_id, result)
                                if request_id not in self.req_output:
                                    self.req_output[request_id] = result
                                    continue
                                self.req_output[request_id].add(result)
                except Exception as e:
                    llm_logger.error(
                        "Unexcepted error happend: {}, {}".format(e, str(traceback.format_exc()))
                    )

    llm = LLMv2(
        model=PATH,
        tensor_parallel_size=int(os.getenv("TP", 1)),
        quantization=os.getenv("QUANTIZATION", "wint4"),
        max_model_len=32768,
        enable_mm=True,
        limit_mm_per_prompt={"image": 100},
        reasoning_parser="ernie-45-vl",
    )
    outputs = llm.generate(
        prompts={"prompt": prompt, "multimodal_data": {"image": images, "video": videos}},
        sampling_params=sampling_params,
    )

    # 输出结果
    for output in outputs:
        prompt = output.prompt
        print(prompt, output)


def llm_engine_generate(thinking):
    import io
    import os
    import requests
    from PIL import Image

    from fastdeploy.entrypoints.llm import LLM
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
    from fastdeploy.utils import FlexibleArgumentParser, api_server_logger, is_port_available
    from fastdeploy.engine.args_utils import EngineArgs
    from fastdeploy.utils import llm_logger, retrive_model_from_server
    from fastdeploy.engine.engine import LLMEngine
    from fastdeploy.engine.request import RequestOutput

    class LLMEngineMonkey(LLMEngine):
        def _get_generated_tokens(self, request_id) -> Generator[RequestOutput, None, None]:
            """
            Get generated tokens for a specific request ID.
            This is a generator function that yields results until the generation is complete.

            Args:
                request_id (str): The ID of the request to get tokens for.

            Yields:
                RequestOutput: The generated tokens for the request.
            """
            finished = False
            while not finished and self.running:
                try:
                    results = self.scheduler.get_results()
                    if request_id in results:
                        contents = results[request_id]
                        for result in contents:
                            # print(request_id, result)
                            yield result
                            if result.finished:
                                finished = True
                                break
                    if not finished:
                        time.sleep(0.001)  # Small sleep to avoid busy waiting
                except Exception as e:
                    llm_logger.error(
                        f"Error in _get_generated_tokens: {e}, {str(traceback.format_exc())}"
                    )
                    break

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    llm_engine = LLMEngineMonkey.from_engine_args(
        EngineArgs(
            model=PATH,
            tensor_parallel_size=int(os.getenv("TP", 1)),
            quantization=os.getenv("QUANTIZATION", "wint4"),
            max_model_len=32768,
            enable_mm=True,
            limit_mm_per_prompt={"image": 10},
            reasoning_parser="ernie-45-vl",
        )
    )
    if not llm_engine.start():
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return
    api_server_logger.info(f"FastDeploy LLM engine initialized!")

    vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
    for i in range(len(vocab_file_names)):
        if os.path.exists(os.path.join(PATH, vocab_file_names[i])):
            ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
            break
    tokenizer = ErnieBotTokenizer.from_pretrained(PATH)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                {"type": "text", "text": "这张图片的内容是什么"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    images, videos = [], []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                image_bytes = requests.get(url).content
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            elif part["type"] == "video_url":
                url = part["video_url"]["url"]
                video_bytes = requests.get(url).content
                videos.append({"video": video_bytes, "max_frames": 30})

    prompts = {"prompt": prompt, "multimodal_data": {"image": images, "video": videos}}
    prompts["request_id"] = str(uuid.uuid4())
    prompts["max_tokens"] = llm_engine.cfg.max_model_len
    print(f"{prompts=}")
    sampling_params = SamplingParams(temperature=0.1, max_tokens=6400, reasoning_max_tokens=1)
    print(f"{sampling_params=} {thinking=}")
    # https://paddlepaddle.github.io/FastDeploy/offline_inference/#text-completion-interface-llmgenerate
    # The generate interface does not currently support passing parameters to control the thinking function (on/off). It always uses the model's default parameters.
    llm_engine.add_requests(prompts, sampling_params, enable_thinking=thinking)

    for result in llm_engine._get_generated_tokens(prompts["request_id"]):
        # print(result)
        if result.outputs and result.outputs.token_ids and len(result.outputs.token_ids) > 0:
            # print(result.outputs.token_ids)
            tokens = tokenizer.decode(result.outputs.token_ids)
            print(tokens, flush=True, end="")

        if result.finished:
            print("\n")
            print(prompts["request_id"], "finished")


def achatbot_engine_generate(thinking):
    import uuid
    import time
    import PIL
    import io
    import os
    import requests
    from PIL import Image

    from fastdeploy.engine.args_utils import EngineArgs
    from achatbot.common.types import SessionCtx, TEST_DIR
    from achatbot.types.llm.fastdeploy import FastDeployEngineArgs, LMGenerateArgs
    from achatbot.core.llm.fastdeploy.vision_ernie4v import FastdeployVisionERNIE4v
    from achatbot.common.session import Session
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    LLM_MODEL = os.getenv("LLM_MODEL")
    PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    generator = FastdeployVisionERNIE4v(
        # https://paddlepaddle.github.io/FastDeploy/parameters/
        **FastDeployEngineArgs(
            serv_args=EngineArgs(
                model=PATH,
                tensor_parallel_size=int(os.getenv("TP", 1)),
                quantization=os.getenv("QUANTIZATION", "wint4"),
                max_model_len=32768,
                enable_mm=True,
                limit_mm_per_prompt={"image": 10, "video": 1},
                reasoning_parser="ernie-45-vl",
                gpu_memory_utilization=0.6,
                use_warmup=0,
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                use_cudagraph=False,
                enable_expert_parallel=False,
            ).__dict__,
            init_chat_prompt="你是一个语音聊天智能助手，不要使用特殊字符回复，请用中文交流。",
            chat_history_size=2,
        ).__dict__,
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
    image_bytes = requests.get(url).content
    img = Image.open(io.BytesIO(image_bytes))
    chat_texts = ["这张图片的内容是什么", "你叫什么名字", "讲个故事"]
    for chat_text in chat_texts:
        session.ctx.state["prompt"] = [
            {"type": "image_url", "image_url": img},
            {"type": "text", "text": chat_text},
        ]
        for result_text in generator.generate(session, thinking=thinking):
            print(result_text, flush=True, end="")
    img.close()


"""
# 0. download paddle model
modal run src/download_models.py --repo-ids "baidu/ERNIE-4.5-0.3B-Paddle"
modal run src/download_models.py --repo-ids "baidu/ERNIE-4.5-VL-28B-A3B-Paddle"

# https://paddlepaddle.github.io/FastDeploy/offline_inference/
# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/offline_inference.md#24-fastdeploysamplingparams
# https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md

# 1. default 80_90 GPU ARCH use A100 run check/generate/chat
modal run src/llm/fastdeploy/offline_inference.py
modal run src/llm/fastdeploy/offline_inference.py --task generate
modal run src/llm/fastdeploy/offline_inference.py --task chat

# 2. 86_89 GPU ARCH use L4 run check/generate/chat
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py --task generate
GPU_ARCHS=86_89 IMAGE_GPU=L4 modal run src/llm/fastdeploy/offline_inference.py --task chat 

# 3. 86_89 GPU ARCH use L40s run vision_chat
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s modal run src/llm/fastdeploy/offline_inference.py --task vision_chat 
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s QUANTIZATION=wint8 TP=1 modal run src/llm/fastdeploy/offline_inference.py --task vision_chat 

# 4. 86_89 GPU ARCH use L40s run vision streaming generate
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s QUANTIZATION=wint4 TP=1 modal run src/llm/fastdeploy/offline_inference.py --task llm_engine_generate 
LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s QUANTIZATION=wint8 TP=1 modal run src/llm/fastdeploy/offline_inference.py --task llm_engine_generate

# 5. 86_89 GPU ARCH use L40s run achatbot vision streaming generate
APP_NAME=achatbot LLM_MODEL=baidu/ERNIE-4.5-VL-28B-A3B-Paddle GPU_ARCHS=86_89 IMAGE_GPU=L40s QUANTIZATION=wint4 TP=1 modal run src/llm/fastdeploy/offline_inference.py --task achatbot_engine_generate
"""


@app.local_entrypoint()
def main(task: str = "check", thinking: bool = True):
    tasks = {
        "check": check,
        "generate": generate,
        "chat": chat,
        "vision_chat": vision_chat,
        "llm_engine_generate": llm_engine_generate,
        "achatbot_engine_generate": achatbot_engine_generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task} {thinking=}")
    run.remote(tasks[task], thinking)
