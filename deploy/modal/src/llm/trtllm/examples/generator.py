# https://nvidia.github.io/TensorRT-LLM/index.html (nice doc)
"""
NeMo -------------
                  |
HuggingFace ------
                  |   convert                             build                    load
Modelopt ---------  ----------> TensorRT-LLM Checkpoint --------> TensorRT Engine ------> TensorRT-LLM ModelRunner
                  |
JAX --------------
                  |
DeepSpeed --------
"""

import os
import modal

app_name = "qwen2.5-0.5B"

app = modal.App("trtllm-generator")

trtllm_image = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "git-lfs", "openmpi-bin", "libopenmpi-dev", "wget"
    )  # OpenMPI for distributed communication
    .pip_install(
        "tensorrt-llm==0.17.0.post1",
        # "pynvml<12",  # avoid breaking change to pynvml version API for tensorrt_llm
        # "tensorrt==10.8.0.43",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.7 8.9 9.0"})
)

achatbot_trtllm_image = trtllm_image.pip_install(
    "achatbot==0.0.9.post2",
    extra_index_url="https://pypi.org/simple/",
).env(
    {
        "TLLM_LLMAPI_BUILD_CACHE": "1",
    }
)

MAX_BATCH_SIZE = 1024  # better throughput at larger batch sizes, limited by GPU RAM
MODEL_ID = "Qwen/Qwen2.5-0.5B"

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
TRT_MODEL_DIR = "/root/trt_models"
trt_model_vol = modal.Volume.from_name("triton_trtllm_models", create_if_missing=True)

TRT_MODEL_CACHE_DIR = "/tmp/.cache/tensorrt_llm/llmapi/"
trt_model_cache_vol = modal.Volume.from_name("triton_trtllm_cache_models", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run_sync():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate
    """
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.bindings.executor import KvCacheConfig

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/llmapi/llm_args.py#L520
    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5)
    # load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine
    llm = LLM(model=MODEL_ID, kv_cache_config=kv_cache_config)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    # Print the outputs.
    for output in outputs:
        print(output)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image.env({}),
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async_stream():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    from tensorrt_llm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    # https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2, detokenize=False)

    # load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine
    llm = LLM(model=MODEL_ID)

    for i, prompt in enumerate(prompts):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for output in generator:
            print(output)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_async_batch_stream():
    """
    https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/llmapi/llm.html#LLM.generate_async
    """
    import asyncio
    import uuid

    from tensorrt_llm import LLM, SamplingParams

    # Prompts to generate
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=2, detokenize=True, return_perf_metrics=True
    )

    # Load HF model, convert to TensorRT, build TensorRT engine, load TensorRT engine
    llm = LLM(model=MODEL_ID)

    lock = asyncio.Lock()

    async def run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex)):
        generator = llm.generate_async(prompt, sampling_params, streaming=True)
        async for item in generator:
            print(f"[{request_id}] tokenId: {item.outputs[0].token_ids[-1]} {item} ")
            async with lock:
                print(f"[{request_id}] {item}")
                print(item.outputs[0].token_ids[-1])
                # u can send this response to a request queue/channle

    tasks = [
        run_async_stream(llm, prompt, sampling_params, request_id=str(uuid.uuid4().hex))
        for prompt in prompts
    ]
    await asyncio.gather(*tasks)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def runner_stream():
    """
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L548
    """
    import time

    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
    from transformers import AutoTokenizer

    init_start = time.monotonic_ns()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(HF_MODEL_DIR, MODEL_ID))
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    # and then we add it from the left, to minimize impact on the output
    # tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id
    print(f"pad {tokenizer.pad_token} {pad_id}")
    print(f"eos {tokenizer.eos_token} {end_id}")

    prompt_cases = [
        {"prompt": "hello, my name is", "kwargs": {"max_new_tokens": 10, "stop_words_list": None}},
        {
            "prompt": "hello, my name is",
            "kwargs": {
                "max_new_tokens": 10,
                "stop_words_list": [[[end_id]]],  # one prompt batch stop ids
            },
        },  # prefill cache token test (default no cache)
        {
            "prompt": "hello, what your name?",
            "kwargs": {
                "max_new_tokens": 10,
            },
        },
    ]

    # load tensorrt engine
    # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.from_dir
    engine = ModelRunner.from_dir(
        engine_dir=os.path.join(TRT_MODEL_DIR, "qwen2.5-0.5B", "trt_engines_bfloat16"),
        rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        max_output_len=100,
        debug_mode=True,
    )

    init_duration_s = (time.monotonic_ns() - init_start) / 1e9
    print(f"ModelRunner init duration: {init_duration_s:.2f} s")

    for i, case in enumerate(prompt_cases):
        # https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/runtime/generation.html#SamplingConfig
        sampling_config = SamplingConfig(
            end_id=end_id,
            pad_id=pad_id,
            temperature=0.8,
            repetition_penalty=1.1,
            max_new_tokens=50,
            top_k=20,
            top_p=0.95,
            # min_p version > 0.17.0
            # min_p=0.0,
            stop_words_list=None,
        )
        sampling_config.update(**case["kwargs"])
        print("sampling_config:", sampling_config)
        tokens = tokenizer([case["prompt"]], return_tensors="pt", padding=False, truncation=False)
        input_ids = tokens["input_ids"]
        print("input_ids:", input_ids)
        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.generate
        generator = engine.generate(
            input_ids,
            streaming=True,
            sampling_config=sampling_config,
        )
        first = True
        start_time = time.perf_counter()
        for output in generator:
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            print(output)
            output = output[:, 0][0]
            # mask = output != pad_id
            # output = output[mask]
            outputs_text = tokenizer.decode(output, skip_special_tokens=True)
            print(output, outputs_text)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def runner_batch_stream():
    """
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L548
    """
    import time
    import random

    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
    from transformers import AutoTokenizer

    # from trtllm-build --max_batch_size 16
    # better throughput at larger batch sizes, but limited by GPU RAM
    MAX_BATCH_SIZE = 16

    init_start = time.monotonic_ns()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(HF_MODEL_DIR, MODEL_ID))
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    # and then we add it from the left, to minimize impact on the output
    # tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id
    print(f"pad {tokenizer.pad_token} {pad_id}")
    print(f"pad {tokenizer.eos_token} {end_id}")

    questions = [
        # Generic assistant questions
        "What are you?",
        "What can you do?",
        # Coding
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        "Implement a Python class for a doubly linked list.",
        "Write a Haskell function that generates prime numbers using the Sieve of Eratosthenes.",
        "Develop a simple HTTP server in Rust.",
        # Literate and creative writing
        "What is the fable involving a fox and grapes?",
        "Who does Harry turn into a balloon?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083 to see robots in the beautiful desert.",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        "Write a poem in the style of Walt Whitman about the modern digital world.",
        "Create a short story about a society where people can only speak in metaphors.",
        "What are the main themes in Dostoevsky's 'Crime and Punishment'?",
        # History and Philosophy
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        "What led to the rise and fall of the Mongol Empire?",
        "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
        "How did the Treaty of Versailles contribute to the outbreak of World War II?",
        "What led to the rise and fall of the Mongol Empire?",
        "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
        "How did the Treaty of Versailles contribute to the outbreak of World War II?",
        "Explain the concept of 'tabula rasa' in John Locke's philosophy.",
        "What does Nietzsche mean by 'ressentiment'?",
        "Compare and contrast the early and late works of Ludwig Wittgenstein. Which do you prefer?",
        "How does the trolley problem explore the ethics of decision-making in critical situations?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        "If you could design a school curriculum for the future, what subjects would you include to prepare students for the next 50 years?",
        "How would society change if teleportation was invented and widely accessible?",
        "Consider a future where artificial intelligence governs countries. What are the potential benefits and pitfalls?",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        "Which countries in the European Union use currencies other than the Euro, and what are those currencies?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
        "¿Cuáles son los principales impactos ambientales de la deforestación en la Amazonía?",
        "Décris la structure et le rôle de la mitochondrie dans une cellule.",
        "Какие были социальные последствия Перестройки в Советском Союзе?",
        # Economics and Business
        "What are the principles of behavioral economics and how do they influence consumer choices?",
        "Discuss the impact of blockchain technology on traditional banking systems.",
        "What are the long-term effects of trade wars on global economic stability?",
        "What is the law of supply and demand?",
        "Explain the concept of inflation and its typical causes.",
        "What is a trade deficit, and why does it matter?",
        "How do interest rates affect consumer spending and saving?",
        "What is GDP and why is it important for measuring economic health?",
        "What is the difference between revenue and profit?",
        "Describe the role of a business plan in startup success.",
        "How does market segmentation benefit a company?",
        "Explain the concept of brand equity.",
        "What are the advantages of franchising a business?",
        "What are Michael Porter's five forces and how do they impact strategy for tech startups?",
        # Science and Technology
        "Discuss the potential impacts of quantum computing on data security.",
        "How could CRISPR technology change the future of medical treatments?",
        "Explain the significance of graphene in the development of future electronics.",
        "How do renewable energy sources compare to fossil fuels in terms of environmental impact?",
        "What are the most promising technologies for carbon capture and storage?",
        "Explain why the sky is blue.",
        "What is the principle behind the operation of a microwave oven?",
        "How does Newton's third law apply to rocket propulsion?",
        "What causes iron to rust?",
        "Describe the process of photosynthesis in simple terms.",
        "What is the role of a catalyst in a chemical reaction?",
        "What is the basic structure of a DNA molecule?",
        "How do vaccines work to protect the body from disease?",
        "Explain the significance of mitosis in cellular reproduction.",
        "What are tectonic plates and how do they affect earthquakes?",
        "How does the greenhouse effect contribute to global warming?",
        "Describe the water cycle and its importance to Earth's climate.",
        "What causes the phases of the Moon?",
        "How do black holes form?",
        "Explain the significance of the Big Bang theory.",
        "What is the function of the CPU in a computer system?",
        "Explain the difference between RAM and ROM.",
        "How does a solid-state drive (SSD) differ from a hard disk drive (HDD)?",
        "What role does the motherboard play in a computer system?",
        "Describe the purpose and function of a GPU.",
        "What is TensorRT? What role does it play in neural network inference?",
    ]

    prefixes = [
        "Hi! ",
        "Hello! ",
        "Hi. ",
        "Hello. ",
        "Hi: ",
        "Hello: ",
        "Greetings. ",
    ]
    # prepending any string that causes a tokenization change is enough to invalidate KV cache
    for ii, prefix in enumerate(prefixes):
        questions += [prefix + question for question in questions[:128]]

    prompt_cases = [
        {
            # "prompts": ["hi, my name is", "hello, my name is"], # test kv cache
            # "prompts": questions, # test batch size
            "prompts": random.sample(questions, MAX_BATCH_SIZE),
            "kwargs": {"max_new_tokens": 10, "stop_words_list": None},
        },
    ]

    # load tensorrt engine
    # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.from_dir
    engine = ModelRunner.from_dir(
        engine_dir=os.path.join(TRT_MODEL_DIR, "qwen2.5-0.5B", "trt_engines_bfloat16"),
        rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        max_output_len=100,
        debug_mode=False,
    )

    init_duration_s = (time.monotonic_ns() - init_start) / 1e9
    print(f"ModelRunner init duration: {init_duration_s:.2f} s")

    for i, case in enumerate(prompt_cases):
        # https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/runtime/generation.html#SamplingConfig
        sampling_config = SamplingConfig(
            end_id=end_id,
            pad_id=pad_id,
            temperature=0.8,
            repetition_penalty=1.1,
            max_new_tokens=50,
            top_k=20,
            top_p=0.95,
            # min_p version > 0.17.0
            # min_p=0.0,
            stop_words_list=None,
        )
        sampling_config.update(**case["kwargs"])
        print("sampling_config:", sampling_config)
        tokens = tokenizer(case["prompts"], return_tensors="pt", padding=True, truncation=False)
        input_ids = tokens["input_ids"]
        print("input_ids:", input_ids.shape)

        num_prompts = input_ids.shape[0]
        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}")

        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.generate
        generator = engine.generate(
            input_ids,
            streaming=True,
            sampling_config=sampling_config,
        )
        first = True
        start_time = time.perf_counter()
        for output in generator:
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate batch_size {output.shape[0]} TTFT time: {ttft} s")
                first = False
            print("raw output", output.shape)
            output = output[:, 0]
            print("output", output.shape)
            # mask = output != pad_id
            # output = output[mask]
            outputs_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            assert len(outputs_text) == output.shape[0]
            print(output.shape, outputs_text)
        batch_stream_generate_cost = time.perf_counter() - start_time
        print(
            f"stream generate batch_size {output.shape[0]} cost time: {batch_stream_generate_cost} s"
        )


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=achatbot_trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_CACHE_DIR: trt_model_cache_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_achatbot_generator():
    import uuid
    import os
    import asyncio
    import time

    from achatbot.core.llm.tensorrt_llm.generator import (
        TrtLLMGenerator,
        TensorRTLLMEngineArgs,
        LMGenerateArgs,
        LlmArgs,
    )
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from transformers import AutoTokenizer, GenerationConfig

    model = os.path.join(HF_MODEL_DIR, MODEL_ID)
    generator = TrtLLMGenerator(
        **TensorRTLLMEngineArgs(serv_args=LlmArgs(model=model).to_dict()).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    generation_config = {}
    if os.path.exists(os.path.join(model, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model, "generation_config.json"
        ).to_dict()
    # async def run():
    prompt_cases = [
        {"prompt": "hello, my name is", "kwargs": {"max_new_tokens": 30, "stop_ids": []}},
        {
            "prompt": "hello, my name is",
            "kwargs": {"max_new_tokens": 30, "stop_ids": [13]},
        },  # prefill cache token test (default no cache)
        {"prompt": "hello, what your name?", "kwargs": {"max_new_tokens": 30, "stop_ids": [13]}},
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
        print("gen_kwargs:", gen_kwargs)
        first = True
        start_time = time.perf_counter()
        gen_texts = ""
        async for token_id in generator.generate(session, **gen_kwargs):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            gen_texts += gen_text
            print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)
    # asyncio.run(run())


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L4"),
    cpu=2.0,
    retries=0,
    image=achatbot_trtllm_image,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        TRT_MODEL_DIR: trt_model_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
async def run_achatbot_runner_generator():
    import uuid
    import os
    import asyncio
    import time

    from achatbot.core.llm.tensorrt_llm.generator import (
        TrtLLMRunnerGenerator,
        TensorRTLLMRunnerArgs,
        TensorRTLLMRunnerEngineArgs,
        LMGenerateArgs,
    )
    from achatbot.common.types import SessionCtx
    from achatbot.common.session import Session
    from transformers import AutoTokenizer, GenerationConfig

    model = os.path.join(HF_MODEL_DIR, MODEL_ID)
    engine_dir = os.path.join(TRT_MODEL_DIR, "qwen2.5-0.5B", "trt_engines_bfloat16")
    generator = TrtLLMRunnerGenerator(
        **TensorRTLLMRunnerEngineArgs(
            serv_args=TensorRTLLMRunnerArgs(engine_dir=engine_dir).__dict__
        ).__dict__,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    generation_config = {}
    if os.path.exists(os.path.join(model, "generation_config.json")):
        generation_config = GenerationConfig.from_pretrained(
            model, "generation_config.json"
        ).to_dict()
    # async def run():
    prompt_cases = [
        {"prompt": "hello, my name is", "kwargs": {"max_new_tokens": 30, "stop_ids": []}},
        {
            "prompt": "hello, my name is",
            "kwargs": {"max_new_tokens": 30, "stop_ids": [13]},
        },  # prefill cache token test (default no cache)
        {"prompt": "hello, what your name?", "kwargs": {"max_new_tokens": 30, "stop_ids": [13]}},
    ]

    # test the same session
    # session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    for case in prompt_cases:
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
        tokens = tokenizer(case["prompt"])
        session.ctx.state["token_ids"] = tokens["input_ids"]
        # gen_kwargs = {**generation_config, **case["kwargs"], **tokens}
        gen_kwargs = {**case["kwargs"], **tokens}
        print("gen_kwargs:", gen_kwargs)
        first = True
        start_time = time.perf_counter()
        gen_texts = ""
        async for token_id in generator.generate(session, **gen_kwargs):
            if first:
                ttft = time.perf_counter() - start_time
                print(f"generate TTFT time: {ttft} s")
                first = False
            gen_text = tokenizer.decode(token_id)
            gen_texts += gen_text
            print(session.ctx.client_id, token_id, gen_text)
        print(session.ctx.client_id, gen_texts)
    # asyncio.run(run())


"""
# llmapi (LLM load hf model, convert to tensorrt, build tensorrt engine, load tensorrt engine)
modal run src/llm/trtllm/examples/generator.py::run_sync (no stream | batch prompts processing | throughput)
modal run src/llm/trtllm/examples/generator.py::run_async_stream (stream | single prompt async processing | latency)
modal run src/llm/trtllm/examples/generator.py::run_async_batch_stream (stream | batch prompts async processing | latency+throughput) (multiple prompts/request or one prompt/request)

# runner (load tensorrt engine to run generate)
modal run src/llm/trtllm/examples/generator.py::runner_stream
modal run src/llm/trtllm/examples/generator.py::runner_batch_stream

# achatbot
modal run src/llm/trtllm/examples/generator.py::run_achatbot_generator
modal run src/llm/trtllm/examples/generator.py::run_achatbot_runner_generator

"""
