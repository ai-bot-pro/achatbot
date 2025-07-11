# author: weedge (weege007@gmail.com)

import os
import subprocess
from threading import Thread
from time import perf_counter
from typing import Optional

import modal


app = modal.App("ERNIE4.5-VL")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
    )
    .pip_install("wheel", "packaging")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .run_commands(
        "pip install git+https://github.com/huggingface/transformers@17b3c96c00cd8421bff85282aec32422bdfebd31"
    )
    .pip_install("accelerate", "av")
    .pip_install("decord", "moviepy", "sentencepiece")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "baidu/ERNIE-4_5-VL-28B-A3B-PT"),
            "ROUND": os.getenv("ROUND", "1"),
            "IS_OUTPUT_THINK": os.getenv("IS_OUTPUT_THINK", "1"),
            "IMAGE_GPU": os.getenv("IMAGE_GPU", None),
        }
    )
)

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
VIDEO_OUTPUT_DIR = "/gen_video"
video_out_vol = modal.Volume.from_name("gen_video", create_if_missing=True)


with img.imports():
    import torch
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

    from transformers.generation.streamers import TextIteratorStreamer


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=0,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        VIDEO_OUTPUT_DIR: video_out_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run(func, thinking):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    func(gpu_prop, thinking)


def print_model_params(model: torch.nn.Module, extra_info=""):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(f"{extra_info} {model_million_params} M parameters")


"""
Ernie4_5_VLMoeForConditionalGeneration(
  (model): Ernie4_5_Model(
    (embed_tokens): Embedding(103424, 2560)
    (layers): ModuleList(
      (0): Ernie4_5_DecoderLayer(
        (self_attn): Ernie4_5_Attention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=512, bias=False)
          (v_proj): Linear(in_features=2560, out_features=512, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): RopeEmbedding()
        )
        (mlp): Ernie4_5_MLP(
          (gate_proj): Linear(in_features=2560, out_features=12288, bias=False)
          (up_proj): Linear(in_features=2560, out_features=12288, bias=False)
          (down_proj): Linear(in_features=12288, out_features=2560, bias=False)
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
        (residual_add1): FusedDropoutImpl(
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (residual_add2): FusedDropoutImpl(
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
      (1-27): 27 x Ernie4_5_DecoderLayer(
        (self_attn): Ernie4_5_Attention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=512, bias=False)
          (v_proj): Linear(in_features=2560, out_features=512, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): RopeEmbedding()
        )
        (mlp): MOEAllGatherLayerV2(
          (gate): TopKGate()
          (experts): ModuleList(
            (0-63): 64 x Ernie4_5_MoeMLP(
              (gate_proj): Linear(in_features=2560, out_features=1536, bias=False)
              (up_proj): Linear(in_features=2560, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=2560, bias=False)
            )
            (64-127): 64 x Ernie4_5_MoeMLP(
              (gate_proj): Linear(in_features=2560, out_features=512, bias=False)
              (up_proj): Linear(in_features=2560, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2560, bias=False)
            )
          )
          (shared_experts): Ernie4_5_MoeMLP(
            (gate_proj): Linear(in_features=2560, out_features=3072, bias=False)
            (up_proj): Linear(in_features=2560, out_features=3072, bias=False)
            (down_proj): Linear(in_features=3072, out_features=2560, bias=False)
          )
          (moe_statics): MoEStatics()
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
        (residual_add1): FusedDropoutImpl(
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (residual_add2): FusedDropoutImpl(
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (norm): RMSNorm()
    (resampler_model): VariableResolutionResamplerModel(
      (spatial_linear): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=5120, bias=True)
        (3): LayerNorm((5120,), eps=1e-06, elementwise_affine=True)
      )
      (temporal_linear): Sequential(
        (0): Linear(in_features=10240, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=5120, bias=True)
        (3): LayerNorm((5120,), eps=1e-06, elementwise_affine=True)
      )
      (mlp): Linear(in_features=5120, out_features=2560, bias=True)
      (after_norm): RMSNorm()
    )
  )
  (lm_head): Linear(in_features=2560, out_features=103424, bias=False)
  (vision_model): DFNRopeVisionTransformerPreTrainedModel(
    (patch_embed): PatchEmbed(
      (proj): Linear(in_features=588, out_features=1280, bias=False)
    )
    (rotary_pos_emb): VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x DFNRopeVisionBlock(
        (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (attn): VisionAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): VisionMlp(
          (fc1): Linear(in_features=1280, out_features=5120, bias=True)
          (act): QuickGELUActivation()
          (fc2): Linear(in_features=5120, out_features=1280, bias=True)
        )
      )
    )
    (ln): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
  )
)
"""


def split_model(model_name, gpu):
    device_map = {}

    # splits layers into different GPUs (need use L4/L40s for bfloat16)
    model_splits = {
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_L4:4": {
            "model": [7, 7, 7, 7],  # 28 layer
            "vision_model": [8, 8, 8, 8],  # 32 layer
        },
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_L40s:2": {
            "model": [11, 17],  # 28 layer
            "vision_model": [14, 18],  # 32 layer
        },
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_A100-80GB": {
            "model": [28],  # 28 layer
            "vision_model": [32],  # 32 layer
        },
    }
    device_map["lm_head"] = 0

    num_layers_per_gpu = model_splits[model_name + "_" + gpu]["model"]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    # exlude layer and last layer on cuda 0
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 0
    device_map["model.resampler_model"] = 0
    device_map[f"model.layers.{num_layers - 1}"] = 0

    num_layers_per_gpu = model_splits[model_name + "_" + gpu]["vision_model"]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"vision_model.blocks.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model.patch_embed"] = 0
    device_map["vision_model.rotary_pos_emb"] = 0
    device_map["vision_model.ln"] = 0
    device_map[f"vision_model.blocks.{num_layers - 1}"] = 0

    return device_map


def test(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.eval()
    model.add_image_preprocess(processor)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                    },
                },
            ],
        },
    ]

    text = f"请用中文描述图片内容，不要使用特殊字符回复。"
    # don't to chat with smolvlm, just do vision task
    # text = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 你好"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    temperature = 0.6
    generated_ids = model.generate(
        inputs=inputs["input_ids"].to(device),
        **inputs,
        max_new_tokens=128,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    output_text = processor.decode(generated_ids[0])
    print(output_text)


def dump_model(gpu_prop, thinking):
    """
    vlm text model use ERNIE4.5 arch no MTP
    """
    for model_name in [
        "baidu/ERNIE-4_5-VL-28B-A3B-PT",
    ]:
        MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"{type(processor)=}", processor)
        processor.eval()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
            trust_remote_code=True,
            device_map="auto",
        )
        model.add_image_preprocess(processor)
        model = model.eval()
        print(f"{model.config=}")
        print_model_params(model, f"{model_name}")

        del model
        torch.cuda.empty_cache()


@torch.inference_mode()
def predict_text(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")
    model = model.eval()
    tokenizer = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)

    # predict with no instruct tpl, use base model | sft-it | rl-it
    text = "你叫什么名字？"
    inputs = tokenizer([text], images=None, videos=None, return_tensors="pt").to(model.device)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = tokenizer.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=128,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_text(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
    ).to("cuda")
    model = model.eval()
    tokenizer = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)

    # Construct prompt
    text = "你叫什么名字？"
    messages = [
        {
            "role": "system",
            "content": "你是一个非常棒的聊天助手，不要使用特殊字符回复。",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    # use instruct sft | rl model
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    inputs = tokenizer([prompt], images=None, videos=None, padding=True, return_tensors="pt").to(
        model.device
    )
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = tokenizer.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def predict(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    gpu = os.getenv("IMAGE_GPU")
    device_map = split_model(model_name, gpu)
    print(device_map)
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    processor.eval()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if gpu_prop.major >= 8 else None,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.add_image_preprocess(processor)
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct prompt
    # text = "请用中文描述图片内容"
    text = f"请用中文描述图片内容，不要使用特殊字符回复。"

    # don't to chat with smolvlm, just do vision task
    # text = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 你好"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                    },
                },
                {"type": "text", "text": text},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.no_grad()
def predict_stream(gpu_prop, thinking):
    model_name = os.getenv("LLM_MODEL")
    gpu = os.getenv("IMAGE_GPU")
    device_map = split_model(model_name, gpu)
    print(device_map)
    # Load model
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.eval()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.add_image_preprocess(processor)
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print(f"{model=}")

    # Construct prompt
    # text = "请用中文描述图片内容"
    text = f"请用中文描述图片内容，不要使用特殊字符回复。"
    # text = "Please reply to my message in Chinese simplified(简体中文), don't use Markdown format. 描述下图片内容"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": text},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    for i in range(3):
        print("---" * 20)
        streamer = TextIteratorStreamer(
            tokenizer=processor, skip_prompt=True, skip_special_tokens=True
        )
        temperature = 0.6
        generation_kwargs = dict(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=1024,
            use_cache=True,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        start = perf_counter()
        times = []
        is_output_think = os.getenv("IS_OUTPUT_THINK", "1") == "1"
        for new_text in streamer:
            times.append(perf_counter() - start)
            print(new_text, end="", flush=True)
            if is_output_think is False:
                if "</think>" in new_text:
                    new_text = new_text.replace("</think>", "").strip("\n")
                    is_output_think = True
                else:
                    continue
            generated_text += new_text
            start = perf_counter()
        print(f"\n{i}. {generated_text=} TTFT: {times[0]:.2f}s total time: {sum(times):.2f}s")


@torch.inference_mode()
def chat(gpu_prop, thinking):
    """
    multi images need more GPU, have some bug
    """
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.eval()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.add_image_preprocess(processor)
    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct history chat messages
    text = "讲一个故事"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": "请描述图片内容"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图中展示了一个撕裂效果的画面 外面是橙黄色的背景 内部展现了一位裹着彩色条纹毯子 红头发的背对观众的人物 她坐在开满粉红色花朵的山坡上 前方是连绵的山谷和山脉 远处天空中有阳光洒落",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": "图片中有几个人"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一个人。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generated_ids = model.generate(
        inputs=inputs["input_ids"].to(model.device),
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def text_vision_chat(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.eval()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.add_image_preprocess(processor)
    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # Construct history chat messages
    text = "讲一个故事"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个图片描述助手，你可以用中文描述图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "我是图片描述助手。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        # 高分辨率的图片需要更多的GPU BHM
                        # "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                    },
                },
                {"type": "text", "text": "图片中有几个人"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图片中有一个人。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    ]

    round = int(os.getenv("ROUND", "1")) * 2 if int(os.getenv("ROUND", "1")) > 0 else 2
    prompt = processor.apply_chat_template(
        messages[:round],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def text_video_chat(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    # process vision need add
    model.add_image_preprocess(processor)
    model = model.eval()
    # print(f"{model.config=}")
    # print(f"{processor=}")
    # print_model_params(model, f"{model_name}")

    # from meigen-multitalk generated video
    video_file = os.path.join(VIDEO_OUTPUT_DIR, "multi_long_exp.mp4")
    print(video_file)

    # Construct history chat messages
    text = "讲一个故事"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个视频描述助手，你可以用中文描述视频,图片内容，不要使用特殊字符回复。",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你叫什么名字"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "我是视频描述助手。",  # remove think content
                }
            ],
        },
        # {
        #    "role": "user",
        #    "content": [
        #        {
        #            "type": "image",
        #            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        #        },
        #        {"type": "text", "text": "图片中有几只蜜蜂"},
        #    ],
        # },
        # {
        #    "role": "assistant",
        #    "content": [
        #        {
        #            "type": "text",
        #            "text": "图片中有一只蜜蜂。",  # remove think content
        #        }
        #    ],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"
                    },
                },
                # {
                #    "type": "video",
                #    "video": video_file,
                # },
                {"type": "text", "text": "描述这个视频"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "画面展示了一个录音室场景，一位穿着浅灰色衬衫的男子和一位身着无袖亮片连衣裙的女子面对面站在专业麦克风前。两人身体微微前倾，靠近麦克风，面部表情专注且投入，似乎在共同演唱。录音室环境有黑色的隔音墙面（带有凸起吸音结构）和左侧的木质窗帘，整体氛围专业且温馨，呈现出两人合作演唱的画面。",  # remove think content
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    ]

    round = int(os.getenv("ROUND", "1")) * 2 if int(os.getenv("ROUND", "1")) > 0 else 2
    prompt = processor.apply_chat_template(
        messages[:round],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        # repetition_penalty=1.5,
        max_new_tokens=2048,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_tool(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    text = "北京的天气"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个智能助手，不要使用特殊字符回复。"
                    "\n 提供的工具如下：\n"
                    "\n Tool: get_weather \n"
                    "\n Description: Get weather of an location, the user shoud supply a location first \n"
                    "\n Arguments: location\n\n"
                    "\n 根据用户的问题选择合适的工具。如果不需要工具，直接回复。\n",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                # {
                #    "type": "image",
                #    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                # },
                {"type": "text", "text": text},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


@torch.inference_mode()
def chat_json_mode(gpu_prop, thinking):
    # Load model
    model_name = os.getenv("LLM_MODEL")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, model_name)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    model = model.eval()

    system_prompt = """
    The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 
    
    EXAMPLE INPUT: 
    Which is the highest mountain in the world? Mount Everest.
    
    EXAMPLE JSON OUTPUT:
    {
        "question": "Which is the highest mountain in the world?",
        "answer": "Mount Everest"
    }
    """
    text = "Which is the longest river in the world? The Nile River."
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                # {
                #    "type": "image",
                #    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                # },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    print(f"{prompt=}")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        [prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    for key, value in inputs.items():
        print(f"{key}: {value.shape=}") if isinstance(value, torch.Tensor) else print(
            f"{key}: {value}"
        )
    input_ids = inputs["input_ids"]
    prompt = processor.decode(input_ids[0])
    print(f"{prompt=}")

    temperature = 0.6
    generated_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )
    print(f"{generated_ids.shape=}")
    generated_text = processor.decode(
        generated_ids[0][len(input_ids[0]) :],
        # skip_special_tokens=True,
    )
    print(generated_text)


"""
https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT

TIPS:
- transformers的实现持flash-attention优化，推理速度很慢, 仅用作离线任务
- 可以使用官方的FastDeploy(like tensorrt_llm)部署Paddle格式模型进行推理(支持Wint4/Wint8 ERNIE-4.5-VL-28B-A3B-Paddle 24GB/48GB)：https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/quick_start_vl.md
- 支持auto-thinking, thinking, non-thinking, 由apply_chat_template `enable_thinking` 设置，分别对应：None,True,False
- 后续还会对vllm支持，会用到transformers的tokenizer

# 0. download model
modal run src/download_models.py --repo-ids "baidu/ERNIE-4.5-VL-28B-A3B-PT" --local-dir "baidu/ERNIE-4_5-VL-28B-A3B-PT"

# 1. dump model
IMAGE_GPU=L40s modal run src/llm/transformers/vlm/ernie4.py --task dump_model

# 2. text model
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task predict_text
IMAGE_GPU=A100-80GB LLM_MODEL=THUDM/GLM-4.1V-9B-Base modal run src/llm/transformers/vlm/ernie4.py --task predict_text
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task chat_text

# 3. vision text/image chat case need 68GB GPU HBM
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/ernie4.py --task predict 
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/ernie4.py --task predict --no-thinking
IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/ernie4.py --task predict --thinking
IMAGE_GPU=L40s:2 modal run src/llm/transformers/vlm/ernie4.py --task predict 
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task predict 
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task predict --no-thinking
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task predict --thinking

IMAGE_GPU=L4:4 modal run src/llm/transformers/vlm/ernie4.py --task predict_stream
IMAGE_GPU=L40s:2 modal run src/llm/transformers/vlm/ernie4.py --task predict_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task predict_stream

# multi images need more GPU HBM, have some bug
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task chat 

ROUND=1 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_vision_chat
ROUND=2 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_vision_chat
ROUND=3 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_vision_chat

# 4. vision text/video chat, need more GPU HBM, have some bug
ROUND=1 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_video_chat
ROUND=2 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_video_chat
ROUND=3 IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task text_video_chat

# 5. 不支持funciton_calling 需要微调
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task chat_tool

# 6. json_mode支持不够好, 输出markdown格式```json {xxx} ``` xxx 需要截断/微调
IMAGE_GPU=A100-80GB modal run src/llm/transformers/vlm/ernie4.py --task chat_json_mode
"""


@app.local_entrypoint()
def main(task: str = "dump_model", thinking: Optional[bool] = None):
    print(task, thinking)
    tasks = {
        "test": test,
        "dump_model": dump_model,
        "predict_text": predict_text,
        "chat_text": chat_text,
        "predict": predict,
        "predict_stream": predict_stream,
        "chat": chat,
        "text_vision_chat": text_vision_chat,
        "text_video_chat": text_video_chat,
        "chat_tool": chat_tool,
        "chat_json_mode": chat_json_mode,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task], thinking)
