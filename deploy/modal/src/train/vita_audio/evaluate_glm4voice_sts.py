import datetime
import os
import subprocess
import argparse
import itertools
import json
import random
import sys
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

import modal


app = modal.App("vita_audio")
IMAGE_GPU = os.getenv("IMAGE_GPU", "A100-80GB")

vita_audio_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .run_commands(
        "git clone -b feat/achatbot https://github.com/weedge/VITA-Audio.git",
        "cd /VITA-Audio && git submodule update --init --recursive",
        "cd /VITA-Audio && pip install -q -r requirements_ds_gpu.txt",
    )
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install("wheel")
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("sentence_transformers", "jieba", "scikit-learn")
    .run_commands(
        "cd /VITA-Audio && git pull origin feat/achatbot",
        "cd /VITA-Audio && git checkout 0a5d6c30dcd587acb8bcafb18f24d3af10a07d52",
    )
    .env(
        {
            # RuntimeError: The kernel on this machine does not support the pidfd_open syscall needed to use IPC for CUDA tensors when expandable_segments:True is set. Consider using expandable_segments:False via torch.cuda.memory._set_allocator_settings('expandable_segments:False') for this allocation.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "LLM_MODEL": os.getenv("LLM_MODEL", "finetune_glm4voice_mtp10_stage2"),
            "AUDIO_ENCODE_MODEL": os.getenv("AUDIO_ENCODE_MODEL", "THUDM/glm-4-voice-tokenizer"),
            "AUDIO_DECODE_MODEL": os.getenv("AUDIO_DECODE_MODEL", "THUDM/glm-4-voice-decoder"),
            "JSON_PATH": os.getenv("JSON_PATH", "/VITA-Audio/asset/eval_sts.jsonl"),
        }
    )
)

EVALUATE_NAME = "glm4voice_qwen2mtp_sts"

HF_MODEL_DIR = "/data/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
TRAIN_OUTPUT_DIR = "/train_output"
train_out_dir = modal.Volume.from_name("train_output", create_if_missing=True)
EVALUATE_OUTPUT_DIR = "/evaluate_output"
evaluate_out_dir = modal.Volume.from_name("evaluate_output", create_if_missing=True)
DATASETS_DIR = "/datasets"
datasets_dir = modal.Volume.from_name("datasets", create_if_missing=True)

with vita_audio_img.imports():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    sys.path.append("/VITA-Audio")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/cosyvoice/")
    sys.path.append("/VITA-Audio/third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    from vita_audio.tokenizer import get_audio_tokenizer
    from evaluation.get_chat_template import qwen2_chat_template as chat_template
    from evaluation.evaluate_sqa import (
        collate_fn,
        STSDataset,
        load_asr_model,
        inference,
    )
    from evaluation.compute_acc_of_contain import evaluate

    # --------------- init ----------------------
    AUDIO_TOKENIZER_TYPE = "glm4voice"
    AUDIO_TOKENIZER_MODEL_PATH = os.path.join(
        HF_MODEL_DIR, os.getenv("AUDIO_ENCODE_MODEL", "THUDM/glm-4-voice-tokenizer")
    )
    FLOW_MODEL_PATH = os.path.join(
        HF_MODEL_DIR, os.getenv("AUDIO_DECODE_MODEL", "THUDM/glm-4-voice-decoder")
    )
    LLM_MODEL = os.getenv("LLM_MODEL", "finetune_glm4voice_mtp10_stage2")
    MODEL_NAME_OR_PATH = os.path.join(HF_MODEL_DIR, LLM_MODEL)
    if LLM_MODEL.startswith("finetune_"):
        MODEL_NAME_OR_PATH = os.path.join(TRAIN_OUTPUT_DIR, LLM_MODEL)

    cur_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    OUTPUT_DIR = os.path.join(EVALUATE_OUTPUT_DIR, f"{EVALUATE_NAME}")
    PRE_OUTPUT_DIR = os.path.join(EVALUATE_OUTPUT_DIR, f"{EVALUATE_NAME}_{cur_date}")
    if os.path.exists(OUTPUT_DIR):
        cmd = f"mv {OUTPUT_DIR} {PRE_OUTPUT_DIR}"
        print(cmd)
        subprocess.run(cmd, shell=True)

    JSON_PATH = os.getenv("JSON_PATH", "/VITA-Audio/asset/eval_sts.jsonl")
    if JSON_PATH.startswith("VITA-MLLM/VITA-Audio-Data"):
        JSON_PATH = os.path.join(DATASETS_DIR, JSON_PATH)
    JSON_NAME = Path(os.path.normpath(JSON_PATH).split(os.sep)[-1]).stem

    os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=1,
    image=vita_audio_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
        TRAIN_OUTPUT_DIR: train_out_dir,
        EVALUATE_OUTPUT_DIR: evaluate_out_dir,
        DATASETS_DIR: datasets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
def run():
    # --------------- run ----------------------
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    print(f"Model config: {config}")

    default_system_message = [
        {
            "role": "system",
            # "content": "Your Name: Luke\nYour Gender: male\nRespond in a text-audio interleaved manner.",
            # "content": "Your Name: Lucy\nYour Gender: female\nRespond in a text-audio interleaved manner.",
            "content": "Your Name: Omni\nYour Gender: female\nRespond in a text-audio interleaved manner.",
        },
    ]

    # ================================================================
    print("Loading model")
    # device_map = "auto"
    device_map = "cuda"
    torch_dtype = torch.bfloat16

    audio_tokenizer = get_audio_tokenizer(
        AUDIO_TOKENIZER_MODEL_PATH, AUDIO_TOKENIZER_TYPE, flow_path=FLOW_MODEL_PATH, rank=0
    )
    audio_tokenizer.load_model()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        chat_template=chat_template,
    )
    # print("tokenizer", tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).eval()
    # print("model", model)

    model.generation_config = GenerationConfig.from_pretrained(
        MODEL_NAME_OR_PATH, trust_remote_code=True
    )

    model.generation_config.max_new_tokens = 4096
    model.generation_config.chat_format = "chatml"
    model.generation_config.max_window_size = 8192
    model.generation_config.use_cache = True
    # model.generation_config.temperature = 0
    # model.generation_config.top_p = None
    # model.generation_config.top_k = None
    model.generation_config.do_sample = True if model.generation_config.temperature > 0 else False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    asr_model = load_asr_model(0)

    # ================================================================
    print("Loading data")
    dataset = STSDataset(
        json_path=JSON_PATH,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        default_system_message=default_system_message,
        add_generation_prompt=True,
    )

    # 数据集需要提前进行切分， 不需要reduce
    # 不使用torchrun dist 在运行时处理切分; 而是独立运行进程(容器化)去评估
    # 也可部署在线推理评估, 可量化评估
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn),
    )

    # ================================================================
    outputs = inference(
        model, tokenizer, audio_tokenizer, dataloader, EVALUATE_OUTPUT_DIR, asr_model
    )

    write_res(outputs)

    # sentence similarity
    # TF-IDF余弦相似度 | Jaccard相似度 | Sequence相似度
    # 基于 bert embedding (sbert)
    evaluate(
        os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_ref_text.json"),
        calculate_transformer_similarity,
        verbose=True,
    )
    evaluate(
        os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_ref_speech.json"),
        calculate_transformer_similarity,
        verbose=True,
    )

    evaluate_out_dir.commit()


def write_res(outputs):
    # JSON_NAME = Path("_".join(os.path.normpath(args.json_path).split(os.sep)[-2:])).stem
    hyp_text_path = os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_text.txt")
    hyp_speech_path = os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_speech.txt")
    ref_path = os.path.join(OUTPUT_DIR, f"{JSON_NAME}_ref.txt")

    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    os.makedirs(os.path.dirname(hyp_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(hyp_speech_path), exist_ok=True)

    hyp_text_file = open(hyp_text_path, "w")
    hyp_speech_file = open(hyp_speech_path, "w")
    ref_file = open(ref_path, "w")

    for sample_idx, (hyp_text, hyp_speech, ref) in enumerate(outputs):
        hyp_text_file.write(f"{sample_idx} {hyp_text}" + "\n")
        hyp_speech_file.write(f"{sample_idx} {hyp_speech}" + "\n")
        ref_file.write(f"{sample_idx} {ref}" + "\n")

    hyp_text_file.close()
    hyp_speech_file.close()
    ref_file.close()

    outputs_speech = [[x[1], x[2]] for x in outputs]
    outputs_text = [[x[0], x[2]] for x in outputs]

    hyp_ref_path = os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_ref_text.json")
    hyp_ref_file = open(hyp_ref_path, "w")
    json.dump(outputs_text, hyp_ref_file, indent=4)
    hyp_ref_file.close()

    hyp_ref_path = os.path.join(OUTPUT_DIR, f"{JSON_NAME}_hyp_ref_speech.json")
    hyp_ref_file = open(hyp_ref_path, "w")
    json.dump(outputs_speech, hyp_ref_file, indent=4)
    hyp_ref_file.close()


"""
# eval training model finetune_glm4voice_mtp10_stage2
IMAGE_GPU=L40s modal run src/train/vita_audio/evaluate_glm4voice_sts.py

# eval  VITA-MLLM/VITA-Audio-Boost
IMAGE_GPU=L40s LLM_MODEL=VITA-MLLM/VITA-Audio-Boost modal run src/train/vita_audio/evaluate_glm4voice_sts.py
# eval  VITA-MLLM/VITA-Audio-Balance
IMAGE_GPU=L40s LLM_MODEL=VITA-MLLM/VITA-Audio-Balance modal run src/train/vita_audio/evaluate_glm4voice_sts.py
"""


def evaluate(pred_gt_json_file, func, verbose=False):
    with open(pred_gt_json_file, "r") as f:
        pred_gt = json.load(f)

    for line in pred_gt:
        pred = line[0].lower()
        gt = line[1].lower()
        sim = func(pred, gt)
        if verbose:
            print(f"{func=} {sim=}")


def calculate_cosine_similarity(text1, text2):
    """使用TF-IDF和余弦相似度计算文本相似度"""
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 对文本进行分词
    text1_cut = " ".join(jieba.cut(text1))
    text2_cut = " ".join(jieba.cut(text2))

    # 创建TF-IDF向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1_cut, text2_cut])

    # 计算余弦相似度
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_jaccard_similarity(text1, text2):
    """计算Jaccard相似度"""
    import jieba

    # 对文本进行分词
    set1 = set(jieba.cut(text1))
    set2 = set(jieba.cut(text2))

    # 计算交集和并集
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0


def calculate_sequence_similarity(text1, text2):
    """使用SequenceMatcher计算文本相似度"""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, text1, text2).ratio()


def calculate_transformer_similarity(text1, text2):
    """使用Sentence Transformers计算文本相似度"""
    from sentence_transformers import SentenceTransformer

    model_path = os.path.join(HF_MODEL_DIR, "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_path)

    # 计算文本嵌入
    embeddings = model.encode([text1, text2])

    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(
        torch.FloatTensor(embeddings[0]).unsqueeze(0),
        torch.FloatTensor(embeddings[1]).unsqueeze(0),
    )

    return similarity.item()
