import os
from typing import Any, Dict, List

import modal

app = modal.App("download_datasets")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("datasets", "transformers", "jinja2")
    .env(
        {
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            "DATA_PATH": os.getenv("DATA_PATH", "Jiayi-Pan/Countdown-Tasks-3to4"),
        }
    )
)

HF_DATASET_DIR = "/datasets"
hf_dataset_vol = modal.Volume.from_name("datasets", create_if_missing=True)
HF_MODEL_DIR = "/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)


with download_image.imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    import datasets


@app.function(
    # gpu="T4",
    retries=0,
    cpu=8.0,
    image=download_image,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_DATASET_DIR: hf_dataset_vol,
        HF_MODEL_DIR: hf_model_vol,
    },
    timeout=1200,
)
def process(task: str = "w", read_data_type: str = "train"):
    data_path = os.getenv("DATA_PATH", "Jiayi-Pan/Countdown-Tasks-3to4")
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    LLM_CHAT_MODEL_PATH = os.path.join(HF_MODEL_DIR, llm_model)
    DATA_PATH = os.path.join(HF_DATASET_DIR, data_path)

    tokenizer = AutoTokenizer.from_pretrained(LLM_CHAT_MODEL_PATH)

    if task == "r":
        parquet_file = os.path.join(DATA_PATH, f"trl_{read_data_type}.parquet")
        read(parquet_file)
        return

    ############################################
    # Prompts and Dataset
    ############################################

    SYSTEM_MESSAGE = (
        "You are a helpful assistant. You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    PROMPT_TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final equation and answer in "
        "<answer> </answer> tags, for example <answer>(1 + 2) / 3 = 1</answer>."
    )

    dataset = datasets.load_dataset(DATA_PATH, split="train")
    dataset = dataset.map(
        generate_r1_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
        num_proc=8,
    )

    # Split dataset
    # train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"{train_dataset[0]=}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"{test_dataset[0]=}")

    train_dataset.to_parquet(os.path.join(DATA_PATH, "trl_train.parquet"))
    test_dataset.to_parquet(os.path.join(DATA_PATH, "trl_test.parquet"))


# Load and process dataset
def generate_r1_prompt(
    item: Dict[str, Any],
    tokenizer: "AutoTokenizer",
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = item["nums"]
    target: int = item["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    prompt = tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)
    # input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    # prompt = tokenizer.decode(
    #    input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )
    return {"prompt": prompt, "target": target}


def read(parquet_file: str):
    data = datasets.load_dataset("parquet", data_files=parquet_file)
    print(f"read data {data['train'][0]}")


"""
# use IT chat template tokenizer
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B-Instruct"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-1.5B-Instruct"
modal run src/download_models.py --repo-ids "Qwen/Qwen2.5-3B-Instruct"

# download datasets
modal run src/download_datasets.py --repo-ids "Jiayi-Pan/Countdown-Tasks-3to4"

# data preporcess
modal run src/train/trl/data_preprocess/countdown.py
modal run src/train/trl/data_preprocess/countdown.py --read-data-type test --task r
modal run src/train/trl/data_preprocess/countdown.py --read-data-type train --task r

LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/data_preprocess/countdown.py
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/data_preprocess/countdown.py --read-data-type test --task r
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct modal run src/train/trl/data_preprocess/countdown.py --read-data-type train --task r

LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/data_preprocess/countdown.py
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/data_preprocess/countdown.py --read-data-type test --task r
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct modal run src/train/trl/data_preprocess/countdown.py --read-data-type train --task r
"""
