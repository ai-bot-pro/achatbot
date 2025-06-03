import os
import re
import modal

app = modal.App("download_datasets")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

download_image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("curl").pip_install("datasets")
)

HF_DATASET_DIR = "/datasets"
hf_dataset_vol = modal.Volume.from_name("datasets", create_if_missing=True)


@app.function(
    # gpu="T4",
    retries=0,
    cpu=8.0,
    image=download_image,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={HF_DATASET_DIR: hf_dataset_vol},
    timeout=1200,
)
def process():
    import datasets

    data_source = "openai/gsm8k"
    data_source_path = os.path.join(HF_DATASET_DIR, data_source)

    dataset = datasets.load_dataset(data_source_path, "main")

    train_dataset = dataset["train"]
    print("raw train dataset", train_dataset[0])
    test_dataset = dataset["test"]
    print("raw test dataset", test_dataset[0])

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    print(train_dataset[0],flush=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(test_dataset[0],flush=True)

    local_dir = data_source_path

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    # print(final_solution)
    return final_solution


"""
modal run src/train/verl/examples/data_preprocess/gsm8k.py
"""


@app.local_entrypoint()
def main():
    process.remote()
