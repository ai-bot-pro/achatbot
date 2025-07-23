import os
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk


def _create_preprocess_function(processor):
    """Creates a preprocessing function that captures the processor."""

    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        labels = [1 if lb else 0 for lb in batch["endpoint_bool"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(labels)
        return inputs

    return preprocess_function


def load_dataset_at(path: str):
    """Loads a dataset from a local path or the Hugging Face Hub."""
    if os.path.isdir(path):
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]


def validate_audio_lengths(dataset, dataset_name):
    """Validate that all audio samples are between 0 and 16 seconds"""
    for i, sample in enumerate(dataset):
        audio_array = sample["audio"]["array"]
        duration = len(audio_array) / 16000
        if duration <= 0:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' has zero or negative length ({duration} seconds)"
            )
        if duration > 16:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' exceeds 16 seconds limit ({duration} seconds)"
            )


def prepare_datasets(processor, config):
    """
    Loads, splits, preprocesses, and organizes datasets based on config settings.

    Args:
        processor: A Hugging Face processor for the audio data.
        config: A dictionary containing dataset configurations.

    Returns:
        A dictionary with "training", "eval", and "test" datasets.
    """
    datasets_training = config["datasets_training"]
    datasets_test = config["datasets_test"]

    overlap = set(datasets_training).intersection(set(datasets_test))
    if overlap:
        raise ValueError(f"Found overlapping datasets in training and test: {overlap}")

    training_splits, eval_splits, test_splits = [], [], {}
    for dataset_path in datasets_training:
        dataset_name = dataset_path.split("/")[-1]
        full_dataset = load_dataset_at(dataset_path)
        validate_audio_lengths(full_dataset, dataset_name)
        dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
        training_splits.append(dataset_dict["train"])
        eval_test_dict = dataset_dict["test"].train_test_split(test_size=0.5, seed=42)
        eval_splits.append(eval_test_dict["train"])
        test_splits[dataset_name] = eval_test_dict["test"]

    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)

    for dataset_path in datasets_test:
        dataset_name = dataset_path.split("/")[-1]
        test_dataset = load_dataset_at(dataset_path)
        validate_audio_lengths(test_dataset, dataset_name)
        test_splits[dataset_name] = test_dataset

    preprocess_function = _create_preprocess_function(processor)

    def apply_preprocessing(dataset):
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=8,
            remove_columns=["audio", "endpoint_bool"],
            num_proc=4,
        )
        processed_dataset.set_format(
            type="torch", columns=["input_values", "attention_mask", "labels"]
        )
        return processed_dataset

    merged_training_dataset = apply_preprocessing(merged_training_dataset)
    merged_eval_dataset = apply_preprocessing(merged_eval_dataset)
    for dataset_name, dataset in test_splits.items():
        test_splits[dataset_name] = apply_preprocessing(dataset)

    return {"training": merged_training_dataset, "eval": merged_eval_dataset, "test": test_splits}
