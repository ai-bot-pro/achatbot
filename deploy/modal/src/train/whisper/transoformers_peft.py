import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import soundfile as sf
import torch
from datasets import Audio, Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        bos_mask = labels[:, 0] == self.processor.tokenizer.bos_token_id
        if bos_mask.all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def check_audio_file(file_path):
    try:
        with sf.SoundFile(file_path):
            return True
    except Exception:
        return False


def load_dataset(data_dir, json_file):
    with open(os.path.join(data_dir, json_file)) as f:
        transcriptions = json.load(f)

    valid_data = []
    for k, v in tqdm(transcriptions.items(), desc="Checking audio files"):
        file_path = os.path.join(data_dir, f"{k}.wav")
        if check_audio_file(file_path):
            valid_data.append({"audio": file_path, "sentence": v})

    dataset = Dataset.from_list(valid_data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def prepare_dataset(batch, processor, device):
    audio = batch["audio"]
    input_features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_features"] = torch.tensor(input_features).to(device)
    batch["labels"] = torch.tensor(processor.tokenizer(batch["sentence"]).input_ids).to(device)

    return batch


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.state.global_step % 50 == 0:
            self.loss_log.append((self.state.global_step, loss.item()))
            print(f"Step {self.state.global_step}: Loss = {loss.item()}")
        return loss


def main(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, language=args.language, task=args.task
    )

    dataset = load_dataset(args.data_dir, args.json_file)
    dataset = dataset.train_test_split(test_size=0.1)

    print("Preparing dataset...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor, device),
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        remove_unused_columns=False,
        label_names=["labels"],
        local_rank=rank,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Save loss log
    with open(os.path.join(args.output_dir, "loss_log.json"), "w") as f:
        json.dump(trainer.loss_log, f)

    cleanup()


def inference(args):
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    merged_model = peft_model.merge_and_unload()

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)

    audio_file = "path/to/your/audio/file.wav"
    audio_input = processor(audio_file, return_tensors="pt").input_features

    generated_ids = merged_model.generate(audio_input)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing wav files and json file"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="successful_texts.json",
        help="JSON file containing transcriptions",
    )
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--output_dir", type=str, default="./whisper_output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    elif args.mode == "inference":
        inference(args)
