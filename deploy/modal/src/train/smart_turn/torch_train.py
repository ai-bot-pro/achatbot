import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import Wav2Vec2Processor

from logger import log, log_model_structure, log_dataset_statistics, log_dependencies
from torch_model import Wav2Vec2ForEndpointingTorch
from data_preprocess import prepare_datasets

import modal

IMAGE_GPU = os.getenv("IMAGE_GPU", "L40s")

# Define Modal stub and volume.
app = modal.App("endpointing-training")
TRAINING_DATA_DIR = "/data"
volume = modal.Volume.from_name("endpointing-training-data", create_if_missing=True)

torch.multiprocessing.set_sharing_strategy("file_system")


# Define Modal image with required dependencies.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchaudio",
        "transformers[torch]==4.48.2",
        "datasets==3.2.0",
        "scikit-learn==1.6.1",
        "seaborn",
        "matplotlib",
        "numpy",
        "librosa",
        "soundfile",
        "wandb",
    )
    .env(
        {
            "DDP_BACKEND": "nccl",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NPROC_PER_NODE": IMAGE_GPU.split(":")[-1] if len(IMAGE_GPU.split(":")) > 1 else "1",
        }
    )
    .add_local_python_source("logger", "torch_model", "data_preprocess")
)

# Hyperparameters and configuration
CONFIG = {
    "model_name": "facebook/wav2vec2-base-960h",
    "datasets_training": [
        "pipecat-ai/rime_2",
        "pipecat-ai/human_5_all",
        "pipecat-ai/human_convcollector_1",
        "pipecat-ai/orpheus_grammar_1",
        "pipecat-ai/orpheus_midfiller_1",
        "pipecat-ai/orpheus_endfiller_1",
        "pipecat-ai/chirp3_1",
    ],
    "datasets_test": [],  # e.g. "/data/datasets/human_5_filler"
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "train_batch_size": 30,
    "eval_batch_size": 64,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,
    "eval_steps": 500,
    "logging_steps": 100,
    "early_stopping_patience": 5,
    "output_dir": TRAINING_DATA_DIR,
}


def process_predictions(logits):
    """Converts raw logits into squeezed probability predictions and binary predictions."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits).squeeze()
    preds = (probs > 0.5).int()

    # Ensure output is numpy for sklearn metrics
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    return probs, preds


def compute_metrics(eval_pred):
    """Computes and returns a dictionary of metrics."""
    logits, labels = eval_pred
    probs, preds = process_predictions(logits)

    # Ensure confusion matrix is for binary classification
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def get_predictions(model, dataloader, device):
    """Get model predictions for a given dataset."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch.pop("labels").to(device)
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_values", "attention_mask"]
            }
            outputs = model(**inputs)
            logits = outputs["logits"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate_and_log(model, dataloader, device, split_name, output_dir, step=None):
    """Evaluate model, log metrics and plots to wandb."""
    log.info(f"Evaluating on {split_name} set...")
    logits, labels = get_predictions(model, dataloader, device)
    metrics = compute_metrics((logits.numpy(), labels.numpy()))

    log_metrics = {f"{split_name}/{k}": v for k, v in metrics.items()}
    if step:
        log_metrics["train/global_step"] = step

    wandb.log(log_metrics)
    log.info(f"Metrics for {split_name}: {metrics}")

    # Plotting
    probs, preds = process_predictions(logits)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels.numpy(), preds, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Incomplete", "Complete"],
        yticklabels=["Incomplete", "Complete"],
    )
    plt.title(f"Confusion Matrix - {split_name.capitalize()} Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, alpha=0.5, label="Probability of Complete")
    plt.title(f"Distribution of Completion Probabilities - {split_name.capitalize()} Set")
    plt.xlabel("Probability of Complete")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    prob_dist_path = os.path.join(plots_dir, f"probability_distribution_{split_name}.png")
    plt.savefig(prob_dist_path)
    plt.close()

    wandb.log(
        {
            f"{split_name}/confusion_matrix": wandb.Image(cm_path),
            f"{split_name}/probability_distribution": wandb.Image(prob_dist_path),
        }
    )

    return metrics


@app.function(
    image=image,
    gpu=IMAGE_GPU,
    memory=16384,
    cpu=10.0,
    volumes={TRAINING_DATA_DIR: volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("achatbot")],
)
def train():
    """Main training function."""
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY environment variable not set. Wandb logging will fail.")
        return

    log_dependencies()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"torch-v2-linearclassifier-{now}"
    output_dir = os.path.join(CONFIG["output_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    processor = Wav2Vec2Processor.from_pretrained(CONFIG["model_name"])
    model = Wav2Vec2ForEndpointingTorch().to(device)
    log_model_structure(model, CONFIG)

    datasets = prepare_datasets(processor, CONFIG)
    train_dataset, eval_dataset = datasets["training"], datasets["eval"]
    test_datasets = datasets["test"]

    wandb.init(project="speech-endpointing", name=run_name, config=CONFIG)
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")
    wandb.define_metric("final_eval/*", step_metric="train/global_step")
    for name in test_datasets.keys():
        wandb.define_metric(f"final_test_{name}/*", step_metric="train/global_step")

    log_dataset_statistics("training", train_dataset)
    log_dataset_statistics("eval", eval_dataset)
    for name, ds in test_datasets.items():
        log_dataset_statistics(f"test_{name}", ds)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=CONFIG["eval_batch_size"], num_workers=0, pin_memory=True
    )
    test_dataloaders = {
        name: DataLoader(ds, batch_size=CONFIG["eval_batch_size"], num_workers=0, pin_memory=True)
        for name, ds in test_datasets.items()
    }

    optimizer = AdamW(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )

    total_steps = len(train_dataloader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-7, end_factor=1.0, total_iters=warmup_steps
    )
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
    )

    global_step = 0
    best_eval_f1 = -1
    evals_no_improve = 0
    stop_training = False

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        for batch in progress_bar:
            optimizer.zero_grad()

            labels = batch.pop("labels").to(device)
            inputs = {
                k: v.to(device) for k, v in batch.items() if k in ["input_values", "attention_mask"]
            }

            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1

            if global_step % CONFIG["logging_steps"] == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                    }
                )
                progress_bar.set_postfix(loss=loss.item())

            if global_step % CONFIG["eval_steps"] == 0:
                eval_metrics = evaluate_and_log(
                    model, eval_dataloader, device, "eval", output_dir, step=global_step
                )

                if eval_metrics["f1"] > best_eval_f1:
                    best_eval_f1 = eval_metrics["f1"]
                    evals_no_improve = 0
                    best_model_dir = os.path.join(output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin")
                    )
                    processor.save_pretrained(best_model_dir)
                    log.info(
                        f"New best model saved to {best_model_dir} with F1: {best_eval_f1:.4f}"
                    )
                else:
                    evals_no_improve += 1

                if evals_no_improve >= CONFIG["early_stopping_patience"]:
                    log.info(
                        f"Early stopping triggered after {CONFIG['early_stopping_patience']} evaluations without improvement."
                    )
                    stop_training = True
                    break
                model.train()
        if stop_training:
            break

    log.info("Training finished. Loading best model for final evaluation.")
    best_model_dir = os.path.join(output_dir, "best_model")
    best_model_weights = os.path.join(best_model_dir, "pytorch_model.bin")

    if not os.path.exists(best_model_weights):
        log.warning("No best model was saved. Using the last model state for final evaluation.")
        best_model = model
    else:
        best_model = Wav2Vec2ForEndpointingTorch().to(device)
        best_model.load_state_dict(torch.load(best_model_weights, map_location=device))

    evaluate_and_log(best_model, eval_dataloader, device, "final_eval", output_dir)
    for name, dataloader in test_dataloaders.items():
        evaluate_and_log(best_model, dataloader, device, f"final_test_{name}", output_dir)

    final_save_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_save_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(final_save_dir, "pytorch_model.bin"))
    processor.save_pretrained(final_save_dir)
    log.info(f"Final model saved to {final_save_dir}")

    wandb.finish()


"""
modal run src/train/smart_turn/torch_train.py

IMAGE_GPU=A100 modal run src/train/smart_turn/torch_train.py
"""


@app.local_entrypoint()
def main():
    train.remote()
