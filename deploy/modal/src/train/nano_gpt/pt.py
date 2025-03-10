import os
import logging as L
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PosixPath

import modal
from pydantic import BaseModel

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app = modal.App("train-nano-gpt")

# We'll use A10G GPUs for training, which are able to train the model to recognizably improved performance
# in ~15 minutes while keeping costs under ~$1.
gpu = os.getenv("IMAGE_GPU", "A10G")

volume = modal.Volume.from_name("train-slm-volume", create_if_missing=True)
volume_path = PosixPath("/data")
model_filename = "nano_gpt_model.pt"
best_model_filename = "best_nano_gpt_model.pt"
tb_log_path = volume_path / "tb_logs"
model_save_path = volume_path / "models"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("pydantic==2.9.1")
    .run_commands(
        "git clone https://github.com/modal-labs/modal-examples.git /modal_examples",
    )
)

train_image = base_image.pip_install(
    "torch==2.1.2",
    "numpy<2",
)

with train_image.imports():
    import os
    import sys
    from timeit import default_timer as timer

    import torch

    sys.path.insert(1, "/modal_examples/06_gpu_and_ml/hyperparameter-sweep")

    from src.dataset import Dataset
    from src.logs_manager import LogsManager
    from src.model import AttentionModel
    from src.tokenizer import Tokenizer


@app.function(
    image=train_image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=1 * HOURS,
)
def train_model(
    node_rank,
    n_nodes,
    hparams,
    experiment_name,
    run_to_first_save=False,
    n_steps=3000,
    n_steps_before_eval=None,
    n_steps_before_checkpoint=None,
):
    # optimizer, data, and model prep
    batch_size = 64
    learning_rate = 3e-4

    n_eval_steps = 100
    if n_steps_before_eval is None:
        n_steps_before_eval = int(n_steps / 8)  # eval eight times per run
    if n_steps_before_checkpoint is None:
        n_steps_before_checkpoint = int(n_steps / 4)  # save four times per run

    train_percent = 0.9

    L.basicConfig(
        level=L.INFO,
        format=f"\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] [Node {node_rank + 1}] %(message)s\033[0m",
        datefmt="%b %d %H:%M:%S",
    )

    # use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.info("Remote Device: %s // GPU: %s", device, gpu)

    input_file_path = volume_path / "shakespeare_char.txt"
    text = prepare_data(input_file_path, volume)

    # construct tokenizer & dataset
    tokenizer = Tokenizer(text)
    dataset = Dataset(
        tokenizer.encode(text),
        train_percent,
        batch_size,
        hparams.context_size,
        device,
    )

    # build the model
    model = build_model(hparams, tokenizer.vocab_size, device)
    num_parameters = sum(p.numel() for p in model.parameters())
    L.info(f"Num parameters: {num_parameters}")

    optimizer = setup_optimizer(model, learning_rate)

    # TensorBoard logging & checkpointing prep
    logs_manager = LogsManager(experiment_name, hparams, num_parameters, tb_log_path)
    L.info(f"Model name: {logs_manager.model_name}")

    model_save_dir = model_save_path / experiment_name / logs_manager.model_name
    if model_save_dir.exists():
        L.info("Loading model from checkpoint...")
        checkpoint = torch.load(str(model_save_dir / model_filename))
        is_best_model = not run_to_first_save
        if is_best_model:
            make_best_symbolic_link(model_save_dir, model_filename, experiment_name)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint["steps"] + 1
    else:
        model_save_dir.mkdir(parents=True, exist_ok=True)
        start_step = 0
        checkpoint = init_checkpoint(model, tokenizer, optimizer, start_step, hparams)

    checkpoint_path = model_save_dir / model_filename

    out = training_loop(
        start_step,
        n_steps,
        n_steps_before_eval,
        n_steps_before_checkpoint,
        n_eval_steps,
        dataset,
        tokenizer,
        model,
        optimizer,
        logs_manager,
        checkpoint,
        checkpoint_path,
        run_to_first_save,
    )

    return node_rank, float(out["val"]), hparams


# ## Launch a hyperparameter sweep from a `local_entrypoint`

# The main entry point coordinates the hyperparameter optimization.
# First we specify the default hyperparameters for the model, taken from
# [Andrej Karpathy's walkthrough](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5976s).
# For better performance, you can increase the `context_size` and scale up the GPU accordingly.


@dataclass
class ModelHyperparameters:
    n_heads: int = 6
    n_embed: int = 384  # hidden_size
    n_blocks: int = 6  # batch_size
    context_size: int = 256  # seq_len
    dropout: float = 0.2


"""
modal run src/train/nano_gpt/pt.py --n-steps 20 --n-steps-before-checkpoint 10 --n-steps-before-eval 10
modal run src/train/nano_gpt/pt.py
"""


@app.local_entrypoint()
def main(
    n_steps: int = 3000,
    n_steps_before_checkpoint: int = None,
    n_steps_before_eval: int = None,
):
    from datetime import datetime
    from itertools import product

    experiment_name = f"E{datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')}"
    default_hparams = ModelHyperparameters()

    # build list of hyperparameters to train & validate
    # 增加训练超参数目， 可以构建多组实验，根据不同的实验配置组，并行执行单独的训练，
    # 从而快速筛选出训练loss值低的训练参数,
    # 然后单独进行后续的模型训练
    nheads_options = (1, default_hparams.n_heads)
    context_size_options = (8, default_hparams.context_size)
    dropout_options = (0.1, default_hparams.dropout)

    hparams_list = [
        ModelHyperparameters(n_heads=h, context_size=c, dropout=d)
        for h, c, d in product(nheads_options, context_size_options, dropout_options)
    ]

    # run training for each hyperparameter setting
    results = []
    stop_early = True  # stop early so we can compare val losses
    print(f"Testing {len(hparams_list)} hyperparameter settings")
    n_nodes = len(hparams_list)
    static_params = (
        experiment_name,
        stop_early,
        n_steps,
        n_steps_before_eval,
        n_steps_before_checkpoint,
    )
    # https://modal.com/docs/guide/scale
    for result in train_model.starmap(
        [(i, n_nodes, h, *static_params) for i, h in enumerate(hparams_list)],
        order_outputs=False,
    ):
        # result = (node_rank, val_loss, hparams)
        node_rank = result[0]
        results.append(result)
        print(
            f"[Node {node_rank + 1}/{n_nodes}] Finished. Early stop val loss result: {result[1:]}"
        )

    # find the model and hparams with the lowest validation loss
    best_result = min(results, key=lambda x: x[1])
    print(f"Best early stop val loss result: {best_result}")
    best_hparams = best_result[-1]

    # finish training with best hparams
    node_rank = 0
    n_nodes = 1  # only one node for final training run
    train_model.remote(
        node_rank,
        n_nodes,
        best_hparams,
        experiment_name,
        not stop_early,
        n_steps,
        n_steps_before_eval,
        n_steps_before_checkpoint,
    )


# ## Addenda

# The remainder of this code is boilerplate.

# ### Training Loop

# There's quite a lot of code for just the training loop! If you'd rather not write this stuff yourself,
# consider a training framework like [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable)
# or [Hugging Face](https://huggingface.co/transformers/main_classes/trainer.html).


def training_loop(
    start_step,
    n_steps,
    n_steps_before_eval,
    n_steps_before_checkpoint,
    n_eval_steps,
    dataset,
    tokenizer,
    model,
    optimizer,
    logs_manager,
    checkpoint,
    checkpoint_path,
    run_to_first_save,
):
    @torch.no_grad()
    def eval_model(model, dataset, tokenizer, n_eval_steps):
        """Evaluate model on train and validation data."""
        out = {}
        model.eval()  # Turn off gradients
        for split in ("train", "val"):
            losses = torch.zeros(n_eval_steps)
            for k in range(n_eval_steps):
                xb, yb = dataset.get_batch(split)
                logits, loss = model.forward(xb, yb)
                losses[k] = loss
            out[split] = losses.mean()

        # Generate some output samples
        out["sample"] = model.generate_from_text(tokenizer, "\n", 1000)

        model.train()  # Turn on gradients
        return out

    t_last = timer()
    for step in range(start_step, n_steps + 1):
        # sample a batch of data
        xb, yb = dataset.get_batch("train")

        # evaluate the loss, calculate & apply gradients
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log training loss
        logs_manager.add_train_scalar("Cross Entropy Loss", loss.item(), step)

        # evaluate model on validation set
        if step % n_steps_before_eval == 0:
            out = eval_model(model, dataset, tokenizer, n_eval_steps)
            log_evals(out, step, t_last, logs_manager)
            t_last = timer()

        # save model with checkpoint information
        if step > 0 and step % n_steps_before_checkpoint == 0:
            checkpoint["steps"] = step
            checkpoint["val_loss"] = out["val"]

            # mark as finished if we hit n steps.
            checkpoint["finished_training"] = step >= n_steps

            L.info(f"Saving checkpoint to {checkpoint_path}\t {checkpoint['finished_training']})")
            save_checkpoint(checkpoint, checkpoint_path)

            if run_to_first_save:
                L.info("Stopping early...")
                break
    return out


def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)
    volume.commit()


def build_model(hparams, vocab_size, device):
    """Initialize the model and move it to the device."""
    model = AttentionModel(vocab_size, hparams, device)
    model.to(device)
    return model


def setup_optimizer(model, learning_rate):
    """Set up the optimizer for the model."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ### Miscellaneous
# The remaining code includes small helper functions for training the model.


def prepare_data(input_file_path: Path, volume: modal.Volume) -> str:
    """Download and read the dataset."""
    volume.reload()
    if not input_file_path.exists():
        L.info("Downloading Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(data_url, input_file_path)
        volume.commit()
    return input_file_path.read_text()


def make_best_symbolic_link(model_save_dir, model_filename, experiment_name):
    # create symlink to the best model so it's easy to find for web serving
    os.symlink(
        str(model_save_dir / model_filename),
        str(model_save_path / experiment_name / best_model_filename),
    )
    volume.commit()  # commit the symlink


def init_checkpoint(model, tokenizer, optimizer, start_step, hparams):
    return {
        "model": model.state_dict(),
        "unique_chars": tokenizer.unique_chars,
        "optimizer": optimizer.state_dict(),
        "val_loss": float("inf"),
        "steps": start_step,
        "hparams": hparams,
        "finished_training": False,
    }


def log_evals(result, step, t_last, logs_manager):
    runtime_s = timer() - t_last
    L.info(
        f"{step:5d}) // {runtime_s:>5.2f}s // Train Loss: {result['train']:.2f} // Val Loss: {result['val']:.2f}"
    )
    logs_manager.add_val_scalar("Cross Entropy Loss", result["val"], step)
    logs_manager.add_val_text("Sample Output", result["sample"], step)
    logs_manager.flush()
    volume.commit()  # Make sure TensorBoard container will see it.

    return result
