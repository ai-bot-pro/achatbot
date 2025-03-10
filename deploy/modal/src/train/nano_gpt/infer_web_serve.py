import os
import logging as L
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PosixPath

import modal
from pydantic import BaseModel

app = modal.App("infer-nano-gpt")

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("pydantic==2.9.1")
    .run_commands(
        "git clone https://github.com/modal-labs/modal-examples.git /modal_examples",
    )
)

web_image = base_image.pip_install("fastapi[standard]==0.115.4", "starlette==0.41.2")

# ### Adding a simple web endpoint

# The `ModelInference` class above is available for use
# from any other Python environment with the right Modal credentials
# and the `modal` package installed -- just use [`lookup`](https://modal.com/docs/reference/modal.Cls#lookup).

# But we can also expose it as a web endpoint for easy access
# from anywhere, including other programming languages or the command line.


class GenerationRequest(BaseModel):
    prompt: str


# ```bash
# curl -X POST -H 'Content-Type: application/json' --data-binary '{"prompt": "\n"}' https://your-workspace-name--modal-nano-gpt-web-generate.modal.run
# ```

# which will return something like:

# ```json
# {
# "output":
#    "BRUTUS:
#     The broy trefore anny pleasory to
#     wip me state of villoor so:
#     Fortols listhey for brother beat the else
#     Be all, ill of lo-love in igham;
#     Ah, here all that queen and hould you father offer"
# }
# ```


@app.function(image=web_image)
@modal.fastapi_endpoint(method="POST", docs=True)  # v0.73.82
def web_generate(request: GenerationRequest):
    output = ModelInference().generate.remote(request.prompt)
    return {"output": output}


volume = modal.Volume.from_name("train-slm-volume", create_if_missing=True)
volume_path = PosixPath("/data")
best_model_filename = "best_nano_gpt_model.pt"
model_save_path = volume_path / "models"

torch_image = base_image.pip_install(
    "torch==2.1.2",
)

with torch_image.imports():
    import glob
    import os
    import sys

    import torch

    sys.path.insert(1, "/modal_examples/06_gpu_and_ml/hyperparameter-sweep")
    from src.model import AttentionModel
    from src.tokenizer import Tokenizer


@app.cls(
    image=torch_image,
    volumes={volume_path: volume},
    gpu=os.getenv("IMAGE_GPU", "T4"),
)
class ModelInference:
    experiment_name: str = modal.parameter(default="")

    def get_latest_available_model_dirs(self, n_last):
        """Find the latest models that have a best model checkpoint saved."""
        save_model_dirs = glob.glob(f"{model_save_path}/*")
        sorted_model_dirs = sorted(save_model_dirs, key=os.path.getctime, reverse=True)

        valid_model_dirs = []
        for latest_model_dir in sorted_model_dirs:
            if Path(f"{latest_model_dir}/{best_model_filename}").exists():
                valid_model_dirs.append(Path(latest_model_dir))
            if len(valid_model_dirs) >= n_last:
                return valid_model_dirs
        return valid_model_dirs

    @modal.method()
    def get_latest_available_experiment_names(self, n_last):
        return [d.name for d in self.get_latest_available_model_dirs(n_last)]

    def load_model_impl(self):
        if self.experiment_name != "":  # user selected model
            use_model_dir = f"{model_save_path}/{self.experiment_name}"
        else:  # otherwise, pick latest
            try:
                use_model_dir = self.get_latest_available_model_dirs(1)[0]
            except IndexError:
                raise ValueError("No models available to load.")

        if self.use_model_dir == use_model_dir and self.is_fully_trained:
            return  # already loaded fully trained model.

        ckpt_path = f"{use_model_dir}/{best_model_filename}"
        if not os.path.exists(ckpt_path):
            print(f"{ckpt_path} does not exist.")
            raise ValueError(f"{ckpt_path} does not exist.")

        print(f"Loading experiment: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

        self.use_model_dir = use_model_dir
        hparams = checkpoint["hparams"]
        key = (  # for backwards compatibility
            "unique_chars" if "unique_chars" in checkpoint else "chars"
        )
        unique_chars = checkpoint[key]
        steps = checkpoint["steps"]
        val_loss = checkpoint["val_loss"]
        self.is_fully_trained = checkpoint["finished_training"]

        print(
            f"Loaded model with {steps} train steps"
            f" and val loss of {val_loss:.2f}"
            f" (fully_trained={self.is_fully_trained})"
        )

        self.tokenizer = Tokenizer(unique_chars)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AttentionModel(self.tokenizer.vocab_size, hparams, self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)

    @modal.enter()
    def load_model(self):
        self.use_model_dir = None
        self.is_fully_trained = False
        self.load_model_impl()

    @modal.method()
    def generate(self, prompt):
        self.load_model_impl()  # load updated model if available

        n_new_tokens = 1000
        return self.model.generate_from_text(self.tokenizer, prompt, n_new_tokens)
