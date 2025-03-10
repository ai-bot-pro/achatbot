import os
from pathlib import Path, PosixPath

import modal

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

ui_image = web_image.pip_install("gradio~=4.44.0")

volume = modal.Volume.from_name("train-slm-volume", create_if_missing=True)
volume_path = PosixPath("/data")
best_model_filename = "best_nano_gpt_model.pt"
model_save_path = volume_path / "models"


# The Gradio UI
@app.function(
    image=ui_image,
    max_containers=1,
    volumes={volume_path: volume},
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    # call out to the inference in a separate Modal environment with a GPU
    def generate(text="", experiment_name=""):
        if not text:
            text = "\n"
        generated = ModelInference(experiment_name=experiment_name).generate.remote(text)
        return text + generated

    example_prompts = [
        "DUKE OF YORK:\nWhere art thou Lucas?",
        "ROMEO:\nWhat is a man?",
        "CLARENCE:\nFair is foul and foul is fair, but who are you?",
        "Brevity is the soul of wit, so what is the soul of foolishness?",
    ]

    web_app = FastAPI()

    # custom styles: an icon, a background, and a theme
    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/modal_examples/06_gpu_and_ml/hyperparameter-sweep/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse(
            "/modal_examples/06_gpu_and_ml/hyperparameter-sweep/assets/background.svg"
        )

    with open("/modal_examples/06_gpu_and_ml/hyperparameter-sweep/assets/index.css") as f:
        css = f.read()

    n_last = 20
    experiment_names = ModelInference().get_latest_available_experiment_names.remote(n_last)
    theme = gr.themes.Default(primary_hue="green", secondary_hue="emerald", neutral_hue="neutral")

    # add a Gradio UI around inference
    with gr.Blocks(theme=theme, css=css, title="SLM") as interface:
        # title
        gr.Markdown("# GPT-style Shakespeare text generation.")

        # Model Selection
        with gr.Row():
            gr.Markdown("## Model Version")
        with gr.Row():
            experiment_dropdown = gr.Dropdown(experiment_names, label="Select Model Version")

        # input and output
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input:")
                input_box = gr.Textbox(  # input text component
                    label="",
                    placeholder="Write some Shakespeare like text or keep it empty!",
                    lines=10,
                )
            with gr.Column():
                gr.Markdown("## Output:")
                output_box = gr.Textbox(  # output text component
                    label="",
                    lines=10,
                )

        # button to trigger inference and a link to Modal
        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary", scale=2)
            generate_button.click(
                fn=generate,
                inputs=[input_box, experiment_dropdown],
                outputs=output_box,
            )  # connect inputs and outputs with inference function

            gr.Button(  # shameless plug
                " Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        # example prompts
        with gr.Column(variant="compact"):
            # add in a few examples to inspire users
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=input_box)

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


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

        print(f"Loading experiment: {Path(use_model_dir).name}...")
        checkpoint = torch.load(f"{use_model_dir}/{best_model_filename}")

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
