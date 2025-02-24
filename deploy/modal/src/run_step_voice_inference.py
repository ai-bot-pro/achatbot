import os
import modal

app = modal.App("step-voice-inference")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "sox")
    .run_commands(
        "git clone https://github.com/weedge/Step-Audio.git -b feat/dev",
        "cd Step-Audio && pip install -r requirements.txt",
        "pip install -U rotary_embedding_torch",
        "pip install hdbscan",
        # "cd Step-Audio && huggingface-cli download stepfun-ai/Step-Audio-Tokenizer --quie --local-dir MODEL_DIR/stepfun-ai/Step-Audio-Tokenizer",
        # "cd Step-Audio && huggingface-cli download stepfun-ai/Step-Audio-TTS-3B --quie --local-dir MODEL_DIR/stepfun-ai/Step-Audio-TTS-3B",
        # "ls -lh MODEL_DIR/stepfun-ai",
    )
    .pip_install()
)

MODEL_DIR = "/root/models"
ASSETS_DIR = "/root/assets"
model_dir = modal.Volume.from_name("models", create_if_missing=True)
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40S:8"),
    retries=3,
    image=inference_image,
    volumes={MODEL_DIR: model_dir, ASSETS_DIR: assets_dir},
)
def voice_inference(text: str) -> str:
    import os
    import sys
    import torchaudio

    sys.path.insert(1, "/Step-Audio")

    from stepaudio import StepAudio

    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    for repo_id in [
        "stepfun-ai/Step-Audio-Tokenizer",
        "stepfun-ai/Step-Audio-TTS-3B",
        "stepfun-ai/Step-Audio-Chat",
    ]:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns="*",
            local_dir=os.path.join(MODEL_DIR, repo_id),
        )
        print(f"{repo_id} model to dir:{MODEL_DIR} done")

    model_path = os.path.join(MODEL_DIR, "stepfun-ai")
    model = StepAudio(
        tokenizer_path=f"{model_path}/Step-Audio-Tokenizer",
        tts_path=f"{model_path}/Step-Audio-TTS-3B",
        llm_path=f"{model_path}/Step-Audio-Chat",
    )
    # example for text input
    text = "你好，我是你的朋友，我叫小明，你叫什么名字？" if text == "" else text
    text, audio, sr = model(
        [{"role": "user", "content": text}],
        "Tingting",
    )
    print(text)
    torchaudio.save(os.path.join(ASSETS_DIR, "output_e2e_tqta.wav"), audio, sr)

    # example for audio input
    text, audio, sr = model(
        [
            {
                "role": "user",
                "content": {
                    "type": "audio",
                    "audio": os.path.join(ASSETS_DIR, "output_e2e_tqta.wav"),
                },
            }
        ],
        "Tingting",
    )
    print(text)
    torchaudio.save(os.path.join(ASSETS_DIR, "output_e2e_aqta.wav"), audio, sr)


@app.local_entrypoint()
def main(text: str = ""):
    voice_inference.remote(text)
