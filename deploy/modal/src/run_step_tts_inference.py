import modal

app = modal.App("step-tts-inference")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "sox")
    .run_commands(
        "git clone https://github.com/weedge/Step-Audio.git -b feat/dev ",
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
    gpu="T4",
    retries=3,
    image=inference_image,
    volumes={MODEL_DIR: model_dir, ASSETS_DIR: assets_dir},
)
def tts_inference(synthesis_type: str, text: str = "") -> str:
    import os
    import sys
    import torchaudio

    sys.path.insert(1, "/Step-Audio")

    from tts import StepAudioTTS
    from tokenizer import StepAudioTokenizer

    # https://huggingface.co/docs/huggingface_hub/guides/download
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="stepfun-ai/Step-Audio-Tokenizer",
        repo_type="model",
        allow_patterns="*",
        local_dir=os.path.join(MODEL_DIR, "stepfun-ai/Step-Audio-Tokenizer"),
    )
    print(f"Tokenizer model to dir:{MODEL_DIR} done")
    snapshot_download(
        repo_id="stepfun-ai/Step-Audio-TTS-3B",
        repo_type="model",
        allow_patterns="*",
        local_dir=os.path.join(MODEL_DIR, "stepfun-ai/Step-Audio-TTS-3B"),
    )
    print(f"TTS model to dir:{MODEL_DIR} done")

    encoder = StepAudioTokenizer(f"{MODEL_DIR}/stepfun-ai/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(f"{MODEL_DIR}/stepfun-ai/Step-Audio-TTS-3B", encoder)

    if synthesis_type == "tts":
        tts_text = (
            "（RAP）我踏上自由的征途，追逐那遥远的梦想，挣脱束缚的枷锁，让心灵随风飘荡，每一步都充满力量，每一刻都无比闪亮，自由的信念在燃烧，照亮我前进的方向!"
            if not text
            else text
        )
        output_audio, sr = tts_engine(tts_text, "Tingting")
        torchaudio.save(f"{ASSETS_DIR}/output_tts.wav", output_audio, sr)
    else:
        clone_speaker = {
            "speaker": "test",
            "prompt_text": "叫做秋风起蟹脚痒，啊，什么意思呢？就是说这秋风一起啊，螃蟹就该上市了。",
            "wav_path": "examples/prompt_wav_yuqian.wav",
        }
        text_clone = (
            "人活一辈子，生老病死，总得是有高峰，有低谷，有顺境，有逆境，每个人都差不多。要不老话怎么讲，三十年河东，三十年河西呢。"
            if not text
            else text
        )
        output_audio, sr = tts_engine(text_clone, "", clone_speaker)
        torchaudio.save(f"{ASSETS_DIR}/output_clone.wav", output_audio, sr)


@app.local_entrypoint()
def main(synthesis_type: str = "tts", text: str = "你好"):
    tts_inference.remote(synthesis_type, text)
