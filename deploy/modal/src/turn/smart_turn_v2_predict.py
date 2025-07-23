import os
import subprocess
import asyncio

import modal

APP_NAME = os.getenv("APP_NAME", "achatbot")
TAG = os.getenv("TURN_ANALYZER_TAG", "v2_smart_turn_analyzer")

app = modal.App("smart_turn_v2_predict")
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake", "ninja-build")
    .pip_install("wheel")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "transformers",
    )
    # https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TQDM_DISABLE": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", ""),
            "TURN_ANALYZER_TAG": TAG,
        }
    )
)

if APP_NAME == "achatbot":
    img = img.pip_install(
        f"achatbot==0.0.22.dev0",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)


with img.imports():
    import torch


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=1,
    image=img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(gpu_prop)
    else:
        func(gpu_prop)


async def achatbot_predict_end(gpu_prop):
    import os

    from achatbot.common.logger import Logger
    from achatbot.common.interface import EndOfTurnState, ITurnAnalyzer
    from achatbot.modules.speech.turn_analyzer import TurnAnalyzerEnvInit

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    tag = os.getenv("TURN_ANALYZER_TAG", "v2_smart_turn_analyzer")
    kwargs = {
        "model_path": os.path.join(HF_MODEL_DIR, "pipecat-ai/smart-turn-v2"),
        "torch_dtype": "float32",
    }
    turn: ITurnAnalyzer = TurnAnalyzerEnvInit.initTurnAnalyzerEngine(tag, kwargs)

    assert turn.speech_triggered is False

    audio_bytes = b""
    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    state = turn.append_audio(audio_bytes, True)
    # print(state)
    assert turn.speech_triggered is True
    assert state == EndOfTurnState.INCOMPLETE

    state, result = await turn.analyze_end_of_turn()
    # print(state, result)
    assert turn.speech_triggered is False
    assert state == EndOfTurnState.COMPLETE
    assert result["prediction"] == 1

    turn.clear()
    assert turn.speech_triggered is False


async def achatbot_predict_no_end(gpu_prop):
    import os

    import wave
    from pydub import AudioSegment

    from achatbot.common.logger import Logger
    from achatbot.common.interface import EndOfTurnState, ITurnAnalyzer
    from achatbot.modules.speech.turn_analyzer import TurnAnalyzerEnvInit

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

    tag = os.getenv("TURN_ANALYZER_TAG", "v2_smart_turn_analyzer")
    kwargs = {
        "model_path": os.path.join(HF_MODEL_DIR, "pipecat-ai/smart-turn-v2"),
        "torch_dtype": "float32",
    }
    turn: ITurnAnalyzer = TurnAnalyzerEnvInit.initTurnAnalyzerEngine(tag, kwargs)

    assert turn.speech_triggered is False

    audio_file = os.path.join(ASSETS_DIR, "asr_example_zh.wav")
    audio = AudioSegment.from_file(audio_file, format="wav")

    total_duration = len(audio)
    duration_to_capture = int(total_duration * 3 / 4)
    audio_clip = audio[:duration_to_capture]

    tmp_path = os.path.join(ASSETS_DIR, "turn_temp_clip.wav")
    audio_clip.export(tmp_path, format="wav")
    print(f"Exported temp clip to {tmp_path}")
    assert os.path.exists(tmp_path)

    with wave.open(tmp_path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)

        print(f"Channels: {n_channels}")
        print(f"Sample Width: {sample_width}")
        print(f"Frame Rate: {frame_rate}")
        print(f"Number of Frames: {n_frames}")
        print(f"Audio Data length: {len(audio_bytes)}")

        state = turn.append_audio(audio_bytes, True)
        # print(state)
        assert turn.speech_triggered is True
        assert state == EndOfTurnState.INCOMPLETE

        state, result = await turn.analyze_end_of_turn()
        print(state, result)
        assert turn.speech_triggered is True
        assert state == EndOfTurnState.INCOMPLETE
        assert result["prediction"] != 1

        turn.clear()
        assert turn.speech_triggered is False


"""
# just test GPU with torch  to optimize performance
# use L4 to detect 5s chinese audio cost 35 ms
IMAGE_GPU=L4 modal run src/turn/smart_turn_v2_predict.py --task achatbot_predict_end 
IMAGE_GPU=L4 modal run src/turn/smart_turn_v2_predict.py --task achatbot_predict_no_end 
# use L40s to detect 5s chinese audio cost 15 ms
IMAGE_GPU=L40s modal run src/turn/smart_turn_v2_predict.py --task achatbot_predict_end 
IMAGE_GPU=L40s modal run src/turn/smart_turn_v2_predict.py --task achatbot_predict_no_end 
"""


@app.local_entrypoint()
def main(task: str = "achatbot_predict_end"):
    print(task)
    tasks = {
        "achatbot_predict_end": achatbot_predict_end,
        "achatbot_predict_no_end": achatbot_predict_no_end,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
